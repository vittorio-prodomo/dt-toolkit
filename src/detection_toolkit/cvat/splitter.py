"""Split a CVAT task into a new task with different job boundaries."""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Returned by :meth:`TaskSplitter.run`."""

    new_task_id: int
    new_task_name: str
    jobs: list[dict]
    """List of ``{"job_id": int, "num_frames": int, "first_frame": str, "last_frame": str}``."""
    annotations_total: int


class TaskSplitter:
    """Create a new CVAT task by re-splitting an existing task's frames.

    Parameters
    ----------
    client : CvatClient
        Authenticated CVAT connection (from ``cvat.client``).
    task_id : int
        Source task to split.
    cuts : list[int]
        Frame indices where new jobs begin. For a 490-frame task,
        ``cuts=[250]`` produces two jobs: frames 0-249 and 250-489.
        ``cuts=[100, 300]`` produces three jobs: 0-99, 100-299, 300-489.
    job_id : int, optional
        If the source task has multiple jobs and you only want to split
        one of them, pass its ID here. Other jobs are preserved as-is.
    work_dir : Path, optional
        Temporary directory for backup files. Created automatically if
        omitted and cleaned up on success.
    """

    def __init__(
        self,
        client,
        task_id: int,
        cuts: List[int],
        *,
        job_id: Optional[int] = None,
        work_dir: Optional[Path] = None,
    ) -> None:
        self._sdk = client.sdk
        self._task_id = task_id
        self._cuts = sorted(cuts)
        self._job_id = job_id
        self._work_dir = Path(work_dir) if work_dir else None
        self._owns_work_dir = work_dir is None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> SplitResult:
        """Execute the split and return a :class:`SplitResult`."""
        work = self._work_dir or Path(mkdtemp(prefix="dt_cvat_split_"))
        work.mkdir(parents=True, exist_ok=True)

        try:
            return self._run(work)
        finally:
            if self._owns_work_dir and work.exists():
                shutil.rmtree(work, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, work: Path) -> SplitResult:
        backup_path = work / "original_backup.zip"
        modified_path = work / "modified_backup.zip"
        annotations_path = work / "annotations.zip"

        # 1. Fetch metadata
        task = self._sdk.tasks.retrieve(self._task_id)
        logger.info(
            "Source task %d: '%s', %d frames, %d job(s)",
            task.id, task.name, task.size, task.jobs.count,
        )

        all_jobs = task.get_jobs()
        frame_map = self._build_frame_map(all_jobs)

        # 2. Validate cuts
        self._validate(frame_map)

        # 3. Compute new job layout
        new_job_layout = self._compute_layout(frame_map)
        for i, jl in enumerate(new_job_layout):
            logger.info("  New job %d: %d frames", i, len(jl["files"]))

        # 4. Download backup
        logger.info("Downloading task backup...")
        task.download_backup(str(backup_path))
        logger.info(
            "  Downloaded: %.1f MB", backup_path.stat().st_size / (1024 * 1024)
        )

        # 5. Build modified backup
        logger.info("Building modified backup...")
        self._build_modified_backup(backup_path, modified_path, task, new_job_layout)

        # 6. Import as new task
        logger.info("Importing modified backup as new task...")
        new_task = self._sdk.tasks.create_from_backup(str(modified_path))
        logger.info("  Created task %d: '%s'", new_task.id, new_task.name)

        # 7. Export annotations from original and import into new task
        logger.info("Exporting annotations from original task...")
        task.export_dataset(
            format_name="CVAT for images 1.1",
            filename=str(annotations_path),
            include_images=False,
        )
        logger.info("Importing annotations into new task...")
        new_task.import_annotations(
            format_name="CVAT 1.1",
            filename=str(annotations_path),
        )

        # 8. Verify and build result
        return self._build_result(new_task)

    def _build_frame_map(
        self, all_jobs: Sequence,
    ) -> list[dict]:
        """Return per-job metadata: ``[{"job_id", "frames", "type"}]``."""
        entries = []
        for j in all_jobs:
            frames = j.get_frames_info()
            entries.append({
                "job_id": j.id,
                "type": j.type,
                "frame_names": [f.name for f in frames],
            })
        # Sort by job type (annotation jobs first) then by first frame name
        # to get a stable, logical order
        entries.sort(key=lambda e: (e["type"] != "annotation", e["frame_names"][0] if e["frame_names"] else ""))
        return entries

    def _validate(self, frame_map: list[dict]) -> None:
        """Validate cuts against the frame map."""
        if self._job_id is not None:
            target = [e for e in frame_map if e["job_id"] == self._job_id]
            if not target:
                raise ValueError(
                    f"Job {self._job_id} not found in task {self._task_id}. "
                    f"Available jobs: {[e['job_id'] for e in frame_map]}"
                )
            n_frames = len(target[0]["frame_names"])
        else:
            if len(frame_map) != 1:
                annotation_jobs = [e for e in frame_map if e["type"] == "annotation"]
                if len(annotation_jobs) != 1:
                    raise ValueError(
                        f"Task {self._task_id} has {len(annotation_jobs)} annotation jobs. "
                        f"Use --job to specify which one to split. "
                        f"Available: {[e['job_id'] for e in annotation_jobs]}"
                    )
                self._job_id = annotation_jobs[0]["job_id"]
            else:
                self._job_id = frame_map[0]["job_id"]
            n_frames = len([e for e in frame_map if e["job_id"] == self._job_id][0]["frame_names"])

        for c in self._cuts:
            if c <= 0 or c >= n_frames:
                raise ValueError(
                    f"Cut point {c} is out of range for job {self._job_id} "
                    f"which has {n_frames} frames (valid range: 1..{n_frames - 1})"
                )

    def _compute_layout(self, frame_map: list[dict]) -> list[dict]:
        """Compute the full job layout for the new task.

        Returns a list of dicts, each with ``"files"`` (list of frame names),
        ``"status"`` and ``"type"``.
        """
        layout = []

        for entry in frame_map:
            if entry["job_id"] == self._job_id:
                # This is the job being split
                names = entry["frame_names"]
                boundaries = [0] + self._cuts + [len(names)]
                for i in range(len(boundaries) - 1):
                    start, end = boundaries[i], boundaries[i + 1]
                    layout.append({
                        "files": names[start:end],
                        "status": "annotation",
                        "type": "annotation",
                        "frames": [],
                    })
            else:
                # Preserve other jobs as-is
                layout.append({
                    "files": entry["frame_names"],
                    "status": "annotation",
                    "type": entry["type"],
                    "frames": [],
                })

        return layout

    def _build_modified_backup(
        self,
        src_path: Path,
        dst_path: Path,
        task,
        new_job_layout: list[dict],
    ) -> None:
        """Rewrite the backup ZIP with new job boundaries."""
        with zipfile.ZipFile(src_path, "r") as src_zip:
            task_json = json.loads(src_zip.read("task.json"))

            # Rename
            task_json["name"] = f"{task_json['name']} (split)"

            # Force local storage so importer reads images from the ZIP
            task_json["data"]["storage"] = "local"

            # Enable custom segments
            task_json["data"]["custom_segments"] = True

            # Compute start_frame / stop_frame for each job entry
            offset = 0
            jobs_meta = []
            for jl in new_job_layout:
                n = len(jl["files"])
                jobs_meta.append({
                    "start_frame": offset,
                    "stop_frame": offset + n - 1,
                    "status": jl["status"],
                    "type": jl["type"],
                    "frames": jl["frames"],
                    "files": jl["files"],
                })
                offset += n

            task_json["jobs"] = jobs_meta

            # Build empty annotations (one entry per job, all empty)
            # Real annotations are imported separately via CVAT XML
            empty_annotations = [
                {"version": 0, "tags": [], "shapes": [], "tracks": []}
                for _ in new_job_layout
            ]

            # Write modified ZIP
            with zipfile.ZipFile(dst_path, "w", zipfile.ZIP_DEFLATED) as dst_zip:
                for item in src_zip.namelist():
                    if item in ("task.json", "annotations.json"):
                        continue
                    if item.startswith("jobs/"):
                        continue
                    dst_zip.writestr(item, src_zip.read(item))

                dst_zip.writestr(
                    "task.json", json.dumps(task_json, indent=2)
                )
                dst_zip.writestr(
                    "annotations.json", json.dumps(empty_annotations, indent=2)
                )

    def _build_result(self, new_task) -> SplitResult:
        """Fetch final state and return a SplitResult."""
        new_jobs = new_task.get_jobs()
        jobs_info = []
        total_annotations = 0

        for j in new_jobs:
            frames = j.get_frames_info()
            ann = self._sdk.api_client.jobs_api.retrieve_annotations(j.id)
            if isinstance(ann, tuple):
                ann = ann[0]
            n_shapes = len(ann.shapes) if hasattr(ann, "shapes") else 0
            total_annotations += n_shapes

            jobs_info.append({
                "job_id": j.id,
                "num_frames": len(frames),
                "first_frame": frames[0].name if frames else "",
                "last_frame": frames[-1].name if frames else "",
                "num_shapes": n_shapes,
            })

        return SplitResult(
            new_task_id=new_task.id,
            new_task_name=new_task.name,
            jobs=jobs_info,
            annotations_total=total_annotations,
        )
