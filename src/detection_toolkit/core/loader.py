"""Dataset loader: format detection, split detection, and unified loading.

Handles all five common directory layouts transparently:

    YOLO-A:  images/{split}/ + labels/{split}/ + data.yaml
    YOLO-B:  {split}/images/ + {split}/labels/ + data.yaml
    COCO-A:  images/{split}/ + annotations/instances_{split}.json
    COCO-B:  {split}/_annotations.coco.json  (Roboflow inline)
    COCO-C:  annotations/*.json + images/  (flat, single-split)
"""

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml

from .image_utils import _in_hidden_dir, find_image_file, find_images_in_dir
from .models import DatasetFormat, DatasetInfo, DatasetSplit, TaskType

logger = logging.getLogger(__name__)

# Split names we recognise in directory structures.
_KNOWN_SPLIT_DIRS = {"train", "val", "valid", "test"}

# Normalise common synonyms to canonical names.
_SPLIT_ALIASES = {"valid": "val"}


def _normalise_split(name: str) -> str:
    return _SPLIT_ALIASES.get(name.lower(), name.lower())


class DatasetLoader:
    """Load YOLO or COCO datasets with automatic format and split detection.

    Usage::

        loader = DatasetLoader("/path/to/dataset")
        info = loader.load_dataset()          # auto-detect format
        info = loader.load_dataset(DatasetFormat.YOLO)  # explicit
    """

    def __init__(
        self, dataset_dir: Union[str, Path], verbose: bool = True
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.verbose = verbose
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_dir}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_format(self) -> DatasetFormat:
        """Probe the directory tree and return the detected format.

        Raises ``ValueError`` if the format cannot be determined
        or if multiple formats are detected.
        """
        has_yolo = self._has_yolo_indicators()
        has_coco = self._has_coco_indicators()
        has_cvat = self._has_cvat_indicators()

        detected = [
            f
            for f, v in [
                (DatasetFormat.YOLO, has_yolo),
                (DatasetFormat.COCO, has_coco),
                (DatasetFormat.CVAT, has_cvat),
            ]
            if v
        ]

        if len(detected) > 1:
            raise ValueError(
                f"Multiple formats detected ({', '.join(f.value for f in detected)}) "
                f"in {self.dataset_dir}. Specify the format explicitly."
            )
        if len(detected) == 1:
            return detected[0]
        raise ValueError(
            f"Could not detect dataset format in {self.dataset_dir}"
        )

    def load_dataset(
        self,
        format: Optional[DatasetFormat] = None,
    ) -> DatasetInfo:
        """Load a full dataset into a :class:`DatasetInfo` object.

        Args:
            format: If ``None``, the format is auto-detected.
        """
        if format is None:
            format = self.detect_format()

        if format == DatasetFormat.YOLO:
            return self._load_yolo_dataset()
        if format == DatasetFormat.COCO:
            return self._load_coco_dataset()
        raise ValueError(f"Unsupported format: {format}")

    # ==================================================================
    # YOLO loading
    # ==================================================================

    def _load_yolo_dataset(self) -> DatasetInfo:
        data_yaml = self._find_data_yaml()
        class_names = self._parse_data_yaml_classes(data_yaml)
        splits = self._detect_yolo_splits(data_yaml)
        total_images = sum(s.num_images for s in splits.values())
        task_type = self._detect_yolo_task_type(splits)

        if self.verbose:
            logger.info(
                "Loaded YOLO dataset: %d splits, %d classes, %d images, task=%s",
                len(splits),
                len(class_names),
                total_images,
                task_type,
            )

        return DatasetInfo(
            format=DatasetFormat.YOLO,
            root_dir=self.dataset_dir,
            splits=splits,
            class_names=class_names,
            total_images=total_images,
            task_type=TaskType(task_type),
        )

    def _find_data_yaml(self) -> Path:
        candidates = list(self.dataset_dir.glob("*.yaml")) + list(
            self.dataset_dir.glob("*.yml")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No data.yaml / data.yml found in {self.dataset_dir}"
            )
        return candidates[0]

    def _parse_data_yaml_classes(self, yaml_path: Path) -> Dict[int, str]:
        with open(yaml_path, "r") as fh:
            data = yaml.safe_load(fh)

        names = data.get("names")
        if names is None:
            return {}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}
        return {}

    def _detect_yolo_splits(
        self, data_yaml: Path
    ) -> Dict[str, DatasetSplit]:
        """Detect splits from data.yaml paths — supports both YOLO-A and YOLO-B."""
        with open(data_yaml, "r") as fh:
            data = yaml.safe_load(fh)

        split_entries: Dict[str, str] = {}
        for key in ("train", "val", "test"):
            if key in data:
                split_entries[key] = data[key]

        if not split_entries:
            raise ValueError(f"No train/val/test paths found in {data_yaml}")

        splits: Dict[str, DatasetSplit] = {}
        for name, rel_path in split_entries.items():
            images_dir = self._resolve_yolo_images_dir(rel_path)
            if images_dir is None or not images_dir.exists():
                if self.verbose:
                    logger.warning(
                        "Skipping split %s: images dir not found (%s)",
                        name,
                        rel_path,
                    )
                continue

            labels_dir = self._infer_labels_dir(images_dir)
            image_paths = find_images_in_dir(images_dir)
            label_paths = sorted(
                p for p in labels_dir.rglob("*.txt")
                if not _in_hidden_dir(p, labels_dir)
            ) if labels_dir and labels_dir.exists() else []

            canon = _normalise_split(name)
            splits[canon] = DatasetSplit(
                name=canon,
                image_paths=image_paths,
                label_paths=label_paths,
            )

        return splits

    def _resolve_yolo_images_dir(self, rel_path: str) -> Optional[Path]:
        """Resolve a data.yaml relative path to an absolute images directory."""
        candidate = self.dataset_dir / rel_path
        if candidate.exists():
            return candidate
        # Some YAMLs use ./images/train — resolve from dataset root
        clean = rel_path.lstrip("./")
        candidate = self.dataset_dir / clean
        if candidate.exists():
            return candidate
        return None

    @staticmethod
    def _infer_labels_dir(images_dir: Path) -> Optional[Path]:
        """Given an images directory, infer the parallel labels directory.

        Works for both layout variants:
        - YOLO-A: ``images/train`` → ``labels/train``
        - YOLO-B: ``train/images`` → ``train/labels``
        """
        parts = list(images_dir.parts)
        if "images" in parts:
            label_parts = [p if p != "images" else "labels" for p in parts]
            return Path(*label_parts)
        return images_dir.parent / "labels"

    def _detect_yolo_task_type(
        self, splits: Dict[str, DatasetSplit]
    ) -> str:
        """Examine the first label file to determine detection vs segmentation."""
        for split in splits.values():
            for lbl in split.label_paths:
                if lbl.stat().st_size == 0:
                    continue
                with open(lbl, "r") as fh:
                    for raw_line in fh:
                        line = raw_line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        n_coords = len(parts) - 1
                        if n_coords >= 6:
                            return "segmentation"
                        if n_coords == 4:
                            return "detection"
                        break
                break
        return "detection"

    # ==================================================================
    # COCO loading
    # ==================================================================

    def _load_coco_dataset(self) -> DatasetInfo:
        splits = self._detect_coco_splits()
        class_names = self._load_coco_class_names(splits)
        total_images = sum(s.num_images for s in splits.values())
        task_type = self._detect_coco_task_type(splits)

        if self.verbose:
            logger.info(
                "Loaded COCO dataset: %d splits, %d classes, %d images, task=%s",
                len(splits),
                len(class_names),
                total_images,
                task_type,
            )

        return DatasetInfo(
            format=DatasetFormat.COCO,
            root_dir=self.dataset_dir,
            splits=splits,
            class_names=class_names,
            total_images=total_images,
            task_type=TaskType(task_type),
        )

    def _detect_coco_splits(self) -> Dict[str, DatasetSplit]:
        """Try COCO-A, then COCO-B, then COCO-C."""
        # COCO-A: annotations/instances_{split}.json
        splits = self._try_coco_a()
        if splits:
            return splits

        # COCO-B: {split}/_annotations.coco.json  (Roboflow inline)
        splits = self._try_coco_b()
        if splits:
            return splits

        # COCO-C: flat annotations/*.json  (single-split)
        splits = self._try_coco_c()
        if splits:
            return splits

        raise FileNotFoundError(
            "No COCO annotations found in any recognised layout "
            f"under {self.dataset_dir}"
        )

    def _try_coco_a(self) -> Optional[Dict[str, DatasetSplit]]:
        """Standard COCO: ``annotations/instances_{split}.json``."""
        ann_dir = self.dataset_dir / "annotations"
        if not ann_dir.is_dir():
            return None

        splits: Dict[str, DatasetSplit] = {}
        for json_path in sorted(ann_dir.glob("*.json")):
            split_name = self._split_name_from_json(json_path.name)
            if split_name is None:
                continue
            image_paths = self._image_paths_from_coco_json(
                json_path, search_root=self.dataset_dir
            )
            splits[split_name] = DatasetSplit(
                name=split_name,
                image_paths=image_paths,
                coco_ann_path=json_path,
                coco_images_root=None,
            )

        return splits or None

    def _try_coco_b(self) -> Optional[Dict[str, DatasetSplit]]:
        """Roboflow inline: ``{split}/_annotations.coco.json``."""
        splits: Dict[str, DatasetSplit] = {}
        for dir_name in ("train", "valid", "val", "test"):
            json_path = self.dataset_dir / dir_name / "_annotations.coco.json"
            if not json_path.exists():
                continue
            split_name = _normalise_split(dir_name)
            image_paths = self._image_paths_from_coco_json(
                json_path, search_root=json_path.parent
            )
            splits[split_name] = DatasetSplit(
                name=split_name,
                image_paths=image_paths,
                coco_ann_path=json_path,
                coco_images_root=json_path.parent,
            )
        return splits or None

    def _try_coco_c(self) -> Optional[Dict[str, DatasetSplit]]:
        """Flat COCO: a single JSON in ``annotations/`` with no split dirs."""
        ann_dir = self.dataset_dir / "annotations"
        if not ann_dir.is_dir():
            return None
        jsons = sorted(ann_dir.glob("*.json"))
        if not jsons:
            return None
        # Use the first JSON as the single "train" split.
        json_path = jsons[0]
        image_paths = self._image_paths_from_coco_json(
            json_path, search_root=self.dataset_dir
        )
        return {
            "train": DatasetSplit(
                name="train",
                image_paths=image_paths,
                coco_ann_path=json_path,
                coco_images_root=None,
            )
        }

    # ------------------------------------------------------------------
    # COCO helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_name_from_json(filename: str) -> Optional[str]:
        """Extract a split name from a COCO JSON filename.

        Recognises patterns like ``instances_train.json``,
        ``instances_val2017.json``, ``_train.json``, ``_default.json``.
        """
        lower = filename.lower()
        for suffix, split in [
            ("_train.json", "train"),
            ("_train2017.json", "train"),
            ("_val.json", "val"),
            ("_val2017.json", "val"),
            ("_validation.json", "val"),
            ("_test.json", "test"),
            ("_test2017.json", "test"),
            ("_default.json", "train"),
        ]:
            if lower.endswith(suffix):
                return split
        return None

    def _image_paths_from_coco_json(
        self, json_path: Path, search_root: Path
    ) -> List[Path]:
        """Read image filenames from a COCO JSON and resolve them on disk."""
        with open(json_path, "r") as fh:
            data = json.load(fh)

        paths: List[Path] = []
        for img_entry in data.get("images", []):
            fn = img_entry.get("file_name", "")
            if not fn:
                continue
            # Try direct concatenation first (faster than rglob)
            direct = search_root / fn
            if direct.exists():
                paths.append(direct)
                continue
            # Fallback: look inside images/ subdir
            via_images = search_root / "images" / fn
            if via_images.exists():
                paths.append(via_images)
                continue
            # Last resort: recursive search
            found = find_image_file(Path(fn).name, search_root)
            if found:
                paths.append(found)
        return paths

    def _load_coco_class_names(
        self, splits: Dict[str, DatasetSplit]
    ) -> Dict[int, str]:
        """Load category names from the first available COCO JSON."""
        for split in splits.values():
            if split.coco_ann_path and split.coco_ann_path.exists():
                return self._parse_coco_categories(split.coco_ann_path)
        return {}

    @staticmethod
    def _parse_coco_categories(json_path: Path) -> Dict[int, str]:
        with open(json_path, "r") as fh:
            data = json.load(fh)
        return {
            int(cat["id"]): str(cat["name"])
            for cat in data.get("categories", [])
        }

    def _detect_coco_task_type(
        self, splits: Dict[str, DatasetSplit]
    ) -> str:
        """Check the first annotation for segmentation data."""
        for split in splits.values():
            if not split.coco_ann_path:
                continue
            with open(split.coco_ann_path, "r") as fh:
                data = json.load(fh)
            for ann in data.get("annotations", []):
                if ann.get("segmentation"):
                    return "segmentation"
                if ann.get("bbox"):
                    return "detection"
        return "detection"

    # ==================================================================
    # Format probing
    # ==================================================================

    def _has_yolo_indicators(self) -> bool:
        """Return True if any YOLO-style layout is present."""
        # Must have a data.yaml
        yamls = list(self.dataset_dir.glob("*.yaml")) + list(
            self.dataset_dir.glob("*.yml")
        )
        if not yamls:
            return False

        # YOLO-A: images/ + labels/ at root
        if (self.dataset_dir / "images").is_dir() and (
            self.dataset_dir / "labels"
        ).is_dir():
            return True

        # YOLO-B: split dirs each containing images/ + labels/
        for d in self._iter_split_dirs():
            if (d / "images").is_dir() and (d / "labels").is_dir():
                return True

        return False

    def _has_coco_indicators(self) -> bool:
        """Return True if any COCO-style layout is present."""
        # COCO-A / COCO-C: annotations/ dir with JSON files
        ann_dir = self.dataset_dir / "annotations"
        if ann_dir.is_dir() and list(ann_dir.glob("*.json")):
            return True

        # COCO-B: split dirs with inline _annotations.coco.json
        for d in self._iter_split_dirs():
            if (d / "_annotations.coco.json").exists():
                return True

        return False

    def _has_cvat_indicators(self) -> bool:
        """Return True if a CVAT Image 1.1 XML file is found at the root."""
        for name in ("labels.xml", "annotations.xml"):
            xml_path = self.dataset_dir / name
            if not xml_path.is_file():
                continue
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                if root.tag == "annotations":
                    version_el = root.find("version")
                    if version_el is not None and version_el.text == "1.1":
                        return True
            except ET.ParseError:
                continue
        return False

    def _iter_split_dirs(self):
        """Yield existing subdirectories whose name is a known split."""
        for name in _KNOWN_SPLIT_DIRS:
            d = self.dataset_dir / name
            if d.is_dir():
                yield d
