"""Grid thumbnails for FiftyOne image datasets.

Thumbnails live in a hidden ``.{dataset}_thumbnails/`` folder that MIRRORS the
source tree below the samples' common folder, so two images that share a file
name in different folders (robot video frames, drone photo dumps) always get
two different thumbnails. A flat ``<folder>/<basename>`` layout — what
``fiftyone.utils.image.transform_images`` produces without ``rel_dir`` — makes
same-named images overwrite each other and the App grid shows the wrong photo
(measured: 1,380 samples of one dataset, 2026-09-26).

Module scope stays free of FiftyOne imports so the pure helpers (planning,
audit) import cheaply.
"""
import hashlib
import logging
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

FIELD = "thumbnail_path"
INFO_KEY = "thumbnails"
FORMATS = ("webp", "jpg", "keep")
_INLINE_BELOW = 32          # fewer images than this: skip the thread pool


def thumbnails_root(
    filepaths: Sequence[str],
    dataset_name: str,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Return ``(thumbnails_dir, rel_root)``, both absolute.

    ``rel_root`` is the deepest folder containing every sample; thumbnails
    mirror the tree below it. The thumbnails folder is created inside
    ``output_dir`` when given, else beside ``rel_root`` (a sibling, so globs
    over the dataset folder never pick thumbnails up) — moved further up when
    that would sit inside a DVC-tracked folder (``<dir>.dvc`` beside it).
    """
    rel_root = Path(os.path.commonpath([os.path.dirname(os.path.abspath(f))
                                        for f in filepaths]))
    if output_dir is None:
        if len(rel_root.parts) < 3:
            raise ValueError(
                f"the samples' common folder is {rel_root}; pass output_dir "
                "explicitly rather than writing thumbnails next to it")
        parent = rel_root.parent
        for folder in [parent, *parent.parents]:
            if (folder.parent / f"{folder.name}.dvc").exists():
                parent = folder.parent
    else:
        parent = Path(output_dir).resolve()
    return parent / f".{dataset_name}_thumbnails", rel_root


def plan_thumbnail_paths(
    filepaths: Sequence[str],
    thumbnails_dir: Union[str, Path],
    rel_root: Union[str, Path],
    fmt: str = "webp",
) -> List[str]:
    """One thumbnail path per filepath, mirroring the tree below ``rel_root``.

    The format's extension is APPENDED (``f.png`` -> ``f.png.webp``), never
    substituted: ``f.png`` and ``f.jpg`` in one folder must not both become
    ``f.webp``. ``fmt="keep"`` writes the source's own format.
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown thumbnail format {fmt!r}; choose from {FORMATS}")
    suffix = "" if fmt == "keep" else f".{fmt}"
    rel_root = os.path.abspath(rel_root)
    out = []
    for fp in filepaths:
        rel = os.path.relpath(os.path.abspath(fp), rel_root)
        if rel.startswith(os.pardir):
            raise ValueError(f"{fp} is outside the thumbnail root {rel_root}")
        out.append(str(Path(thumbnails_dir) / rel) + suffix)
    return out


def needs_thumbnail(src: str, thumb: str, force: bool = False) -> bool:
    """True when the thumbnail is missing, older than its source, or forced."""
    if force or not os.path.exists(thumb):
        return True
    return os.path.getmtime(thumb) < os.path.getmtime(src)


def make_thumbnail(src: str, dst: str, size: int) -> bool:
    """Write a ``size``-px-high, aspect-preserving, EXIF-upright copy of ``src``.

    Never upscales. Written to a temp file and renamed, so an interrupted run
    never leaves a truncated thumbnail that later passes the mtime check.
    """
    from PIL import Image, ImageOps

    try:
        with Image.open(src) as im:
            if im.format == "JPEG":
                im.draft("RGB", (size, size))   # fast shrink-on-load
            im = ImageOps.exif_transpose(im)
            if im.height > size:
                width = max(1, round(im.width * size / im.height))
                im = im.resize((width, size), Image.LANCZOS)
            ext = os.path.splitext(dst)[1].lower()
            if ext in (".jpg", ".jpeg") and im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            elif ext == ".webp" and im.mode not in ("RGB", "RGBA"):
                alpha = "A" in im.getbands() or "transparency" in im.info
                im = im.convert("RGBA" if alpha else "RGB")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = f"{dst}.tmp{threading.get_ident()}{ext}"
            if ext in (".jpg", ".jpeg"):
                im.save(tmp, quality=85)
            elif ext == ".webp":
                im.save(tmp, quality=80, method=4)
            else:
                im.save(tmp)
        os.replace(tmp, dst)
        return True
    except Exception as err:            # one bad file must not sink the run
        logger.warning("thumbnail failed for %s: %s", src, err)
        return False


def _make_one(job):
    return make_thumbnail(*job)


def _generate(jobs: List[Tuple[str, str, int]], num_workers: Optional[int]) -> List[bool]:
    # Threads, not processes: Pillow releases the GIL while decoding and
    # resizing, and a process pool breaks when the caller has no importable
    # __main__ (stdin, some embedded sessions) or holds a pymongo client.
    if len(jobs) < _INLINE_BELOW or num_workers == 1:
        return [_make_one(j) for j in jobs]
    workers = num_workers or min(16, os.cpu_count() or 2)
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(_make_one, jobs))


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audit_thumbnails(filepaths: Sequence[str], thumbnail_paths: Sequence[Optional[str]]) -> Dict:
    """Read-only health check of a dataset's thumbnails.

    ``shared_by_different_images`` counts distinct images whose thumbnail file
    is also used by a DIFFERENT image (byte-identical copies are exempt — a
    shared thumbnail is then still correct).
    """
    unset = relative = missing = 0
    groups = defaultdict(set)
    for fp, tp in zip(filepaths, thumbnail_paths):
        if not tp:
            unset += 1
            continue
        if not os.path.isabs(tp):
            relative += 1
        if not os.path.exists(tp):
            missing += 1
        groups[tp].add(fp)
    shared = 0
    for sources in groups.values():
        if len(sources) > 1:
            readable = [s for s in sources if os.path.exists(s)]
            if len({_md5(s) for s in readable}) > 1 or len(readable) < len(sources):
                shared += len(sources)
    rep = {"n": len(filepaths), "unset": unset, "relative": relative,
           "missing_file": missing, "shared_by_different_images": shared}
    rep["ok"] = not any((unset, relative, missing, shared))
    return rep


def _still_covers(previous: Dict, filepaths: Sequence[str]) -> bool:
    rel_root = previous.get("rel_root")
    if not (previous.get("root") and rel_root):
        return False
    prefix = os.path.join(os.path.abspath(rel_root), "")
    return all(os.path.abspath(fp).startswith(prefix) for fp in filepaths)


def create_thumbnails(
    dataset,
    output_dir: Optional[Union[str, Path]] = None,
    size: int = 256,
    force: bool = False,
    num_workers: Optional[int] = None,
    fmt: str = "webp",
) -> Dict:
    """Create/refresh grid thumbnails and wire them into the App.

    Incremental: only missing or out-of-date thumbnails are written; a
    different ``size`` from the last run regenerates all of them. Every sample
    gets an ABSOLUTE ``thumbnail_path``; an image that cannot be read falls
    back to its own filepath so the grid still shows it. Existing
    ``app_config.media_fields`` entries are kept. ``fmt`` is ``"webp"``
    (default, ~20-30x smaller than PNG thumbnails), ``"jpg"`` or ``"keep"``;
    changing it regenerates everything.

    Returns ``{"generated", "reused", "failed", "root"}``.
    """
    ids = dataset.values("id")
    filepaths = dataset.values("filepath")
    if not filepaths:
        logger.warning("Dataset is empty — skipping thumbnail creation")
        return {"generated": 0, "reused": 0, "failed": 0, "root": None}

    previous = (dataset.info or {}).get(INFO_KEY) or {}
    if output_dir is None and _still_covers(previous, filepaths):
        # a sample added in a new subfolder must not move the whole tree
        thumbs_dir, rel_root = Path(previous["root"]), Path(previous["rel_root"])
    else:
        thumbs_dir, rel_root = thumbnails_root(filepaths, dataset.name, output_dir)
    planned = plan_thumbnail_paths(filepaths, thumbs_dir, rel_root, fmt=fmt)

    if (previous.get("size") not in (None, size)
            or previous.get("root") not in (None, str(thumbs_dir))
            or previous.get("format", "keep") != fmt):
        force = True

    unique = dict(zip(filepaths, planned))
    todo = [(fp, tp, size) for fp, tp in unique.items() if needs_thumbnail(fp, tp, force)]
    logger.info("Thumbnails in %s: %d to write, %d up to date",
                thumbs_dir, len(todo), len(unique) - len(todo))
    ok = _generate(todo, num_workers)
    failed = {job[0] for job, good in zip(todo, ok) if not good}

    values = {sid: (fp if fp in failed else tp)
              for sid, fp, tp in zip(ids, filepaths, planned)}
    dataset.set_values(FIELD, values, key_field="id")

    media = list(dataset.app_config.media_fields or ["filepath"])
    for field in ("filepath", FIELD):
        if field not in media:
            media.append(field)
    dataset.app_config.media_fields = media
    dataset.app_config.grid_media_field = FIELD
    info = dict(dataset.info or {})
    info[INFO_KEY] = {"root": str(thumbs_dir), "rel_root": str(rel_root), "size": size,
                      "format": fmt}
    dataset.info = info
    dataset.save()

    summary = {"generated": len(todo) - len(failed), "reused": len(unique) - len(todo),
               "failed": len(failed), "root": str(thumbs_dir)}
    logger.info("Thumbnails: %s", summary)
    return summary
