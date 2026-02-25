"""Image utilities: size extraction and file lookup."""

from pathlib import Path
from typing import Union

from cocoyolo.image_utils import get_image_size  # noqa: F401 — re-export
from cocoyolo.io_utils import find_coco_image as find_image_file  # noqa: F401

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


def _in_hidden_dir(path: Path, root: Path) -> bool:
    """Return *True* if *path* is inside a hidden (dot-prefixed) directory under *root*."""
    return any(part.startswith(".") for part in path.relative_to(root).parts[:-1])


def find_images_in_dir(directory: Union[str, Path]) -> list[Path]:
    """Return a sorted list of image files found (recursively) under *directory*.

    Skips files inside hidden (dot-prefixed) sub-directories.
    """
    directory = Path(directory)
    out: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        out.extend(p for p in directory.rglob(f"*{ext}") if not _in_hidden_dir(p, directory))
        out.extend(p for p in directory.rglob(f"*{ext.upper()}") if not _in_hidden_dir(p, directory))
    return sorted(set(out))
