"""Image utilities: size extraction and file lookup."""

from pathlib import Path
from typing import Union

from cocoyolo.image_utils import get_image_size  # noqa: F401 — re-export
from cocoyolo.io_utils import find_coco_image as find_image_file  # noqa: F401

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


def find_images_in_dir(directory: Union[str, Path]) -> list[Path]:
    """Return a sorted list of image files found (recursively) under *directory*."""
    directory = Path(directory)
    out: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        out.extend(directory.rglob(f"*{ext}"))
        out.extend(directory.rglob(f"*{ext.upper()}"))
    return sorted(set(out))
