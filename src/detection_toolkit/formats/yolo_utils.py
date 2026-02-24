"""YOLO format utilities: data.yaml creation, annotation remapping, task detection."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml

from ..core.models import DatasetFormat, DatasetInfo, TaskType

logger = logging.getLogger(__name__)


def create_data_yaml(
    output_path: Path,
    classes: Union[List[str], Dict[int, str]],
    splits: Union[List[str], Set[str], Dict[str, Any]],
    sort_classes: bool = False,
) -> Path:
    """Create a standardised YOLO ``data.yaml`` file.

    Args:
        output_path: Directory where ``data.yaml`` will be written.
        classes: Class names as a list (0-indexed) or ``{id: name}`` dict.
        splits: Split names.
        sort_classes: If *True*, sort class names alphabetically
            and re-assign sequential IDs.

    Returns:
        Path to the created ``data.yaml`` file.
    """
    output_path = Path(output_path)

    # Normalise classes → {int: str}
    if isinstance(classes, list):
        if sort_classes:
            classes = sorted(classes)
        class_dict = {i: name for i, name in enumerate(classes)}
    elif isinstance(classes, dict):
        if sort_classes:
            sorted_items = sorted(classes.items(), key=lambda x: x[1])
            class_dict = {i: name for i, (_, name) in enumerate(sorted_items)}
        else:
            class_dict = {int(k): str(v) for k, v in classes.items()}
    else:
        raise TypeError(f"classes must be a list or dict, got {type(classes)}")

    # Normalise split names
    if isinstance(splits, (list, set)):
        split_names = sorted(splits) if isinstance(splits, set) else list(splits)
    elif isinstance(splits, dict):
        split_names = sorted(splits.keys())
    else:
        raise TypeError(f"splits must be a list, set, or dict, got {type(splits)}")

    # Build YAML payload
    yaml_data: Dict[str, Any] = {
        "nc": len(class_dict),
        "names": class_dict,
    }
    for name in split_names:
        yaml_data[name] = f"images/{name}"

    yaml_path = output_path / "data.yaml"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as fh:
        yaml.dump(yaml_data, fh, default_flow_style=False, sort_keys=False)

    logger.info("Created data.yaml with %d classes at %s", len(class_dict), yaml_path)
    return yaml_path


def create_data_yaml_from_dataset_info(
    output_path: Path,
    dataset_info: DatasetInfo,
    sort_classes: bool = False,
) -> Path:
    """Convenience wrapper around :func:`create_data_yaml`."""
    return create_data_yaml(
        output_path=output_path,
        classes=dataset_info.class_names,
        splits=list(dataset_info.splits.keys()),
        sort_classes=sort_classes,
    )


def remap_yolo_annotations(
    src_file: Path,
    dst_file: Path,
    class_mapping: Dict[int, int],
    skip_unmapped: bool = False,
) -> int:
    """Re-write a YOLO label file with remapped class IDs.

    Args:
        src_file: Source ``.txt`` label file.
        dst_file: Destination ``.txt`` label file (created / overwritten).
        class_mapping: ``{old_id: new_id}``.  A ``new_id < 0`` is treated
            as "drop this class" when *skip_unmapped* is *True*.
        skip_unmapped: If *True*, lines whose old class id maps to a
            negative value are silently dropped.

    Returns:
        Number of lines written.
    """
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(src_file, "r") as fin, open(dst_file, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                old_id = int(parts[0])
            except ValueError:
                logger.warning("Skipping malformed line in %s: %s", src_file, line.strip())
                continue

            if old_id not in class_mapping:
                if not skip_unmapped:
                    fout.write(line)
                    written += 1
                continue

            new_id = class_mapping[old_id]
            if skip_unmapped and new_id < 0:
                continue

            fout.write(f"{new_id} {' '.join(parts[1:])}\n")
            written += 1

    return written


def detect_task_type(input_path: Union[str, Path]) -> TaskType:
    """Detect whether a YOLO dataset targets detection or segmentation.

    Examines the first non-empty label line:
    4 values after the class id → detection,
    6+ values → segmentation.

    *input_path* may be a ``.txt`` label file, a ``data.yaml`` / ``.yml``,
    or a dataset directory.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    txt_file: Optional[Path] = None
    if input_path.is_file():
        if input_path.suffix.lower() == ".txt":
            txt_file = input_path
        elif input_path.suffix.lower() in (".yaml", ".yml"):
            txt_file = _find_label_from_yaml(input_path)
        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
    elif input_path.is_dir():
        txt_file = _find_label_in_dir(input_path)

    if txt_file is None:
        raise RuntimeError(f"No label files found under {input_path}")
    return _examine_label_file(txt_file)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _find_label_from_yaml(yaml_path: Path) -> Optional[Path]:
    """Resolve a ``data.yaml`` → first non-empty label file."""
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)

    root = yaml_path.parent
    if isinstance(data.get("path"), str) and data["path"] != ".":
        p = Path(data["path"])
        root = p if p.is_absolute() else yaml_path.parent / p

    for subdir in [root / "labels" / "train", root / "labels", root]:
        if not subdir.is_dir():
            continue
        for txt in subdir.rglob("*.txt"):
            if txt.stat().st_size > 0:
                return txt
    return None


def _find_label_in_dir(folder: Path) -> Optional[Path]:
    """Find a non-empty ``.txt`` in *folder* or via its ``data.yaml``."""
    for txt in folder.rglob("*.txt"):
        if txt.stat().st_size > 0:
            return txt
    yamls = list(folder.glob("*.yaml")) + list(folder.glob("*.yml"))
    if yamls:
        return _find_label_from_yaml(yamls[0])
    return None


def _examine_label_file(txt_file: Path) -> TaskType:
    with open(txt_file) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            n_coords = len(parts) - 1
            if n_coords >= 6 and n_coords % 2 == 0:
                return TaskType.SEGMENTATION
            if n_coords == 4:
                return TaskType.DETECTION
            raise ValueError(
                f"Expected 4 (bbox) or 6+ (polygon) coords, got {n_coords} in {txt_file}"
            )
    raise ValueError(f"No valid lines in {txt_file}")
