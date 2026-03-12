"""Parallel processing utilities for dataset tiling.

Provides the process-count heuristic and the standalone worker function
that can be dispatched to a :class:`~concurrent.futures.ProcessPoolExecutor`.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def optimal_process_count(
    num_images: int, max_processes: Optional[int] = None
) -> int:
    """Return a reasonable worker count for tiling *num_images*.

    Heuristic: ``min(cpu_count - 1, num_images)`` capped by *max_processes*.
    """
    cpus = os.cpu_count() or 1
    n = max(1, cpus - 1)  # leave one core for the main process

    if max_processes is not None:
        n = min(n, max_processes)

    return max(1, min(n, num_images))


# ------------------------------------------------------------------
# YOLO worker (top-level function, pickle-safe)
# ------------------------------------------------------------------

def tile_single_yolo_image(args: Tuple) -> Tuple[int, int]:
    """Tile one image and write YOLO label files.

    Designed for :func:`ProcessPoolExecutor.map`.

    Args:
        args: ``(img_path, label_path, output_path, split_name,
                 pad_width, config_dict)``

    Returns:
        ``(positive_tiles, negative_tiles)``
    """
    import pyvips

    from .config import TilingConfig
    from .geometry import compute_tile_grid

    (
        img_path_str,
        label_path_str,
        output_path_str,
        split_name,
        pad_width,
        config_dict,
    ) = args

    img_path = Path(img_path_str)
    label_path = Path(label_path_str) if label_path_str else None
    output_path = Path(output_path_str)
    cfg = TilingConfig.from_dict(config_dict)

    img_out = output_path / "images" / split_name
    lbl_out = output_path / "labels" / split_name

    positive = 0
    negative = 0

    try:
        img = pyvips.Image.new_from_file(str(img_path))
        img_w, img_h = img.width, img.height

        # Parse annotations
        annotations = _parse_yolo_labels(label_path, img_w, img_h)

        # Compute tile grid
        tiles = compute_tile_grid(
            img_w, img_h, cfg.tile_size,
            cfg.overlap_percent, cfg.edge_behavior,
        )

        for idx, (tx, ty, tw, th) in enumerate(tiles):
            tile_name = f"{img_path.stem}_tile_{idx:0{pad_width}d}"

            # Clip annotations to tile
            tile_anns = _clip_annotations(
                annotations, tx, ty, tw, th, cfg,
            )

            # Write tile image
            tile_img_path = img_out / f"{tile_name}{img_path.suffix}"
            _write_tile_image(img, tx, ty, tw, th, tile_img_path, cfg)

            # Write tile label
            tile_lbl_path = lbl_out / f"{tile_name}.txt"
            if tile_anns:
                _write_yolo_label(
                    tile_lbl_path, tile_anns, tw, th,
                    disjoint_strategy=cfg.disjoint_strategy,
                )
                positive += 1
            else:
                # Empty label for negative tile
                tile_lbl_path.write_text("")
                negative += 1

    except Exception as exc:
        logger.error("Failed to tile %s: %s", img_path.name, exc)
        return (0, 0)

    return (positive, negative)


def tile_single_image(args: Tuple) -> int:
    """Tile one image without annotations.

    Designed for :func:`ProcessPoolExecutor.map`.

    Args:
        args: ``(img_path, output_dir, tile_prefix, pad_width, config_dict)``

    Returns:
        Number of tiles written.
    """
    import pyvips

    from .config import TilingConfig
    from .geometry import compute_tile_grid

    img_path_str, output_dir_str, tile_prefix, pad_width, config_dict = args
    img_path = Path(img_path_str)
    output_dir = Path(output_dir_str)
    cfg = TilingConfig.from_dict(config_dict)

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    try:
        img = pyvips.Image.new_from_file(str(img_path))
        tiles = compute_tile_grid(
            img.width, img.height, cfg.tile_size,
            cfg.overlap_percent, cfg.edge_behavior,
        )
        for idx, (tx, ty, tw, th) in enumerate(tiles):
            tile_name = f"{tile_prefix}_tile_{idx:0{pad_width}d}"
            out_path = output_dir / f"{tile_name}{img_path.suffix}"
            _write_tile_image(img, tx, ty, tw, th, out_path, cfg)
            count += 1
    except Exception as exc:
        logger.error("Failed to tile %s: %s", img_path.name, exc)
        return 0

    return count


# ------------------------------------------------------------------
# Internal helpers (used inside the worker)
# ------------------------------------------------------------------

def _parse_yolo_labels(
    label_path: Optional[Path],
    img_w: int,
    img_h: int,
) -> List[Dict]:
    """Parse a YOLO label file into annotation dicts."""
    if label_path is None or not label_path.exists():
        return []

    annotations: List[Dict] = []
    with open(label_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except (ValueError, IndexError):
                continue

            if len(coords) == 4:
                xc, yc, w, h = coords
                abs_x = (xc - w / 2) * img_w
                abs_y = (yc - h / 2) * img_h
                abs_w = w * img_w
                abs_h = h * img_h
                annotations.append({
                    "class_id": class_id,
                    "type": "bbox",
                    "bbox": [abs_x, abs_y, abs_w, abs_h],
                })
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                points = [
                    (coords[i] * img_w, coords[i + 1] * img_h)
                    for i in range(0, len(coords), 2)
                ]
                annotations.append({
                    "class_id": class_id,
                    "type": "polygon",
                    "polygons": [points],
                })

    return annotations


def _clip_annotations(
    annotations: List[Dict],
    tx: int,
    ty: int,
    tw: int,
    th: int,
    cfg,
) -> List[Dict]:
    """Clip annotation dicts to a tile and filter by area ratio.

    Works with annotation dicts from both YOLO label parsing and FiftyOne
    extraction — both use the ``{class_id, type, bbox/polygons}`` format.
    """
    from .geometry import (
        bbox_area_ratio,
        clip_bbox_to_tile,
        clip_polygon_to_tile,
        polygon_area_ratio,
        to_tile_coords,
    )

    result: List[Dict] = []

    for ann in annotations:
        if ann["type"] == "polygon":
            input_polys = ann["polygons"]
            kept: List[list] = []
            for input_poly in input_polys:
                clipped_polys = clip_polygon_to_tile(
                    input_poly, tx, ty, tw, th,
                    densify_factor=cfg.densify_factor,
                    smoothing_tolerance=cfg.smoothing_tolerance,
                )
                if not clipped_polys:
                    continue
                for poly in clipped_polys:
                    local = to_tile_coords(poly, tx, ty)
                    ratio = polygon_area_ratio(local, tw, th)
                    if cfg.min_area_ratio <= ratio <= cfg.max_area_ratio:
                        kept.append(local)

            if kept:
                result.append({
                    "class_id": ann["class_id"],
                    "type": "polygon",
                    "polygons": kept,
                })

        elif ann["type"] == "bbox":
            clipped = clip_bbox_to_tile(ann["bbox"], tx, ty, tw, th)
            if clipped is None:
                continue

            ratio = bbox_area_ratio(clipped, tw, th)
            if cfg.min_area_ratio <= ratio <= cfg.max_area_ratio:
                result.append({
                    "class_id": ann["class_id"],
                    "type": "bbox",
                    "bbox": clipped,
                })

    return result


def _write_tile_image(
    img,  # pyvips.Image
    tx: int,
    ty: int,
    tw: int,
    th: int,
    out_path: Path,
    cfg,
) -> None:
    """Crop a tile from *img* and write it to *out_path*."""
    import pyvips

    read_w = min(tw, img.width - tx)
    read_h = min(th, img.height - ty)

    tile = img.crop(tx, ty, read_w, read_h)

    # Pad edge tiles if needed
    if cfg.edge_behavior == "padding" and (read_w < tw or read_h < th):
        r, g, b = cfg.padding_value
        bg = [r, g, b, 255] if img.bands > 3 else [r, g, b]
        tile = tile.gravity(
            "north-west", tw, th,
            extend="background", background=bg,
        )

    ext = out_path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        tile.jpegsave(str(out_path), Q=cfg.compression)
    elif ext == ".png":
        tile.pngsave(str(out_path))
    else:
        tile.write_to_file(str(out_path))


# ------------------------------------------------------------------
# FiftyOne worker (top-level function, serialisation-safe)
# ------------------------------------------------------------------

def tile_single_fo_sample(args: Tuple) -> Tuple[int, int, List[Dict]]:
    """Tile one image using pre-extracted FiftyOne annotations.

    Designed for :func:`ProcessPoolExecutor.map`.

    Args:
        args: ``(img_path_str, annotations, output_dir_str, split_name,
                 tile_prefix, pad_width, class_list, config_dict,
                 output_format)``

            *annotations* is a list of dicts from
            :func:`~detection_toolkit.tiling.fo_extract.extract_annotations`.
            *output_format* is ``"yolo"`` or ``"coco"``.

    Returns:
        ``(positive_tiles, negative_tiles, coco_entries)`` where
        *coco_entries* is an empty list for YOLO output.
    """
    import pyvips

    from .config import TilingConfig
    from .geometry import compute_tile_grid

    (
        img_path_str,
        annotations,
        output_dir_str,
        split_name,
        tile_prefix,
        pad_width,
        class_list,
        config_dict,
        output_format,
    ) = args

    img_path = Path(img_path_str)
    output_dir = Path(output_dir_str)
    cfg = TilingConfig.from_dict(config_dict)

    img_out = output_dir / "images" / split_name
    img_out.mkdir(parents=True, exist_ok=True)

    if output_format == "yolo":
        lbl_out = output_dir / "labels" / split_name
        lbl_out.mkdir(parents=True, exist_ok=True)

    positive = 0
    negative = 0
    coco_entries: List[Dict] = []

    try:
        img = pyvips.Image.new_from_file(str(img_path))
        img_w, img_h = img.width, img.height

        tiles = compute_tile_grid(
            img_w, img_h, cfg.tile_size,
            cfg.overlap_percent, cfg.edge_behavior,
        )

        for idx, (tx, ty, tw, th) in enumerate(tiles):
            tile_name = f"{tile_prefix}_tile_{idx:0{pad_width}d}"

            # Clip annotations to tile
            tile_anns = _clip_annotations(
                annotations, tx, ty, tw, th, cfg,
            )

            # Write tile image
            tile_img_path = img_out / f"{tile_name}{img_path.suffix}"
            _write_tile_image(img, tx, ty, tw, th, tile_img_path, cfg)

            has_anns = bool(tile_anns)

            if output_format == "yolo":
                tile_lbl_path = lbl_out / f"{tile_name}.txt"
                if has_anns:
                    _write_yolo_label(
                        tile_lbl_path, tile_anns, tw, th,
                        disjoint_strategy=cfg.disjoint_strategy,
                    )
                    positive += 1
                else:
                    tile_lbl_path.write_text("")
                    negative += 1

            elif output_format == "coco":
                tile_file_name = f"{tile_name}{img_path.suffix}"
                image_entry = {
                    "file_name": tile_file_name,
                    "width": tw,
                    "height": th,
                }
                if has_anns:
                    ann_entries = _to_coco_annotations(
                        tile_anns, tw, th, class_list,
                    )
                    coco_entries.append({
                        "image": image_entry,
                        "annotations": ann_entries,
                    })
                    positive += 1
                else:
                    coco_entries.append({
                        "image": image_entry,
                        "annotations": [],
                    })
                    negative += 1

    except Exception as exc:
        logger.error("Failed to tile %s: %s", img_path.name, exc)
        return (0, 0, [])

    return (positive, negative, coco_entries)


def _to_coco_annotations(
    tile_anns: List[Dict],
    tile_w: int,
    tile_h: int,
    class_list: List[str],
) -> List[Dict]:
    """Convert clipped tile annotations to COCO format dicts.

    Returns a list of COCO annotation dicts with ``category_id`` (1-based),
    ``bbox``, ``area``, and optionally ``segmentation``.
    """
    coco_anns: List[Dict] = []

    from .geometry import bbox_from_polygons, polygon_area

    for ann in tile_anns:
        category_id = ann["class_id"] + 1  # COCO is 1-based

        if ann["type"] == "polygon":
            all_flat: List[list] = []
            total_area = 0.0

            for points in ann["polygons"]:
                flat = []
                for x, y in points:
                    cx = max(0.0, min(float(tile_w), float(x)))
                    cy = max(0.0, min(float(tile_h), float(y)))
                    flat.extend([cx, cy])
                all_flat.append(flat)
                total_area += polygon_area(points)

            bbox = bbox_from_polygons(
                ann["polygons"], clamp_w=tile_w, clamp_h=tile_h,
            )

            coco_anns.append({
                "category_id": category_id,
                "bbox": bbox,
                "area": total_area,
                "segmentation": all_flat,
                "iscrowd": 0,
            })

        elif ann["type"] == "bbox":
            x, y, w, h = ann["bbox"]
            coco_anns.append({
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })

    return coco_anns


def _write_yolo_label(
    path: Path,
    annotations: List[Dict],
    tile_w: int,
    tile_h: int,
    disjoint_strategy: str = "bridge",
) -> None:
    """Write clipped annotations as a YOLO label file.

    Polygon annotations use the ``"polygons"`` key (list of point lists).
    When an annotation has multiple sub-polygons (from clipping a concave
    shape), *disjoint_strategy* controls the output:

    * ``"bridge"`` — connect via zero-width bridges into a single YOLO line.
    * ``"split"``  — emit each sub-polygon as a separate YOLO line.
    """
    from .geometry import bbox_to_yolo_center, normalize_to_yolo

    lines: List[str] = []
    for ann in annotations:
        cid = ann["class_id"]
        if ann["type"] == "polygon":
            sub_polys = ann["polygons"]

            if len(sub_polys) > 1 and disjoint_strategy == "bridge":
                from cocoyolo.geometry import bridge_disjoint
                # Convert tuples to lists for bridge_disjoint
                point_lists = [
                    [[float(x), float(y)] for x, y in poly]
                    for poly in sub_polys
                ]
                bridged = bridge_disjoint(point_lists)
                coords = normalize_to_yolo(bridged, tile_w, tile_h)
                if len(coords) >= 4 and coords[0] == coords[-2] and coords[1] == coords[-1]:
                    coords = coords[:-2]
                coord_str = " ".join(f"{c:.6f}" for c in coords)
                lines.append(f"{cid} {coord_str}")
            else:
                # Single polygon, or split strategy
                for poly in sub_polys:
                    coords = normalize_to_yolo(poly, tile_w, tile_h)
                    if len(coords) >= 4 and coords[0] == coords[-2] and coords[1] == coords[-1]:
                        coords = coords[:-2]
                    coord_str = " ".join(f"{c:.6f}" for c in coords)
                    lines.append(f"{cid} {coord_str}")

        elif ann["type"] == "bbox":
            yolo = bbox_to_yolo_center(ann["bbox"], tile_w, tile_h)
            coord_str = " ".join(f"{c:.6f}" for c in yolo)
            lines.append(f"{cid} {coord_str}")

    path.write_text("\n".join(lines) + "\n" if lines else "")
