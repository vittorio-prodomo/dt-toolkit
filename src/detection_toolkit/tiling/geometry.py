"""Geometry operations for polygon processing during tiling.

Provides densification, smoothing, and tile-clipping of polygons
using Shapely for robust computational geometry.
"""

import logging
from typing import List, Optional, Tuple, Union

from shapely.geometry import MultiPolygon, Polygon, box

logger = logging.getLogger(__name__)

# Type alias: list of (x, y) tuples
PointList = List[Tuple[float, float]]


# ------------------------------------------------------------------
# Polygon densification
# ------------------------------------------------------------------

def densify_polygon(
    points: PointList, factor: float = 0.5
) -> PointList:
    """Add intermediate points along polygon edges.

    The maximum segment length is ``perimeter × factor``.  Smaller
    *factor* values produce denser output.

    Args:
        points: Polygon vertices as ``[(x, y), ...]``.
        factor: Densification factor (0 < factor ≤ 1).

    Returns:
        Densified polygon vertices (always closed: first == last).
    """
    if len(points) < 3 or factor <= 0:
        return points

    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    max_dist = poly.length * factor

    if max_dist <= 0:
        return points

    return _densify_ring(list(poly.exterior.coords), max_dist)


def _densify_ring(
    coords: List[Tuple[float, float]], max_dist: float
) -> List[Tuple[float, float]]:
    """Insert points along a coordinate ring so no segment exceeds *max_dist*."""
    result: List[Tuple[float, float]] = []
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg_len = (dx * dx + dy * dy) ** 0.5

        result.append((x0, y0))
        if seg_len > max_dist:
            n_steps = int(seg_len / max_dist) + 1
            for s in range(1, n_steps):
                t = s / n_steps
                result.append((x0 + dx * t, y0 + dy * t))

    # Close ring
    if result and result[0] != result[-1]:
        result.append(result[0])
    return result


# ------------------------------------------------------------------
# Polygon smoothing
# ------------------------------------------------------------------

def smooth_polygon(
    points: PointList, tolerance: float = 0.1
) -> PointList:
    """Simplify a polygon using Shapely's topology-preserving algorithm.

    Args:
        points: Polygon vertices as ``[(x, y), ...]``.
        tolerance: Simplification tolerance in the same units as coordinates.

    Returns:
        Simplified polygon vertices.
    """
    if len(points) < 3 or tolerance <= 0:
        return points

    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    simplified = poly.simplify(tolerance, preserve_topology=True)

    if simplified.is_empty or not isinstance(simplified, Polygon):
        return points

    coords = list(simplified.exterior.coords)
    if len(coords) < 4:  # degenerate after simplification
        return points
    return coords


# ------------------------------------------------------------------
# Tile clipping
# ------------------------------------------------------------------

def clip_polygon_to_tile(
    points: PointList,
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
    densify_factor: float = 0.5,
    smoothing_tolerance: float = 0.1,
) -> Optional[List[PointList]]:
    """Clip a polygon to a tile rectangle, with optional densify + smooth.

    Args:
        points: Polygon vertices in absolute image coordinates.
        tile_x, tile_y: Top-left corner of the tile.
        tile_w, tile_h: Tile dimensions.
        densify_factor: Densification factor (0 disables).
        smoothing_tolerance: Simplification tolerance (0 disables).

    Returns:
        List of clipped polygon coordinate lists (may produce multiple
        polygons if the original straddles the tile edge), or ``None``
        if nothing remains after clipping.
    """
    if len(points) < 3:
        return None

    # Optional preprocessing
    if densify_factor > 0:
        points = densify_polygon(points, densify_factor)
    if smoothing_tolerance > 0:
        points = smooth_polygon(points, smoothing_tolerance)

    poly = _make_valid_polygon(points)
    if poly is None:
        return None

    tile_box = box(tile_x, tile_y, tile_x + tile_w, tile_y + tile_h)

    if not poly.intersects(tile_box):
        return None

    intersection = poly.intersection(tile_box)
    if intersection.is_empty:
        return None

    return _extract_polygons(intersection)


def clip_bbox_to_tile(
    bbox: List[float],
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
) -> Optional[List[float]]:
    """Clip a COCO-style ``[x, y, w, h]`` bbox to a tile.

    Returns the clipped ``[x, y, w, h]`` in tile-local coordinates,
    or ``None`` if the intersection is empty.
    """
    bx, by, bw, bh = bbox
    x1 = max(bx, tile_x)
    y1 = max(by, tile_y)
    x2 = min(bx + bw, tile_x + tile_w)
    y2 = min(by + bh, tile_y + tile_h)

    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0:
        return None

    # Convert to tile-local
    return [x1 - tile_x, y1 - tile_y, cw, ch]


# ------------------------------------------------------------------
# Coordinate transforms
# ------------------------------------------------------------------

def to_tile_coords(
    points: PointList,
    tile_x: int,
    tile_y: int,
) -> PointList:
    """Translate absolute image coordinates to tile-local coordinates."""
    return [(x - tile_x, y - tile_y) for x, y in points]


def normalize_to_yolo(
    points: PointList,
    tile_w: int,
    tile_h: int,
) -> List[float]:
    """Convert tile-local polygon to YOLO normalised flat coordinate list.

    Returns ``[x1/w, y1/h, x2/w, y2/h, ...]`` clamped to [0, 1].
    """
    result: List[float] = []
    for x, y in points:
        result.append(max(0.0, min(1.0, x / tile_w)))
        result.append(max(0.0, min(1.0, y / tile_h)))
    return result


def bbox_to_yolo_center(
    bbox: List[float], tile_w: int, tile_h: int
) -> List[float]:
    """Convert tile-local ``[x, y, w, h]`` to YOLO center format.

    Returns ``[xc, yc, w, h]`` normalised to [0, 1].
    """
    x, y, w, h = bbox
    xc = (x + w / 2) / tile_w
    yc = (y + h / 2) / tile_h
    nw = w / tile_w
    nh = h / tile_h
    return [
        max(0.0, min(1.0, xc)),
        max(0.0, min(1.0, yc)),
        max(0.0, min(1.0, nw)),
        max(0.0, min(1.0, nh)),
    ]


# ------------------------------------------------------------------
# Tile grid computation
# ------------------------------------------------------------------

def compute_tile_grid(
    img_width: int,
    img_height: int,
    tile_size: int,
    overlap_percent: float = 0.0,
    edge_behavior: str = "crop",
) -> List[Tuple[int, int, int, int]]:
    """Compute ``(x, y, w, h)`` for each tile covering the image.

    Args:
        img_width, img_height: Image dimensions.
        tile_size: Desired tile width/height.
        overlap_percent: Fractional overlap (0.0–1.0).
        edge_behavior: ``"crop"`` or ``"padding"``.

    Returns:
        List of ``(x, y, tile_w, tile_h)`` tuples.
    """
    if img_width <= tile_size and img_height <= tile_size:
        # Image fits in a single tile
        if edge_behavior == "padding":
            return [(0, 0, tile_size, tile_size)]
        return [(0, 0, img_width, img_height)]

    stride = max(1, int(tile_size * (1.0 - overlap_percent)))
    tiles: List[Tuple[int, int, int, int]] = []

    y = 0
    while y < img_height:
        x = 0
        while x < img_width:
            if edge_behavior == "padding":
                tw, th = tile_size, tile_size
            else:
                tw = min(tile_size, img_width - x)
                th = min(tile_size, img_height - y)
            tiles.append((x, y, tw, th))
            x += stride
            if x >= img_width:
                break
        y += stride
        if y >= img_height:
            break

    return tiles


# ------------------------------------------------------------------
# Area helpers
# ------------------------------------------------------------------

def polygon_area_ratio(
    points: PointList,
    tile_w: int,
    tile_h: int,
) -> float:
    """Polygon area as a fraction of tile area."""
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    tile_area = tile_w * tile_h
    if tile_area == 0:
        return 0.0
    return poly.area / tile_area


def bbox_area_ratio(
    bbox: List[float], tile_w: int, tile_h: int
) -> float:
    """Bbox area as a fraction of tile area."""
    _, _, bw, bh = bbox
    tile_area = tile_w * tile_h
    if tile_area == 0:
        return 0.0
    return (bw * bh) / tile_area


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _make_valid_polygon(
    points: PointList,
) -> Optional[Polygon]:
    """Create a valid Shapely Polygon, returning None on failure."""
    if len(points) < 3:
        return None
    try:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        return poly
    except Exception:
        return None


def _extract_polygons(
    geom: Union[Polygon, MultiPolygon],
) -> List[PointList]:
    """Extract coordinate lists from a Shapely geometry result."""
    polygons: List[PointList] = []

    if isinstance(geom, Polygon):
        coords = list(geom.exterior.coords)
        if len(coords) >= 4:
            polygons.append(coords)
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            coords = list(g.exterior.coords)
            if len(coords) >= 4:
                polygons.append(coords)
    else:
        # GeometryCollection or other — try to extract polygons
        if hasattr(geom, "geoms"):
            for g in geom.geoms:
                if isinstance(g, Polygon):
                    coords = list(g.exterior.coords)
                    if len(coords) >= 4:
                        polygons.append(coords)

    return polygons if polygons else None
