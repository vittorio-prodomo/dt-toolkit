"""Geometry operations for polygon processing during tiling.

Provides densification, tile-clipping and topology-preserving simplification of
polygons using Shapely.

TOPOLOGY CONTRACT: an annotation may be multi-part and may contain holes.
Disjoint parts are returned as SEPARATE rings (COCO unions a list of rings);
holes are bridged into their part's exterior with a zero-width seam, because the
polygon format cannot subtract. Never discard a part, never drop an interior.
"""

import logging
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
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

    # Resample the ring AS GIVEN. This used to route through Shapely, which meant
    # `max(geoms, key=area)` on a bridged ring (dropping every part but the
    # largest) and `exterior.coords` (dropping holes) -- the same two bugs as
    # downstream, but firing FIRST, before anything could recover them.
    # Densification is pure resampling: it needs a perimeter, not a validity
    # opinion, so compute the perimeter from the points themselves.
    ring = list(points)
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    perimeter = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        perimeter += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    max_dist = perimeter * factor
    if max_dist <= 0:
        return points

    return _densify_ring(ring, max_dist)


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

    # Order matters: densify -> un-bridge -> clip -> simplify -> bridge holes.
    # Simplification used to run FIRST, on the raw ring, where it both dropped
    # holes and risked prying a zero-width seam open into a filled wedge.
    if densify_factor > 0:
        points = densify_polygon(points, densify_factor)

    poly = _make_valid_polygon(points)
    if poly is None:
        return None

    tile_box = box(tile_x, tile_y, tile_x + tile_w, tile_y + tile_h)

    if not poly.intersects(tile_box):
        return None

    intersection = poly.intersection(tile_box)
    if intersection.is_empty:
        return None

    intersection = _simplify_geometry(intersection, smoothing_tolerance)
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
    points: Sequence[Sequence[float]],
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


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Compute polygon area using the shoelace formula."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x0, y0 = points[i][0], points[i][1]
        x1, y1 = points[(i + 1) % n][0], points[(i + 1) % n][1]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def bbox_from_polygons(
    polygons: Sequence[Sequence[Sequence[float]]],
    clamp_w: float = float("inf"),
    clamp_h: float = float("inf"),
) -> List[float]:
    """Compute a COCO-style ``[x, y, w, h]`` bbox spanning all *polygons*.

    Optionally clamp to ``[0, clamp_w]`` × ``[0, clamp_h]``.
    """
    all_xs = [pt[0] for poly in polygons for pt in poly]
    all_ys = [pt[1] for poly in polygons for pt in poly]
    x_min = max(0.0, min(all_xs))
    y_min = max(0.0, min(all_ys))
    x_max = min(clamp_w, max(all_xs))
    y_max = min(clamp_h, max(all_ys))
    return [x_min, y_min, x_max - x_min, y_max - y_min]


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
) -> Optional[Union[Polygon, MultiPolygon]]:
    """Create a valid Shapely geometry from a ring, returning None on failure.

    ``buffer(0)`` is what UN-BRIDGES an incoming ring: annotations reach us as a
    single self-touching ring (``cocoyolo.mask_to_polygons`` bridges disjoint
    parts and holes through zero-width seams, because YOLO-seg holds only one
    ring per instance), which Shapely correctly reports as invalid and correctly
    resolves back into its true parts.

    The result may therefore legitimately be a MultiPolygon, and every part is
    real annotation. This function used to keep only ``max(geoms, key=area)``,
    silently deleting every other part -- measured as exactly 50% pixel loss on a
    two-stroke crack. Return the whole geometry; the caller decides how to encode
    it.
    """
    if len(points) < 3:
        return None
    try:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        return poly
    except Exception:
        return None


def _simplify_geometry(
    geom: Union[Polygon, MultiPolygon], tolerance: float,
) -> Union[Polygon, MultiPolygon]:
    """Topology-preserving simplification of a whole geometry.

    Runs AFTER clipping and BEFORE any hole bridging. Simplifying a bridged ring
    is unsafe: the two coincident edges of a zero-width seam can be perturbed
    apart, turning an invisible seam into a filled wedge.
    """
    if tolerance <= 0:
        return geom
    try:
        simplified = geom.simplify(tolerance, preserve_topology=True)
    except Exception:
        return geom
    if simplified.is_empty or not isinstance(simplified, (Polygon, MultiPolygon)):
        return geom
    return simplified


def _ring_with_holes_bridged(poly: Polygon) -> Optional[PointList]:
    """One COCO-ready ring for a single polygon part, holes bridged in.

    COCO's ``segmentation`` is a list of rings that decoders UNION, so a hole
    cannot be a separate entry -- it would add area instead of removing it
    (verified: ``[outer, hole]`` decodes to the same pixel count as ``[outer]``).
    A hole survives only as a zero-width detour spliced into its exterior, which
    is what ``cocoyolo.bridge_holes`` builds.
    """
    exterior = list(poly.exterior.coords)[:-1]          # drop the closing repeat
    if len(exterior) < 3:
        return None
    if not poly.interiors:
        return list(poly.exterior.coords)
    holes = [
        np.asarray(list(ring.coords)[:-1], dtype=float)
        for ring in poly.interiors
        if len(ring.coords) >= 4
    ]
    if not holes:
        return list(poly.exterior.coords)
    from cocoyolo.geometry import bridge_holes

    return bridge_holes([list(p) for p in exterior], holes)


def _extract_polygons(
    geom: Union[Polygon, MultiPolygon],
) -> Optional[List[PointList]]:
    """Coordinate lists from a Shapely result, ONE per disjoint part.

    Disjoint parts stay separate -- COCO unions a list of rings, so parts need no
    bridging here and must not get one (a premature bridge adds a ~1px seam that
    the format did not require). Holes, which the format cannot express, are
    bridged into their own part's exterior.

    This function used to return bare ``exterior.coords``, discarding interiors
    and so returning a ring-shaped annotation FILLED -- measured 3.4x inflation.
    """
    if isinstance(geom, Polygon):
        parts = [geom]
    elif hasattr(geom, "geoms"):
        parts = [g for g in geom.geoms if isinstance(g, Polygon)]
    else:
        parts = []

    polygons: List[PointList] = []
    for part in parts:
        if part.is_empty:
            continue
        ring = _ring_with_holes_bridged(part)
        if ring and len(ring) >= 4:
            polygons.append(ring)

    return polygons if polygons else None
