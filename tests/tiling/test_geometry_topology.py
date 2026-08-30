"""Tile clipping must preserve annotation TOPOLOGY: holes and disjoint parts.

Regression tests for two silent data-corruption bugs found 2026-08-29 in
a tiled crack dataset, where only 1 of 32 crack-bearing tiles carried a mask
matching the original annotation cropped to the same window:

* holes were discarded (``exterior.coords`` only) -> a ring-shaped annotation
  came back FILLED, measured 3.4x inflation;
* disjoint parts were discarded (``max(geoms, key=area)``) -> a two-stroke
  annotation came back as its larger stroke alone, exactly 50% of the pixels.

Both are invisible on simple single-part hole-free polygons, which is why they
survived for months.

Note on representation: the tiler emits COCO, whose ``segmentation`` is a LIST of
polygons that decoders UNION. So disjoint parts need no bridging -- they are
separate entries. Holes cannot be expressed that way (a second ring does not
subtract) and must be bridged into their exterior with a zero-width seam.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from detection_toolkit.tiling.geometry import clip_polygon_to_tile

CANVAS = 400
TILE = (0, 0, CANVAS, CANVAS)


def rasterise(polys) -> int:
    """Even-odd fill of the returned rings, as every downstream consumer does."""
    m = np.zeros((CANVAS, CANVAS), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.array(p, np.int32)], 1)
    return int(m.sum())


def holed_ring(outer, hole):
    """Exactly how a holed annotation reaches the tiler: cocoyolo's hole bridge."""
    from cocoyolo.geometry import bridge_holes
    return bridge_holes([list(p) for p in outer], [np.asarray(hole, dtype=float)])


def disjoint_ring(*parts):
    """Exactly how a multi-part annotation reaches the tiler: doubled keyhole.

    Built with the real encoder rather than a hand-rolled one -- a naive bridge
    that traces the second part in reverse gives the loops opposite orientation,
    so their signed areas cancel and the fixture tests nothing.
    """
    from cocoyolo.geometry import bridge_disjoint
    return bridge_disjoint([[list(p) for p in part] for part in parts])


# --------------------------------------------------------------------------
# holes
# --------------------------------------------------------------------------

def test_a_hole_is_not_filled_in():
    outer = [(10, 10), (390, 10), (390, 390), (10, 390)]
    hole = [(100, 100), (300, 100), (300, 300), (100, 300)]
    true_area = 380 * 380 - 200 * 200                      # 104,400

    out = clip_polygon_to_tile(holed_ring(outer, hole), *TILE,
                               densify_factor=0, smoothing_tolerance=0)
    assert out, "clipping dropped the annotation entirely"
    got = rasterise(out)
    assert got == pytest.approx(true_area, rel=0.05), (
        f"hole not preserved: {got} px vs {true_area} expected "
        f"({got / true_area:.1f}x — a filled square is 144400)"
    )


# --------------------------------------------------------------------------
# disjoint parts
# --------------------------------------------------------------------------

STROKE_A = [(20, 40), (200, 40), (200, 70), (20, 70)]      # 180 x 30
STROKE_B = [(20, 300), (200, 300), (200, 330), (20, 330)]  # same, disjoint


def test_both_disjoint_parts_survive():
    true_area = 180 * 30 * 2
    out = clip_polygon_to_tile(disjoint_ring(STROKE_A, STROKE_B), *TILE,
                               densify_factor=0, smoothing_tolerance=0)
    assert out, "clipping dropped the annotation entirely"
    got = rasterise(out)
    assert got == pytest.approx(true_area, rel=0.05), (
        f"parts lost: {got} px vs {true_area} expected "
        f"(half would be {true_area // 2} — the largest-part bug)"
    )


def test_disjoint_parts_come_back_as_separate_polygons():
    """COCO unions a list of rings, so parts must NOT be re-bridged here."""
    out = clip_polygon_to_tile(disjoint_ring(STROKE_A, STROKE_B), *TILE,
                               densify_factor=0, smoothing_tolerance=0)
    assert len(out) == 2, f"expected 2 separate polygons, got {len(out)}"


# --------------------------------------------------------------------------
# the healthy majority must not move
# --------------------------------------------------------------------------

def test_a_simple_polygon_is_untouched():
    simple = [(50, 50), (250, 50), (250, 200), (50, 200)]
    out = clip_polygon_to_tile(simple, *TILE,
                               densify_factor=0, smoothing_tolerance=0)
    assert len(out) == 1
    got = {(round(x), round(y)) for x, y in out[0]}
    assert got == {(50, 50), (250, 50), (250, 200), (50, 200)}


def test_a_simple_polygon_clipped_by_the_tile_edge_still_works():
    simple = [(300, 50), (500, 50), (500, 200), (300, 200)]   # runs off the right
    out = clip_polygon_to_tile(simple, *TILE,
                               densify_factor=0, smoothing_tolerance=0)
    assert len(out) == 1
    assert rasterise(out) == pytest.approx(100 * 150, rel=0.05)


# --------------------------------------------------------------------------
# a real clip that CREATES disjointness
# --------------------------------------------------------------------------

def test_a_c_shape_clipped_into_two_arms_keeps_both():
    """The C opens right; the right-half tile cuts it into two arms."""
    c = [(10, 10), (150, 10), (150, 40), (60, 40),
         (60, 160), (150, 160), (150, 190), (10, 190)]
    out = clip_polygon_to_tile(c, 100, 0, 100, 200,
                               densify_factor=0, smoothing_tolerance=0)
    assert out and len(out) == 2, f"expected both arms, got {len(out or [])}"


# --------------------------------------------------------------------------
# the PRODUCTION defaults (densify 0.5, smoothing 0.1) — what actually ran
# --------------------------------------------------------------------------

def test_holes_survive_with_the_default_densify_and_smoothing():
    outer = [(10, 10), (390, 10), (390, 390), (10, 390)]
    hole = [(100, 100), (300, 100), (300, 300), (100, 300)]
    out = clip_polygon_to_tile(holed_ring(outer, hole), *TILE)   # defaults
    assert out
    assert rasterise(out) == pytest.approx(380 * 380 - 200 * 200, rel=0.05)


def test_disjoint_parts_survive_with_the_default_densify_and_smoothing():
    out = clip_polygon_to_tile(disjoint_ring(STROKE_A, STROKE_B), *TILE)
    assert out
    assert rasterise(out) == pytest.approx(180 * 30 * 2, rel=0.05)


def test_densify_alone_never_drops_a_part():
    """densify_polygon ran BEFORE validity repair, so its own part-dropping
    destroyed annotations that nothing downstream could recover."""
    from detection_toolkit.tiling.geometry import densify_polygon

    ring = disjoint_ring(STROKE_A, STROKE_B)
    dense = densify_polygon(ring, 0.5)
    assert rasterise([dense]) == pytest.approx(rasterise([ring]), rel=0.05), (
        "densification changed the rasterised area — a part or hole was lost"
    )
