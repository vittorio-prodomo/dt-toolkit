"""Extract GPS coordinates from image EXIF and create a FiftyOne brain visualization."""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

import fiftyone as fo
import fiftyone.brain as fob

logger = logging.getLogger(__name__)


def _dms_to_decimal(dms, ref: str) -> float:
    """Convert EXIF DMS (degrees, minutes, seconds) tuple to decimal degrees."""
    d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
    dec = d + m / 60 + s / 3600
    if ref in ("S", "W"):
        dec = -dec
    return dec


def _extract_gps(filepath: str) -> tuple[float, float] | None:
    """Extract (lat, lon) from an image's EXIF GPS data. Returns None if absent."""
    try:
        img = Image.open(filepath)
        exif = img.getexif()
        if not exif:
            return None
        gps = exif.get_ifd(0x8825)
        if not gps or 1 not in gps or 2 not in gps or 3 not in gps or 4 not in gps:
            return None
        lat = _dms_to_decimal(gps[2], gps[1])
        lon = _dms_to_decimal(gps[4], gps[3])
        return lat, lon
    except Exception:
        return None


def enrich_gps(
    dataset: fo.Dataset,
    brain_key: str = "gps_location",
    verbose: bool = True,
) -> bool:
    """Extract GPS lat/lon from EXIF and store as sample fields + brain visualization.

    Adds ``gps_lat`` and ``gps_lon`` float fields to each sample (None if
    no GPS data).  Creates a brain visualization using the coordinates so
    they appear in the Embeddings panel.

    Returns True if at least one sample had GPS data.
    """
    filepaths = dataset.values("filepath")
    sample_ids = dataset.values("id")
    n = len(filepaths)

    lats = [None] * n
    lons = [None] * n
    gps_ids = []
    gps_points = []

    iterator = enumerate(zip(filepaths, sample_ids))
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=n, desc="Extracting GPS", unit="img")

    for i, (fp, sid) in iterator:
        result = _extract_gps(fp)
        if result is not None:
            lat, lon = result
            lats[i] = lat
            lons[i] = lon
            gps_ids.append(sid)
            gps_points.append([lon, lat])  # x=lon, y=lat

    gps_count = len(gps_ids)
    logger.info("GPS data found in %d / %d samples", gps_count, n)

    if gps_count == 0:
        return False

    # Store as sample fields
    dataset.set_values("gps_lat", lats)
    dataset.set_values("gps_lon", lons)
    dataset.add_dynamic_sample_fields()
    logger.info("Stored gps_lat and gps_lon fields")

    # Create brain visualization
    points = np.array(gps_points, dtype=np.float64)
    view = dataset.select(gps_ids)

    if brain_key in dataset.list_brain_runs():
        dataset.delete_brain_run(brain_key)

    fob.compute_visualization(
        view,
        points=points,
        brain_key=brain_key,
        verbose=verbose,
    )
    logger.info("Created brain visualization '%s'", brain_key)

    return True
