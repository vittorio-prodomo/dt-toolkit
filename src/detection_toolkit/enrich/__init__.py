"""Dataset enrichment: compute and store derived attributes.

Sub-modules:

- ``enricher`` — area_px / area_rel for label annotations
- ``gps`` — GPS lat/lon extraction from EXIF + brain visualization

Public API::

    from detection_toolkit.enrich import DatasetEnricher
    from detection_toolkit.enrich.gps import enrich_gps
"""

from .enricher import DatasetEnricher

__all__ = ["DatasetEnricher"]
