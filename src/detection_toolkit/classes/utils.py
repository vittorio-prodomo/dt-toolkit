"""Class-name utilities: normalisation and comparison helpers."""

import re


def normalize_class_name(name: str) -> str:
    """Normalise a class name for fuzzy matching.

    Lowercases, strips whitespace, and collapses runs of ``_``/``-``/space
    into a single underscore.  This allows ``"Exposed Rebar"``,
    ``"exposed-rebar"`` and ``"exposed_rebar"`` to compare as equal.
    """
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-_]+", "_", s)
    return s
