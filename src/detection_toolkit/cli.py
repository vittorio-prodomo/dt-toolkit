"""Shared CLI helpers for all detection_toolkit scripts."""

import argparse
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path=_PROJECT_ROOT / ".env"):
    """Load KEY=VALUE pairs from *path* into ``os.environ``.

    Already-set environment variables take precedence (they are never
    overwritten).  Missing file is silently ignored.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()

MAX_HELP_POSITION = int(os.environ.get("DT_MAX_HELP_POSITION", 40))


def make_parser(description, raw_text=False):
    """Create an ``ArgumentParser`` with a standardised wide help layout.

    Parameters
    ----------
    description : str
        Parser description shown in ``--help``.
    raw_text : bool
        If *True*, use ``RawTextHelpFormatter`` (preserves newlines in
        help strings) instead of the default ``HelpFormatter``.
    """
    base = argparse.RawTextHelpFormatter if raw_text else argparse.HelpFormatter
    return argparse.ArgumentParser(
        description=description,
        formatter_class=lambda prog: base(prog, max_help_position=MAX_HELP_POSITION),
    )


def formatter_class(raw_text=False):
    """Return a ``HelpFormatter`` class suitable for ``add_parser()``.

    Use this when adding subparsers that should share the same column
    width as the top-level parser::

        sub.add_parser("name", formatter_class=formatter_class())
    """
    base = argparse.RawTextHelpFormatter if raw_text else argparse.HelpFormatter
    return lambda prog: base(prog, max_help_position=MAX_HELP_POSITION)
