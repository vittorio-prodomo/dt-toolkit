#!/usr/bin/env python3
"""Rename a FiftyOne dataset and its on-disk directory atomically.

Updates the FiftyOne dataset name, all sample filepaths, and renames the
directory on disk.

Usage::

    # Simple: dataset name == directory name, rename both
    dt-rename old_name new_name

    # Explicit: dataset and directory have different names
    dt-rename old_name new_name --old-dir old_dir_name --new-dir new_dir_name

    # Dry run: show what would change without doing anything
    dt-rename old_name new_name --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASETS_ROOT = Path("/path/to/datasets")


def main():
    parser = argparse.ArgumentParser(
        description="Rename a FiftyOne dataset and its on-disk directory.",
    )
    parser.add_argument("old_name", help="Current FiftyOne dataset name")
    parser.add_argument("new_name", help="New FiftyOne dataset name")
    parser.add_argument(
        "--old-dir",
        help="Current directory name under datasets/ (default: same as old_name)",
    )
    parser.add_argument(
        "--new-dir",
        help="New directory name under datasets/ (default: same as new_name)",
    )
    parser.add_argument(
        "--datasets-root", type=Path, default=DATASETS_ROOT,
        help=f"Parent directory containing dataset folders (default: {DATASETS_ROOT})",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be done without making changes",
    )
    args = parser.parse_args()

    old_dir_name = args.old_dir or args.old_name
    new_dir_name = args.new_dir or args.new_name
    old_dir = args.datasets_root / old_dir_name
    new_dir = args.datasets_root / new_dir_name

    import fiftyone as fo

    # Resolve current state
    datasets = fo.list_datasets()
    has_old_ds = args.old_name in datasets
    has_new_ds = args.new_name in datasets
    rename_ds = args.old_name != args.new_name
    rename_dir = old_dir_name != new_dir_name

    # Determine FiftyOne dataset to load
    if has_old_ds:
        ds_to_load = args.old_name
    elif rename_ds and has_new_ds:
        ds_to_load = args.new_name
        logger.info("Dataset already renamed to '%s' — skipping dataset rename", args.new_name)
        rename_ds = False
    else:
        logger.error("FiftyOne dataset '%s' not found", args.old_name)
        sys.exit(1)

    if rename_ds and has_new_ds:
        logger.error("FiftyOne dataset '%s' already exists", args.new_name)
        sys.exit(1)

    # Determine directory state
    dir_already_done = False
    if rename_dir:
        if new_dir.is_dir() and not old_dir.is_dir():
            logger.info("Directory already renamed to '%s' — skipping directory rename", new_dir.name)
            dir_already_done = True
        elif old_dir.is_dir() and new_dir.exists():
            logger.error("Both '%s' and '%s' exist — cannot rename", old_dir, new_dir)
            sys.exit(1)
        elif not old_dir.is_dir() and not new_dir.is_dir():
            logger.error("Neither '%s' nor '%s' exists", old_dir, new_dir)
            sys.exit(1)

    ds = fo.load_dataset(ds_to_load)
    old_dir_str = str(old_dir)
    new_dir_str = str(new_dir)

    # Count affected filepaths
    paths = ds.values("filepath")
    affected = sum(1 for p in paths if old_dir_str in p)

    # Summary
    logger.info("Plan:")
    if rename_ds:
        logger.info("  Dataset name: '%s' -> '%s'", args.old_name, args.new_name)
    if rename_dir and not dir_already_done:
        logger.info("  Directory:    '%s' -> '%s'", old_dir, new_dir)
    if rename_dir and affected > 0:
        logger.info("  Filepaths:    %d / %d samples to update", affected, len(ds))
    if not rename_ds and (not rename_dir or dir_already_done) and affected == 0:
        logger.info("  Nothing to do — everything already renamed.")
        return

    if args.dry_run:
        logger.info("Dry run — no changes made.")
        return

    # Step 1: Rename directory on disk
    if rename_dir and not dir_already_done:
        logger.info("Renaming directory...")
        old_dir.rename(new_dir)

    # Step 2: Update filepaths in FiftyOne
    if rename_dir and affected > 0:
        logger.info("Updating %d filepaths...", affected)
        new_paths = [p.replace(old_dir_str, new_dir_str) for p in paths]
        ds.set_values("filepath", new_paths)

    # Step 3: Rename FiftyOne dataset
    if rename_ds:
        logger.info("Renaming FiftyOne dataset...")
        ds.name = args.new_name
        ds.save()

    logger.info("Done.")


if __name__ == "__main__":
    main()
