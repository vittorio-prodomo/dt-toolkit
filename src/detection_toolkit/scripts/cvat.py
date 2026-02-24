#!/usr/bin/env python3
"""CVAT task management utilities.

Usage::

    dt-cvat split --task 41 --cuts 250
    dt-cvat split --task 41 --equal 3
    dt-cvat split --task 41 --job 54 --cuts 100 300
"""

import argparse
import logging
import sys

from detection_toolkit.cli import make_parser, formatter_class


def _add_connection_opts(p: argparse.ArgumentParser) -> None:
    """Add CVAT connection flags (override DT_CVAT_* env vars)."""
    g = p.add_argument_group("CVAT connection (or set DT_CVAT_* in .env)")
    g.add_argument("--url", type=str, default=None, metavar="URL",
                   help="CVAT server URL (default: DT_CVAT_URL).")
    g.add_argument("--username", "-u", type=str, default=None, metavar="USER",
                   help="CVAT username (default: DT_CVAT_USERNAME).")
    g.add_argument("--password", type=str, default=None, metavar="PASS",
                   help="CVAT password (default: DT_CVAT_PASSWORD).")
    g.add_argument("--organization", type=str, default=None, metavar="ORG",
                   help="CVAT organization slug (default: DT_CVAT_ORGANIZATION).")


def parse_args(argv=None):
    p = make_parser("CVAT task management utilities.", raw_text=True)
    sub = p.add_subparsers(dest="subcommand", title="subcommands")

    # ---- split -----------------------------------------------------
    sp = sub.add_parser(
        "split",
        help="Split a task's job(s) into a new task with more jobs.",
        description=(
            "Create a new CVAT task by re-distributing frames across jobs.\n"
            "The original task is never modified.\n\n"
            "Examples:\n"
            "  dt-cvat split --task 41 --cuts 250           # 2 jobs: 0-249, 250-end\n"
            "  dt-cvat split --task 41 --cuts 100 300       # 3 jobs\n"
            "  dt-cvat split --task 41 --equal 3            # 3 equal-ish jobs\n"
            "  dt-cvat split --task 41 --job 54 --cuts 250  # split one job in a multi-job task"
        ),
        formatter_class=formatter_class(raw_text=True),
    )
    sp.add_argument("--task", "-t", type=int, required=True,
                    metavar="ID", help="Source task ID.")
    sp.add_argument("--job", "-j", type=int, default=None,
                    metavar="ID",
                    help="Job ID to split (required if task has multiple jobs).")

    split_mode = sp.add_mutually_exclusive_group(required=True)
    split_mode.add_argument(
        "--cuts", "-c", type=int, nargs="+", metavar="FRAME",
        help="Frame indices where new jobs begin (e.g. 250 or 100 300).",
    )
    split_mode.add_argument(
        "--equal", "-e", type=int, metavar="N",
        help="Split into N roughly equal jobs.",
    )

    sp.add_argument("--dry-run", "-n", action="store_true",
                    help="Show the planned split without creating anything.")

    _add_connection_opts(sp)

    verbosity = sp.add_mutually_exclusive_group()
    verbosity.add_argument("--quiet", "-q", action="store_true",
                           help="Suppress informational output.")
    verbosity.add_argument("--verbose", "-v", action="store_true",
                           help="Show debug output.")

    args = p.parse_args(argv)

    if args.subcommand is None:
        p.print_help()
        sys.exit(0)

    return args


def _resolve_cuts(args, client) -> list[int]:
    """Turn ``--equal N`` into concrete cut points, or return ``--cuts`` as-is."""
    if args.cuts is not None:
        return args.cuts

    # --equal N: need to know the frame count
    task = client.sdk.tasks.retrieve(args.task)
    if args.job is not None:
        job = client.sdk.jobs.retrieve(args.job)
        n_frames = len(job.get_frames_info())
    else:
        jobs = task.get_jobs()
        annotation_jobs = [j for j in jobs if j.type == "annotation"]
        if len(annotation_jobs) != 1:
            raise ValueError(
                f"Task {args.task} has {len(annotation_jobs)} annotation jobs. "
                "Use --job to specify which one to split."
            )
        n_frames = len(annotation_jobs[0].get_frames_info())

    n = args.equal
    if n < 2:
        raise ValueError("--equal must be >= 2")
    chunk = n_frames / n
    return [round(chunk * i) for i in range(1, n)]


def main(argv=None):
    args = parse_args(argv)

    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")

    if args.subcommand == "split":
        from detection_toolkit.cvat.client import CvatClient
        from detection_toolkit.cvat.splitter import TaskSplitter

        client = CvatClient(
            url=args.url,
            username=args.username,
            password=args.password,
            organization=args.organization,
        )

        cuts = _resolve_cuts(args, client)

        if args.dry_run:
            _dry_run(client, args.task, args.job, cuts)
            return

        splitter = TaskSplitter(
            client, args.task, cuts, job_id=args.job,
        )
        result = splitter.run()

        print(f"\nSplit complete — new task {result.new_task_id}: "
              f"'{result.new_task_name}'")
        for j in result.jobs:
            print(f"  Job {j['job_id']}: {j['num_frames']} frames, "
                  f"{j.get('num_shapes', 0)} annotations "
                  f"({j['first_frame']} .. {j['last_frame']})")
        print(f"  Total annotations: {result.annotations_total}")


def _dry_run(client, task_id: int, job_id, cuts: list[int]) -> None:
    """Print the planned split without creating anything."""
    task = client.sdk.tasks.retrieve(task_id)
    jobs = task.get_jobs()

    target_job_id = job_id
    if target_job_id is None:
        annotation_jobs = [j for j in jobs if j.type == "annotation"]
        if len(annotation_jobs) == 1:
            target_job_id = annotation_jobs[0].id
        else:
            raise ValueError(
                f"Task has {len(annotation_jobs)} annotation jobs — use --job."
            )

    print(f"Dry run — task {task_id}: '{task.name}', {task.size} frames")
    print(f"Target job: {target_job_id}")
    print(f"Cut points: {cuts}")
    print()

    for j in jobs:
        frames = j.get_frames_info()
        n = len(frames)
        if j.id == target_job_id:
            boundaries = [0] + sorted(cuts) + [n]
            for i in range(len(boundaries) - 1):
                s, e = boundaries[i], boundaries[i + 1]
                print(f"  [NEW] Job: frames {s}-{e - 1} ({e - s} frames)")
        else:
            print(f"  [KEEP] Job {j.id}: {n} frames")


if __name__ == "__main__":
    main()
