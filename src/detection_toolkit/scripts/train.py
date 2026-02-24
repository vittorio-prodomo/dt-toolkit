#!/usr/bin/env python3
"""Unified training entry point for all supported backends.

The backend is auto-detected from the model name or set explicitly.

Usage::

    # Ultralytics YOLO
    python -m detection_toolkit.scripts.train \\
        --model yolo11m-seg --dataset datasets/crack-seg \\
        --epochs 200 --batch 16 --imgsz 640

    # RF-DETR
    python -m detection_toolkit.scripts.train \\
        --model rf-detr-base --dataset datasets/coco_ds \\
        --format coco --epochs 100

    # Detectron2 Mask R-CNN
    python -m detection_toolkit.scripts.train \\
        --model mask_rcnn_R_50_FPN_3x --dataset datasets/coco_ds \\
        --format coco --epochs 50 --lr 0.0025

    # From a YAML config
    python -m detection_toolkit.scripts.train --config training_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from detection_toolkit.cli import make_parser


def parse_args(argv=None):
    p = make_parser("Train a model using the auto-detected backend.")

    # Config file (overrides all other args)
    p.add_argument("--config", "-C", type=Path, default=None,
                   metavar="PATH",
                   help="Path to a TrainingConfig YAML file.")

    # Model
    p.add_argument("--model", "-m", type=str, default="yolo11m-seg",
                   metavar="NAME",
                   help="Model name or path (default: yolo11m-seg).")
    p.add_argument("--backend", "-B", type=str, default=None,
                   choices=["ultralytics", "mask-rcnn", "rf-detr"],
                   help="Force a training backend (auto-detected if omitted).")

    # Data
    p.add_argument("--dataset", "-d", type=Path, required=False,
                   default=Path("."), metavar="DIR",
                   help="Dataset root directory.")
    p.add_argument("--format", "-f", choices=["yolo", "coco"], default="yolo",
                   help="Dataset format (default: yolo).")

    # Training params
    p.add_argument("--epochs", "-e", type=int, default=200, metavar="N")
    p.add_argument("--batch", "-b", type=int, default=32, metavar="N")
    p.add_argument("--imgsz", "-i", type=int, default=640, metavar="PX")
    p.add_argument("--patience", "-p", type=int, default=35, metavar="N")
    p.add_argument("--optimizer", "-o", type=str, default="AdamW", metavar="OPT")
    p.add_argument("--lr", "--learning-rate", "-lr", type=float, default=0.001,
                   dest="lr0", metavar="F")
    p.add_argument("--momentum", "-M", type=float, default=0.937, metavar="F")
    p.add_argument("--weight-decay", "-W", type=float, default=0.0005, metavar="F")
    p.add_argument("--warmup-epochs", "-E", type=float, default=3.0, metavar="F")

    # Runtime
    p.add_argument("--device", "-D", type=str, default="0", metavar="DEV")
    p.add_argument("--workers", "-w", type=int, default=16, metavar="N")
    p.add_argument("--project", "-P", type=str, default=None, metavar="DIR")
    p.add_argument("--name", "-n", type=str, default="run", metavar="NAME")
    p.add_argument("--resume", "-r", action="store_true")
    p.add_argument("--amp", "-a", action="store_true")
    p.add_argument("--seed", "-S", type=int, default=0, metavar="N")
    p.add_argument("--pretrained", "-pt", type=str, default="true", metavar="BOOL")
    p.add_argument("--fraction", "-F", type=float, default=1.0, metavar="F")
    p.add_argument("--save-period", "-sp", type=int, default=-1, metavar="N")

    # Extra backend-specific args (key=value pairs)
    p.add_argument("--extra", "-X", nargs="*", metavar="KEY=VALUE",
                   help="Backend-specific options as key=value pairs.")

    p.add_argument("--quiet", "-q", action="store_true")
    return p.parse_args(argv)


def _parse_extra(pairs):
    """Parse ``["key1=val1", "key2=val2"]`` into a dict with auto-typed values."""
    if not pairs:
        return {}
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid extra arg: {pair!r}. Expected key=value.")
        key, val = pair.split("=", 1)
        # Auto-type: int, float, bool, string
        if val.lower() in ("true", "false"):
            result[key] = val.lower() == "true"
        else:
            try:
                result[key] = int(val)
            except ValueError:
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
    return result


def main(argv=None):
    args = parse_args(argv)

    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")

    from detection_toolkit.training import TrainingConfig, get_trainer
    from detection_toolkit.core.models import DatasetFormat

    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig(
            model_name=args.model,
            backend=args.backend,
            dataset_path=args.dataset,
            dataset_format=DatasetFormat(args.format),
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.lr0,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            resume=args.resume,
            amp=args.amp,
            seed=args.seed,
            pretrained=args.pretrained,
            fraction=args.fraction,
            save_period=args.save_period,
            extra=_parse_extra(args.extra),
        )

    trainer = get_trainer(config)
    trainer.setup()

    if config.resume:
        results = trainer.resume()
    else:
        results = trainer.train()

    print(f"Training complete:")
    print(f"  Model:    {config.model_name}")
    print(f"  Backend:  {config.backend or 'auto-detected'}")
    print(f"  Results:  {list(results.keys())}")


if __name__ == "__main__":
    main()
