"""Detectron2 Mask R-CNN training backend.

Supports all Detectron2 model zoo architectures: Mask R-CNN, Faster R-CNN,
RetinaNet, and related architectures with various backbones.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import TrainerBase
from .config import TrainingConfig

logger = logging.getLogger(__name__)

# Model name → Detectron2 config file mapping
_CONFIG_MAP = {
    "mask_rcnn_R_50_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "mask_rcnn_R_101_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "mask_rcnn_X_101_32x8d_FPN_3x": "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
    "faster_rcnn_R_50_FPN_3x": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "faster_rcnn_R_101_FPN_3x": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "retinanet_R_50_FPN_3x": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
}

# Pretrained weight URLs
_COCO_WEIGHTS = {
    "mask_rcnn_R_50_FPN_3x": "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
    "mask_rcnn_R_101_FPN_3x": "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl",
    "faster_rcnn_R_50_FPN_3x": "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl",
    "retinanet_R_50_FPN_3x": "detectron2://COCO-Detection/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl",
}


class MaskRCNNTrainer(TrainerBase):
    """Training backend for Detectron2 architectures.

    Supports Mask R-CNN, Faster R-CNN, and RetinaNet from the Detectron2
    model zoo.  Datasets must be in COCO format; YOLO datasets are
    auto-converted.

    Supported model patterns::

        mask_rcnn_R_50_FPN_3x, mask_rcnn_R_101_FPN_3x,
        faster_rcnn_R_50_FPN_3x, retinanet_R_50_FPN_3x, ...
    """

    def setup(self) -> None:
        """Register the dataset and build the Detectron2 config."""
        self._ensure_coco_format()
        class_names, num_classes = self._register_dataset()
        self.cfg = self._build_cfg(num_classes)
        self.class_names = class_names

        logger.info(
            "Detectron2 setup: model=%s, classes=%d, dataset=%s",
            self.config.model_name, num_classes, self.config.dataset_path,
        )

    def train(self) -> Dict[str, Any]:
        """Run the Detectron2 training loop."""
        from detectron2.engine import DefaultTrainer

        if not hasattr(self, "cfg"):
            self.setup()

        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)

        logger.info(
            "Starting Detectron2 training: %d max_iter (≈%d epochs), batch=%d",
            self.cfg.SOLVER.MAX_ITER,
            self.config.epochs,
            self.config.batch_size,
        )
        trainer.train()
        self.results = {"output_dir": str(self.cfg.OUTPUT_DIR)}
        return self.results

    def resume(self, checkpoint: Optional[Path] = None) -> Dict[str, Any]:
        """Resume Detectron2 training from a checkpoint."""
        from detectron2.engine import DefaultTrainer

        if not hasattr(self, "cfg"):
            self.setup()

        if checkpoint is not None:
            self.cfg.defrost()
            self.cfg.MODEL.WEIGHTS = str(checkpoint)
            self.cfg.freeze()

        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=True)
        trainer.train()
        self.results = {"output_dir": str(self.cfg.OUTPUT_DIR)}
        return self.results

    def validate(self) -> Dict[str, Any]:
        """Run COCO evaluation."""
        from detectron2.engine import DefaultTrainer
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader

        if not hasattr(self, "cfg"):
            self.setup()

        model = DefaultTrainer.build_model(self.cfg)
        DefaultTrainer.resume_or_load(model)
        evaluator = COCOEvaluator(
            self.cfg.DATASETS.TEST[0], output_dir=self.cfg.OUTPUT_DIR,
        )
        loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
        results = inference_on_dataset(model, loader, evaluator)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_coco_format(self) -> None:
        """Convert dataset to COCO if needed."""
        if not self._needs_format_conversion("coco"):
            return

        logger.info("Dataset is YOLO format — converting to COCO for Detectron2")
        from ..formats.yolo_to_coco import YOLOToCOCOConverter
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="detectron2_coco_"))
        converter = YOLOToCOCOConverter(verbose=False)
        converter.convert(self.config.dataset_path, tmp)
        self.config.dataset_path = tmp

    def _register_dataset(self) -> Tuple[List[str], int]:
        """Register train/val splits with Detectron2."""
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data.datasets import register_coco_instances

        ds_root = self.config.dataset_path
        dataset_name = ds_root.name

        # Find annotation files
        ann_dir = ds_root / "annotations"
        class_names: List[str] = []
        num_classes = 0

        for split in ("train", "val"):
            # Try standard naming patterns
            json_path = None
            for pattern in [
                ann_dir / f"instances_{split}.json",
                ann_dir / f"{split}.json",
                ds_root / split / "_annotations.coco.json",
            ]:
                if pattern.exists():
                    json_path = pattern
                    break

            if json_path is None:
                continue

            # Find image directory
            img_dir = ds_root / "images" / split
            if not img_dir.exists():
                img_dir = ds_root / "images"
            if not img_dir.exists():
                img_dir = ds_root / split

            reg_name = f"{dataset_name}_{split}"

            # Skip if already registered
            if reg_name in DatasetCatalog:
                continue

            register_coco_instances(reg_name, {}, str(json_path), str(img_dir))

            # Parse categories from the first JSON we find
            if not class_names:
                class_names, num_classes = self._parse_categories(json_path)
                MetadataCatalog.get(reg_name).thing_classes = class_names

        if not class_names:
            raise RuntimeError(
                f"No COCO annotation files found in {ds_root}"
            )

        return class_names, num_classes

    @staticmethod
    def _parse_categories(json_path: Path) -> Tuple[List[str], int]:
        """Extract class names from a COCO JSON."""
        with open(json_path) as fh:
            data = json.load(fh)
        cats = sorted(data.get("categories", []), key=lambda c: c["id"])
        names = [c["name"] for c in cats]
        return names, len(names)

    def _build_cfg(self, num_classes: int):
        """Build the Detectron2 CfgNode."""
        from detectron2.config import get_cfg
        from detectron2 import model_zoo

        cfg = get_cfg()

        # Find config file
        config_rel = self._find_config()
        cfg.merge_from_file(model_zoo.get_config_file(config_rel))

        # Pretrained weights
        weights = self._resolve_weights()
        if weights:
            cfg.MODEL.WEIGHTS = weights

        # Dataset
        ds_name = self.config.dataset_path.name
        cfg.DATASETS.TRAIN = (f"{ds_name}_train",)
        cfg.DATASETS.TEST = (f"{ds_name}_val",)

        # Solver
        iters_per_epoch = max(1, self._estimate_dataset_size() // self.config.batch_size)
        max_iter = self.config.epochs * iters_per_epoch

        cfg.SOLVER.MAX_ITER = max_iter
        cfg.SOLVER.BASE_LR = self.config.lr0
        cfg.SOLVER.MOMENTUM = self.config.momentum
        cfg.SOLVER.WEIGHT_DECAY = self.config.weight_decay
        cfg.SOLVER.IMS_PER_BATCH = self.config.batch_size
        cfg.SOLVER.WARMUP_ITERS = int(self.config.warmup_epochs * iters_per_epoch)

        # LR schedule
        step1 = int(max_iter * 0.75)
        step2 = int(max_iter * 0.9)
        cfg.SOLVER.STEPS = (step1, step2)
        cfg.SOLVER.GAMMA = self.config.extra.get("gamma", 0.1)

        # Optimizer mapping
        opt_map = {"sgd": "SGD", "adam": "ADAM", "adamw": "ADAMW"}
        cfg.SOLVER.OPTIMIZER = opt_map.get(
            self.config.optimizer.lower(), "SGD"
        )

        # Input
        cfg.INPUT.MIN_SIZE_TRAIN = (self.config.imgsz,)
        cfg.INPUT.MAX_SIZE_TRAIN = self.config.imgsz
        cfg.INPUT.MIN_SIZE_TEST = self.config.imgsz
        cfg.INPUT.MAX_SIZE_TEST = self.config.imgsz

        # Model head
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        # Evaluation
        eval_period = max(1, self.config.extra.get("eval_period", 1)) * iters_per_epoch
        cfg.TEST.EVAL_PERIOD = eval_period

        # Data loader
        cfg.DATALOADER.NUM_WORKERS = self.config.workers

        # Output
        cfg.OUTPUT_DIR = str(self._resolve_project_dir())

        # AMP
        if self.config.amp:
            cfg.SOLVER.AMP.ENABLED = True

        # Seed
        cfg.SEED = self.config.seed

        # Checkpointing — let Detectron2 handle periodic saves
        cfg.SOLVER.CHECKPOINT_PERIOD = max(1, iters_per_epoch * max(1, self.config.save_period))

        cfg.freeze()
        return cfg

    def _find_config(self) -> str:
        """Map the model name to a Detectron2 config file path."""
        name = self.config.model_name

        if name in _CONFIG_MAP:
            return _CONFIG_MAP[name]

        # Fuzzy match
        lower = name.lower()
        for key, path in _CONFIG_MAP.items():
            if key.lower() in lower or lower in key.lower():
                return path

        # Default
        logger.warning(
            "Unknown Detectron2 model %r, falling back to mask_rcnn_R_50_FPN_3x",
            name,
        )
        return _CONFIG_MAP["mask_rcnn_R_50_FPN_3x"]

    def _resolve_weights(self) -> Optional[str]:
        """Resolve pretrained weight URL or path."""
        strategy = self.config.pretrained.lower()

        if strategy in ("false", "no", "0", "none"):
            return None

        if strategy in ("true", "yes", "1", "coco"):
            name = self.config.model_name
            if name in _COCO_WEIGHTS:
                return _COCO_WEIGHTS[name]
            # Fuzzy
            for key, url in _COCO_WEIGHTS.items():
                if key.lower() in name.lower():
                    return url
            return None

        if strategy == "imagenet":
            return "detectron2://ImageNetPretrained/MSRA/R-50.pkl"

        # Custom path
        return self.config.pretrained

    def _estimate_dataset_size(self) -> int:
        """Estimate the number of training images."""
        ds = self.config.dataset_path
        ann_file = ds / "annotations" / "instances_train.json"
        if ann_file.exists():
            with open(ann_file) as fh:
                data = json.load(fh)
            return len(data.get("images", []))
        # Fallback: count images in directory
        img_dir = ds / "images" / "train"
        if img_dir.exists():
            return len(list(img_dir.iterdir()))
        return 100  # safe fallback
