# detection-toolkit

Modular building blocks for object-detection & instance-segmentation workflows.

## Installation

```bash
pip install -e .                # core only
pip install -e ".[fiftyone]"    # + FiftyOne integration
pip install -e ".[train]"       # + Ultralytics training
pip install -e ".[all]"         # everything
```

## CLI cheat sheet

After installation the following commands are available.
Run any command with `--help` (or `-h`) for full usage details.

### Inspect a dataset

```bash
dt-ingest datasets/my_ds                  # auto-detect format
dt-ingest datasets/my_ds -f yolo          # force YOLO
```

### Convert between formats

```bash
dt-convert datasets/coco_ds out/yolo_ds -t yolo
dt-convert datasets/yolo_ds out/coco_ds -t coco
```

### Tile into fixed-size patches

```bash
dt-tile datasets/ds out/tiled -t 640 -o 0.3
dt-tile datasets/ds out/tiled -c tiling_config.yaml
```

### Filter or merge classes

```bash
dt-filter datasets/ds out/filtered -a keep  -c crack spall
dt-filter datasets/ds out/filtered -a remove -c efflorescence
dt-filter datasets/ds out/merged   -m crack=mild_crack,splitting_crack
```

### Merge multiple datasets

```bash
dt-merge out/merged -d datasets/ds_a datasets/ds_b -c merge_config.yaml
dt-merge out/merged -d datasets/ds_a datasets/ds_b \
    -m crack=mild_crack,splitting_crack -m spall=mechanical_spall,spall
```

### Train a model

```bash
dt-train -m yolo11m-seg -d datasets/crack-seg -e 200 -b 16 -i 640
dt-train -m rf-detr-base -d datasets/coco_ds -f coco -e 100
dt-train -C training_config.yaml
```

### FiftyOne integration

```bash
# Import from disk
dt-import-fo disk datasets/my_ds my_dataset -f yolo

# Import from CVAT
dt-import-fo cvat my_dataset -P "My CVAT Project"

# Export to disk
dt-export-fo my_dataset out/exported -t yolo -s train val
```

### Quick reference

| Command | Description | Extras required |
|---|---|---|
| `dt-ingest` | Ingest a dataset and display its metadata | — |
| `dt-convert` | Convert between YOLO and COCO formats | — |
| `dt-tile` | Tile a dataset into fixed-size patches | — |
| `dt-filter` | Filter or merge classes within a dataset | — |
| `dt-merge` | Merge multiple datasets with class remapping | — |
| `dt-train` | Train a model (auto-detected backend) | `.[train]` / `.[train-detectron2]` / `.[train-rfdetr]` |
| `dt-import-fo` | Import a dataset into FiftyOne (disk or CVAT) | `.[fiftyone]` |
| `dt-export-fo` | Export a FiftyOne dataset to disk | `.[fiftyone]` |
