# detection-toolkit

Modular building blocks for object-detection & instance-segmentation workflows.

## Installation

```bash
pip install -e .                # core only
pip install -e ".[fiftyone]"    # + FiftyOne integration
pip install -e ".[train]"       # + Ultralytics training
pip install -e ".[all]"         # everything
```

## Using it in another project

This toolkit is not tied to its parent project (the project it grew up in). From
any [uv](https://docs.astral.sh/uv/) project:

```toml
[project]
dependencies = ["detection-toolkit[all]"]

[tool.uv.sources]
detection-toolkit = { git = "https://github.com/vittorio-prodomo/detection-toolkit", tag = "v0.1.0" }
```

The toolkit's own `[tool.uv.sources]` pull **our forks** (cocoyolo, fiftyone, eta, fiftyone-brain) from GitHub at pinned
commits — uv applies a dependency's sources for you (verified 2026-10-27), together
with the CUDA 12.8 PyTorch index where it applies. Plain `pip` ignores those sources and
would install the upstream packages, without our patches: use uv.

With the **FiftyOne** extra, add the overrides our FiftyOne forks need (their version
numbers don't satisfy upstream's pins; overrides are never inherited, so the consuming
project must declare them):

```toml
[tool.uv]
override-dependencies = ["fiftyone-brain>=0", "voxel51-eta>=0"]
constraint-dependencies = ["graphql-core<3.3"]

[tool.uv.sources]
voxel51-eta = { git = "https://github.com/vittorio-prodomo/eta", rev = "46eb981821a1d36dd5c17efd74afc0949c631c10" }
fiftyone-brain = { git = "https://github.com/vittorio-prodomo/fiftyone-brain", rev = "f49cac2b2e379d7d1f644f98e43967a30c408885" }
```

To develop against a **local checkout** of a fork instead (what its parent project
does), list the package in `override-dependencies` and give it a `path` source — a path
source alone is ignored for a package that only arrives through this toolkit.

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

# Tile unlabeled images (auto-detected, preserves directory structure)
dt-tile images/ out/tiled -t 640

# Tile unlabeled images with flattened output
dt-tile images/ out/tiled -t 640 --flatten
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

# Import from disk and enrich labels with computed attributes
dt-import-fo disk datasets/my_ds my_dataset --enrich

# Import from CVAT
dt-import-fo cvat my_dataset -P "My CVAT Project"

# Export to disk
dt-export-fo my_dataset out/exported -t yolo -s train val
```

### CVAT task management

```bash
# Split a task's single job into two jobs at frame 250
dt-cvat split --task 41 --cuts 250

# Split into 3 roughly equal jobs
dt-cvat split --task 41 --equal 3

# Multiple custom cut points (produces 3 jobs: 0-99, 100-299, 300-end)
dt-cvat split --task 41 --cuts 100 300

# Split a specific job in a multi-job task
dt-cvat split --task 41 --job 54 --cuts 250

# Preview the split without creating anything
dt-cvat split --task 41 --cuts 250 --dry-run
```

CVAT credentials are read from `DT_CVAT_*` env vars in `.env`
(see `.env.example`) or passed via `--url`, `--username`, `--password`,
`--organization`.

### Enrich a FiftyOne dataset

```bash
# Compute label areas (auto-detects label fields)
dt-enrich my_dataset

# Target specific label fields
dt-enrich my_dataset -l detections -l segmentations

# Adjust batch size for large mask-heavy datasets
dt-enrich my_dataset -b 500
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
| `dt-enrich` | Compute derived label attributes (area) | `.[fiftyone]` |
| `dt-cvat` | CVAT task management (split jobs) | `cvat-sdk` |
