# detection-toolkit

Modular building blocks for object-detection & instance-segmentation workflows.

## Installation

```bash
pip install -e .                # core only
pip install -e ".[fiftyone]"    # + FiftyOne integration
pip install -e ".[train]"       # + Ultralytics training
pip install -e ".[all]"         # everything
```

## CLI tools

After installation the following commands are available:

| Command | Description |
|---|---|
| `dt-ingest` | Ingest a dataset and display its metadata |
| `dt-convert` | Convert between YOLO and COCO formats |
| `dt-tile` | Tile a dataset into fixed-size patches |
| `dt-filter` | Filter or merge classes within a dataset |
| `dt-merge` | Merge multiple datasets with class remapping |
| `dt-train` | Train a model (auto-detected backend) |
| `dt-import-fo` | Import a dataset into FiftyOne |
| `dt-export-fo` | Export a FiftyOne dataset to disk |

Each command supports `--help` for full usage details.
