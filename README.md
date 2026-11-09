# dt-toolkit

Command-line building blocks for **object-detection and instance-segmentation datasets**: inspect, convert, tile,
filter, merge and split them, move them in and out of [FiftyOne](https://github.com/voxel51/fiftyone) and
[CVAT](https://github.com/cvat-ai/cvat), and compute embeddings, projections and clusters for curation.

It grew out of day-to-day work on large drone and robot image collections, where the stock tools kept falling short:
tiling that loses holes and multi-part masks, conversions that split one object into several, CVAT jobs that cannot be
re-cut, datasets that are hard to rename once FiftyOne points at them.

**Highlights**

- **Topology-preserving tiling** (`dt-tile`). Polygons are clipped per tile with Shapely. Holes and parts that the tile
  boundary splits are kept on one instance (bridged), not dropped or filled. Decoding uses pyvips, and tiling runs in
  parallel.
- **YOLO ↔ COCO conversion that keeps holes and multi-part masks on one instance** (`dt-convert`), built on
  [`cocoyolo`](https://github.com/vittorio-prodomo/cocoyolo).
- **Class surgery and dataset merging** with remapping (`dt-filter`, `dt-merge`).
- **FiftyOne bridges** (`dt-import-fo`, `dt-export-fo`, `dt-enrich`, `dt-thumbnails`, `dt-rename`) and **CVAT job
  splitting** (`dt-cvat`).
- **Embedding workflows** (`dt-embed`, `dt-reduce`, `dt-cluster`, `dt-visualize`): FiftyOne zoo, Hugging Face or any
  OpenAI-compatible embedding API, with multi-GPU sharding; UMAP, PaCMAP or PCA projections; clustering.
- **Stratified, group-aware splits** (`dt-split`), so that frames of the same capture never straddle train and val.

## Install

Use [uv](https://docs.astral.sh/uv/). The FiftyOne extra relies on patched forks pinned in `pyproject.toml`; uv
applies those sources for you, plain `pip` does not.

```bash
git clone https://github.com/vittorio-prodomo/dt-toolkit && cd dt-toolkit
uv sync --extra fiftyone            # core + FiftyOne integration
uv sync --all-extras                # everything (adds torch/transformers, UMAP/PaCMAP, scikit-learn)
```

As a dependency of another uv project:

```toml
[project]
dependencies = ["detection-toolkit[fiftyone]"]

[tool.uv.sources]
detection-toolkit = { git = "https://github.com/vittorio-prodomo/dt-toolkit", tag = "v0.1.0" }

# needed with the fiftyone extra: the forks' version numbers don't satisfy upstream pins
[tool.uv]
override-dependencies = ["fiftyone-brain>=0", "voxel51-eta>=0"]
constraint-dependencies = ["graphql-core<3.3"]
```

Requirements: Python ≥ 3.10 and the libvips system library (`apt install libvips` / `brew install vips`).

## Quickstart

```bash
dt-ingest datasets/my_ds                           # detect the format, print splits, classes, counts
dt-convert datasets/coco_ds out/yolo_ds -t yolo    # COCO -> YOLO (holes and parts stay on one instance)
dt-tile datasets/my_ds out/tiled -t 640 -o 0.3     # 640 px tiles with 30% overlap
dt-import-fo disk out/tiled my_tiles -f yolo       # browse the result in FiftyOne
```

Every command has `--help`.

## Commands

| Command | What it does | Extra |
|---|---|---|
| `dt-ingest` | Detect a dataset's format and layout; print splits, classes and counts | — |
| `dt-convert` | Convert between YOLO and COCO (detection and segmentation) | — |
| `dt-tile` | Tile images and labels into fixed-size patches; also unlabeled image folders | — |
| `dt-filter` | Keep, remove or merge classes inside one dataset | — |
| `dt-merge` | Merge several datasets with class remapping | — |
| `dt-cvat` | CVAT task management: split a job at given frames or into N equal jobs | — |
| `dt-import-fo` | Import into FiftyOne from disk (YOLO / COCO / CVAT / plain images) or from CVAT | `fiftyone` |
| `dt-export-fo` | Export a FiftyOne dataset to YOLO or COCO | `fiftyone` |
| `dt-enrich` | Add label areas and EXIF GPS (with a map view) to a FiftyOne dataset | `fiftyone` |
| `dt-thumbnails` | Generate or audit thumbnails for a FiftyOne dataset | `fiftyone` |
| `dt-rename` | Rename a FiftyOne dataset and its directory on disk atomically | `fiftyone` |
| `dt-split` | Stratified train/val/test split, optionally keeping groups together | `fiftyone` |
| `dt-embed` | Compute embeddings (zoo / Hugging Face / OpenAI-compatible API), multi-GPU | `fiftyone`, `embed` |
| `dt-reduce` | Dimensionality reduction of stored embeddings (UMAP, PaCMAP, PCA) | `fiftyone`, `reduce` |
| `dt-cluster` | Cluster stored embeddings | `fiftyone`, `cluster` |
| `dt-visualize` | Build a FiftyOne embeddings visualization | `fiftyone` |

CVAT credentials come from `DT_CVAT_*` environment variables or a `.env` file in the current directory (see
`.env.example`), or from `--url`, `--username`, `--password` and `--organization`.

## Development

```bash
uv sync --all-extras --group dev
uv run pytest
```

Tests that need MongoDB skip themselves when no database answers at `FIFTYONE_DATABASE_URI` (default
`localhost:27017`); the multi-GPU embedding test is opt-in (`DT_EMBED_GPU_TESTS=1`).

## License

MIT, see [LICENSE](LICENSE). The FiftyOne forks it installs are Apache-2.0.
