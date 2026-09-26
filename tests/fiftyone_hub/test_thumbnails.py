"""Thumbnails: collision-free mirrored layout, incremental refresh, audit.

Measured 2026-09-26 on the project database: the old flat layout
(``.{name}_thumbnails/<basename>``) made 1,380 samples of one dataset show
ANOTHER photo's thumbnail, because FiftyOne's ``transform_images`` names its
output by basename when no ``rel_dir`` is given.
"""
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from detection_toolkit.fiftyone_hub.thumbnails import (
    audit_thumbnails,
    needs_thumbnail,
    plan_thumbnail_paths,
    thumbnails_root,
)


def _img(path, color, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return str(path)


# ── pure planning ─────────────────────────────────────────────────────


def test_plan_mirrors_the_tree_so_equal_basenames_never_collide(tmp_path):
    root = tmp_path / "data"
    fps = [str(root / "VID_1" / "nx" / "nx_000000.png"),
           str(root / "VID_2" / "nx" / "nx_000000.png")]
    thumbs = tmp_path / ".ds_thumbnails"
    out = plan_thumbnail_paths(fps, thumbs, root)
    assert out == [str(thumbs / "VID_1" / "nx" / "nx_000000.png.webp"),
                   str(thumbs / "VID_2" / "nx" / "nx_000000.png.webp")]
    assert len(set(out)) == 2


def test_plan_appends_the_format_so_same_stem_files_stay_apart(tmp_path):
    root = tmp_path / "data"
    fps = [str(root / "a" / "f_001.png"), str(root / "a" / "f_001.jpg")]
    out = plan_thumbnail_paths(fps, tmp_path / ".t", root, fmt="webp")
    assert [Path(p).name for p in out] == ["f_001.png.webp", "f_001.jpg.webp"]
    assert plan_thumbnail_paths(fps, tmp_path / ".t", root, fmt="keep") == [
        str(tmp_path / ".t" / "a" / "f_001.png"), str(tmp_path / ".t" / "a" / "f_001.jpg")]
    with pytest.raises(ValueError, match="format"):
        plan_thumbnail_paths(fps, tmp_path / ".t", root, fmt="gif")


def test_plan_refuses_a_file_outside_the_root(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        plan_thumbnail_paths([str(tmp_path / "elsewhere" / "a.jpg")],
                             tmp_path / ".t", tmp_path / "data")


def test_default_root_is_a_hidden_sibling_of_the_common_ancestor(tmp_path):
    fps = [str(tmp_path / "data" / "a" / "1.jpg"), str(tmp_path / "data" / "b" / "2.jpg")]
    thumbs, rel = thumbnails_root(fps, "my-ds")
    assert rel == tmp_path / "data"
    assert thumbs == tmp_path / ".my-ds_thumbnails"
    assert thumbs.is_absolute()


def test_default_root_never_lands_inside_a_dvc_tracked_folder(tmp_path):
    # datasets/my_dataset_v1 is tracked by datasets/my_dataset_v1.dvc: thumbnails
    # written inside it would make `dvc status` report the dataset modified
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "my_dataset_v1.dvc").write_text("outs: []\n")
    fps = [str(tmp_path / "datasets" / "my_dataset_v1" / "images" / "a" / "1.jpg"),
           str(tmp_path / "datasets" / "my_dataset_v1" / "images" / "b" / "2.jpg")]
    thumbs, rel = thumbnails_root(fps, "my-dataset")
    assert thumbs == tmp_path / "datasets" / ".my-dataset_thumbnails"
    assert rel == tmp_path / "datasets" / "my_dataset_v1" / "images"


def test_explicit_output_dir_is_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fps = [str(tmp_path / "data" / "a" / "1.jpg")]
    thumbs, _ = thumbnails_root(fps, "ds", output_dir="relative/place")
    assert thumbs == tmp_path / "relative" / "place" / ".ds_thumbnails"


def test_default_root_refuses_a_shallow_common_ancestor():
    with pytest.raises(ValueError, match="output_dir"):
        thumbnails_root(["/a/1.jpg", "/b/2.jpg"], "ds")


def test_needs_thumbnail_on_missing_stale_or_forced(tmp_path):
    src = Path(_img(tmp_path / "src.jpg", "red"))
    thumb = tmp_path / "thumb.jpg"
    assert needs_thumbnail(str(src), str(thumb))                     # missing
    _img(thumb, "red")
    assert not needs_thumbnail(str(src), str(thumb))
    assert needs_thumbnail(str(src), str(thumb), force=True)
    later = time.time() + 5
    os.utime(src, (later, later))                                    # source newer
    assert needs_thumbnail(str(src), str(thumb))


def test_audit_flags_wrong_shared_missing_and_relative(tmp_path):
    a = _img(tmp_path / "x" / "p.jpg", "red")
    b = _img(tmp_path / "y" / "p.jpg", "blue")
    shared = _img(tmp_path / "t" / "p.jpg", "red")
    rep = audit_thumbnails(
        filepaths=[a, b, a, str(tmp_path / "z.jpg")],
        thumbnail_paths=[shared, shared, shared, "rel/z.jpg"],
    )
    assert rep["n"] == 4
    assert rep["shared_by_different_images"] == 2    # a and b share one file
    assert rep["relative"] == 1
    assert rep["missing_file"] == 1
    assert rep["unset"] == 0
    assert not rep["ok"]


def test_audit_accepts_a_duplicate_source_sharing_its_thumbnail(tmp_path):
    a = _img(tmp_path / "x" / "p.jpg", "red")
    t = _img(tmp_path / "t" / "p.jpg", "red")
    rep = audit_thumbnails([a, a], [t, t])
    assert rep["shared_by_different_images"] == 0 and rep["ok"]


# ── FiftyOne integration ──────────────────────────────────────────────

@pytest.fixture
def fo_dataset():
    import fiftyone as fo

    ds = fo.Dataset(f"dt-thumbs-test-{uuid.uuid4().hex[:8]}")
    yield ds
    ds.delete()


def _mean_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3).mean(0)


@pytest.mark.requires_mongod
def test_create_thumbnails_gives_every_sample_its_own_image(tmp_path, fo_dataset):
    import fiftyone as fo
    from detection_toolkit.fiftyone_hub.utils import create_thumbnails

    root = tmp_path / "data"
    red = _img(root / "VID_1" / "f_000.png", (255, 0, 0), size=(800, 600))
    blue = _img(root / "VID_2" / "f_000.png", (0, 0, 255), size=(800, 600))
    fo_dataset.add_samples([fo.Sample(filepath=red), fo.Sample(filepath=blue)])

    summary = create_thumbnails(fo_dataset, size=32)

    assert summary["generated"] == 2 and summary["failed"] == 0
    by_src = dict(zip(fo_dataset.values("filepath"), fo_dataset.values("thumbnail_path")))
    assert by_src[red] != by_src[blue]
    for src, thumb in by_src.items():
        assert os.path.isabs(thumb)
        assert Image.open(thumb).size[1] == 32
        assert np.allclose(_mean_rgb(thumb), _mean_rgb(src), atol=3)
    assert fo_dataset.app_config.grid_media_field == "thumbnail_path"
    assert audit_thumbnails(fo_dataset.values("filepath"),
                            fo_dataset.values("thumbnail_path"))["ok"]


@pytest.mark.requires_mongod
def test_rerun_only_fills_new_samples_and_keeps_other_media_fields(tmp_path, fo_dataset):
    import fiftyone as fo
    from detection_toolkit.fiftyone_hub.utils import create_thumbnails

    root = tmp_path / "data"
    fo_dataset.add_samples([fo.Sample(filepath=_img(root / "a" / "1.png", "red", (400, 300))),
                            fo.Sample(filepath=_img(root / "b" / "1.png", "blue", (400, 300)))])
    fo_dataset.app_config.media_fields = ["filepath", "depth_path"]
    fo_dataset.save()
    assert create_thumbnails(fo_dataset, size=32)["generated"] == 2

    # a new subfolder under the same root must not move the tree
    fo_dataset.add_sample(fo.Sample(filepath=_img(root / "c" / "1.png", "green", (400, 300))))
    summary = create_thumbnails(fo_dataset, size=32)
    assert summary["generated"] == 1 and summary["reused"] == 2
    assert set(fo_dataset.app_config.media_fields) == {"filepath", "depth_path", "thumbnail_path"}


@pytest.mark.requires_mongod
def test_a_size_change_regenerates_everything(tmp_path, fo_dataset):
    import fiftyone as fo
    from detection_toolkit.fiftyone_hub.utils import create_thumbnails

    fo_dataset.add_sample(fo.Sample(filepath=_img(tmp_path / "d" / "x" / "1.png", "red", (400, 300))))
    create_thumbnails(fo_dataset, size=32)
    summary = create_thumbnails(fo_dataset, size=64)
    assert summary["generated"] == 1
    assert Image.open(fo_dataset.first().thumbnail_path).size[1] == 64


@pytest.mark.requires_mongod
def test_default_output_is_webp_and_a_format_change_regenerates(tmp_path, fo_dataset):
    import fiftyone as fo
    from detection_toolkit.fiftyone_hub.utils import create_thumbnails

    src = _img(tmp_path / "d" / "x" / "1.png", (0, 200, 0), (400, 300))
    fo_dataset.add_sample(fo.Sample(filepath=src))
    create_thumbnails(fo_dataset, size=32)
    thumb = fo_dataset.first().thumbnail_path
    assert thumb.endswith("1.png.webp") and Image.open(thumb).format == "WEBP"

    summary = create_thumbnails(fo_dataset, size=32, fmt="jpg")
    assert summary["generated"] == 1
    assert Image.open(fo_dataset.first().thumbnail_path).format == "JPEG"


@pytest.mark.requires_mongod
def test_a_palette_or_alpha_image_still_becomes_a_webp(tmp_path, fo_dataset):
    import fiftyone as fo
    from detection_toolkit.fiftyone_hub.utils import create_thumbnails

    pal = tmp_path / "d" / "p" / "1.png"
    pal.parent.mkdir(parents=True)
    Image.new("RGB", (300, 200), "red").convert("P").save(pal)
    rgba = tmp_path / "d" / "q" / "1.png"
    rgba.parent.mkdir(parents=True)
    Image.new("RGBA", (300, 200), (0, 0, 255, 128)).save(rgba)
    fo_dataset.add_samples([fo.Sample(filepath=str(pal)), fo.Sample(filepath=str(rgba))])
    assert create_thumbnails(fo_dataset, size=32)["failed"] == 0


@pytest.mark.requires_mongod
def test_an_unreadable_image_falls_back_to_the_original(tmp_path, fo_dataset):
    import fiftyone as fo
    from detection_toolkit.fiftyone_hub.utils import create_thumbnails

    good = _img(tmp_path / "d" / "ok" / "1.png", "red", (400, 300))
    bad = tmp_path / "d" / "bad" / "1.png"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not an image")
    fo_dataset.add_samples([fo.Sample(filepath=good), fo.Sample(filepath=str(bad))])

    summary = create_thumbnails(fo_dataset, size=32)
    assert summary["failed"] == 1 and summary["generated"] == 1
    by_src = dict(zip(fo_dataset.values("filepath"), fo_dataset.values("thumbnail_path")))
    assert by_src[str(bad)] == str(bad)
