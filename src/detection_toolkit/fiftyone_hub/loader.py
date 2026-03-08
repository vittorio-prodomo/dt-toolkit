"""Load on-disk datasets into FiftyOne.

Uses :class:`~detection_toolkit.core.loader.DatasetLoader` for format and
split detection, then feeds each split through FiftyOne's native importers.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import fiftyone as fo
from fiftyone.utils.coco import COCODetectionDatasetImporter
from fiftyone.utils.yolo import YOLOv5DatasetImporter

from ..core.loader import DatasetLoader
from ..core.models import DatasetFormat, DatasetInfo, DatasetSplit, TaskType
from .utils import create_thumbnails

logger = logging.getLogger(__name__)


class FiftyOneDatasetLoader:
    """Import a YOLO, COCO, or CVAT dataset into FiftyOne.

    Automatically detects the on-disk format and directory variant via
    :class:`DatasetLoader`, then delegates to FiftyOne's importers so
    that every sample gets a ``split`` field.

    Usage::

        loader = FiftyOneDatasetLoader()
        dataset = loader.load(
            Path("datasets/my_dataset"),
            "my_dataset",
        )
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def load(
        self,
        dataset_path: Union[str, Path],
        dataset_name: str,
        format: Optional[DatasetFormat] = None,
        persistent: bool = True,
        overwrite: bool = False,
        shuffle: bool = False,
        include_all_data: bool = False,
        use_polylines: bool = False,
        create_thumbs: bool = False,
        thumbnail_size: int = 256,
        data_path: Optional[str] = None,
    ) -> fo.Dataset:
        """Load a dataset into FiftyOne.

        Args:
            dataset_path: Root directory of the dataset.
            dataset_name: Name for the FiftyOne dataset.
            format: Explicit format; auto-detected if *None*.
            persistent: Persist the dataset in FiftyOne's database.
            overwrite: Delete an existing dataset with the same name.
            shuffle: Shuffle sample order after loading.
            include_all_data: Include unlabelled images (YOLO only).
            use_polylines: Import COCO segmentation as polylines
                instead of dense masks.
            create_thumbs: Generate thumbnails for the app grid.
            thumbnail_size: Thumbnail height in pixels.
            data_path: External image directory. When provided, images
                are loaded from this path instead of the default
                ``data/`` sub-directory (CVAT only).

        Returns:
            The loaded :class:`fo.Dataset`.
        """
        dataset_path = Path(dataset_path).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        # Handle existing dataset
        if fo.dataset_exists(dataset_name):
            if overwrite:
                logger.info("Overwriting existing dataset '%s'", dataset_name)
                fo.delete_dataset(dataset_name)
            else:
                raise ValueError(
                    f"Dataset '{dataset_name}' already exists. "
                    "Pass overwrite=True to replace it."
                )

        # Detect format
        core_loader = DatasetLoader(dataset_path, verbose=self.verbose)
        if format is None:
            try:
                format = core_loader.detect_format()
            except ValueError:
                format = None

        # CVAT path — delegate entirely to FiftyOne's importer
        if format == DatasetFormat.CVAT:
            fo_dataset = self._load_cvat_image(
                dataset_path, dataset_name, persistent, shuffle,
                include_all_data, data_path,
            )
            fo_dataset.compute_metadata()
            if create_thumbs:
                create_thumbnails(fo_dataset, dataset_path, thumbnail_size)
            logger.info(
                "Dataset '%s' ready — %d samples",
                dataset_name, len(fo_dataset),
            )
            return fo_dataset

        if format is None:
            # Fall back to plain image directory (unlabeled)
            logger.info(
                "No YOLO/COCO/CVAT format detected; importing as unlabeled "
                "image directory: %s", dataset_path,
            )
            fo_dataset = fo.Dataset.from_images_dir(
                str(dataset_path),
                name=dataset_name,
                persistent=persistent,
            )
            fo_dataset.compute_metadata()
            if create_thumbs:
                create_thumbnails(fo_dataset, dataset_path, thumbnail_size)
            logger.info(
                "Dataset '%s' ready — %d samples",
                dataset_name, len(fo_dataset),
            )
            return fo_dataset

        # YOLO/COCO path — load info first
        info = core_loader.load_dataset(format=format)

        logger.info(
            "Detected %s format with %d splits: %s",
            info.format.value.upper(),
            len(info.splits),
            list(info.splits.keys()),
        )

        if info.format == DatasetFormat.YOLO:
            fo_dataset = self._load_yolo(
                dataset_path, dataset_name, info,
                persistent, shuffle, include_all_data,
            )
        else:
            fo_dataset = self._load_coco(
                dataset_path, dataset_name, info,
                persistent, shuffle, use_polylines,
            )

        fo_dataset.compute_metadata()

        if create_thumbs:
            create_thumbnails(fo_dataset, dataset_path, thumbnail_size)

        logger.info(
            "Dataset '%s' ready — %d samples", dataset_name, len(fo_dataset)
        )
        return fo_dataset

    # ------------------------------------------------------------------
    # CVAT Image (disk)
    # ------------------------------------------------------------------

    def _load_cvat_image(
        self,
        dataset_path: Path,
        dataset_name: str,
        persistent: bool,
        shuffle: bool,
        include_all_data: bool,
        data_path: Optional[str],
    ) -> fo.Dataset:
        from fiftyone.utils.cvat import CVATImageDatasetImporter

        # Find the labels XML
        labels_path = None
        for name in ("labels.xml", "annotations.xml"):
            candidate = dataset_path / name
            if candidate.is_file():
                labels_path = candidate
                break

        importer_kwargs = dict(
            dataset_dir=str(dataset_path),
            labels_path=str(labels_path) if labels_path else None,
            include_all_data=include_all_data,
            shuffle=shuffle,
        )
        if data_path is not None:
            importer_kwargs["data_path"] = data_path

        importer = CVATImageDatasetImporter(**importer_kwargs)
        fo_dataset = fo.Dataset.from_importer(
            importer,
            name=dataset_name,
            persistent=persistent,
            overwrite=True,
        )

        # Set split field from XML subset attributes
        if labels_path is not None:
            self._set_cvat_splits(fo_dataset, labels_path)

        return fo_dataset

    @staticmethod
    def _set_cvat_splits(dataset: fo.Dataset, labels_path: Path) -> None:
        """Parse subset attributes from CVAT XML and set the split field."""
        try:
            tree = ET.parse(labels_path)
        except ET.ParseError:
            return

        root = tree.getroot()
        # Map image name -> subset
        subset_map = {}
        for image_el in root.iter("image"):
            subset = image_el.get("subset")
            if subset:
                name = image_el.get("name", "")
                subset_map[name] = subset

        if not subset_map:
            return

        dataset.add_sample_field("split", fo.StringField)

        # Build filepath -> subset mapping
        splits = []
        for sample in dataset.iter_samples():
            fname = Path(sample.filepath).name
            splits.append(subset_map.get(fname))

        dataset.set_values("split", splits)
        unique = set(s for s in splits if s is not None)
        logger.info("Set split field from CVAT subsets: %s", sorted(unique))

    # ------------------------------------------------------------------
    # CVAT (server)
    # ------------------------------------------------------------------

    def load_cvat(
        self,
        dataset_name: str,
        *,
        project_name=None,
        project_id=None,
        task_ids=None,
        job_ids=None,
        data_path=None,
        label_types=None,
        insert_new=True,
        download_media=False,
        num_workers=None,
        occluded_attr=None,
        group_id_attr=None,
        persistent=True,
        overwrite=False,
        sync=False,
        create_thumbs=False,
        thumbnail_size=256,
        cvat_url=None,
        cvat_username=None,
        cvat_password=None,
        cvat_organization=None,
    ) -> fo.Dataset:
        """Import annotations from a CVAT server into FiftyOne.

        Provide exactly one of *project_name*, *project_id*, *task_ids*,
        or *job_ids* to select what to import.

        Args:
            dataset_name: Name for the FiftyOne dataset.
            project_name: CVAT project name.
            project_id: CVAT project ID.
            task_ids: List of CVAT task IDs.
            job_ids: List of CVAT job IDs.
            data_path: Directory, JSON manifest, or existing dataset name
                to match filenames.
            label_types: Label types to import (e.g. ``["detections"]``).
            insert_new: Create samples for media not already in the dataset.
            download_media: Download media files from CVAT.
            num_workers: Threads for media download.
            occluded_attr: Attribute name to store occlusion info.
            group_id_attr: Attribute name to store group IDs.
            persistent: Persist the dataset in FiftyOne's database.
            overwrite: Delete an existing dataset with the same name.
            sync: Keep existing samples but clear all label fields
                before re-importing. Mutually exclusive with *overwrite*.
            create_thumbs: Generate thumbnails for the app grid.
            thumbnail_size: Thumbnail height in pixels.
            cvat_url: CVAT server URL.
            cvat_username: CVAT username.
            cvat_password: CVAT password.
            cvat_organization: CVAT organization (workspace) name.

        Returns:
            The loaded :class:`fo.Dataset`.
        """
        import fiftyone.utils.cvat as fouc

        # Handle existing dataset
        if fo.dataset_exists(dataset_name):
            if overwrite:
                logger.info("Overwriting existing dataset '%s'", dataset_name)
                fo.delete_dataset(dataset_name)
                dataset = fo.Dataset(dataset_name, persistent=persistent)
                logger.info("Created dataset '%s'", dataset_name)
            elif sync:
                dataset = fo.load_dataset(dataset_name)
                label_fields = dataset._get_label_fields()
                if label_fields:
                    logger.info(
                        "Sync: clearing label fields %s on '%s'",
                        label_fields, dataset_name,
                    )
                    dataset.delete_sample_fields(label_fields)
                else:
                    logger.info(
                        "Sync: no label fields to clear on '%s'",
                        dataset_name,
                    )
            else:
                raise ValueError(
                    f"Dataset '{dataset_name}' already exists. "
                    "Pass overwrite=True or sync=True."
                )
        else:
            dataset = fo.Dataset(dataset_name, persistent=persistent)
            logger.info("Created dataset '%s'", dataset_name)

        # Build CVAT auth kwargs (only pass non-None values)
        cvat_kwargs = {}
        if cvat_url is not None:
            cvat_kwargs["url"] = cvat_url
        if cvat_username is not None:
            cvat_kwargs["username"] = cvat_username
        if cvat_password is not None:
            cvat_kwargs["password"] = cvat_password
        if cvat_organization is not None:
            cvat_kwargs["organization"] = cvat_organization

        fouc.import_annotations(
            dataset,
            project_name=project_name,
            project_id=project_id,
            task_ids=task_ids,
            job_ids=job_ids,
            data_path=data_path,
            label_types=label_types,
            insert_new=insert_new,
            download_media=download_media,
            num_workers=num_workers,
            occluded_attr=occluded_attr,
            group_id_attr=group_id_attr,
            **cvat_kwargs,
        )

        if create_thumbs:
            create_thumbnails(dataset, size=thumbnail_size)

        logger.info(
            "Dataset '%s' ready — %d samples", dataset_name, len(dataset)
        )
        return dataset

    # ------------------------------------------------------------------
    # YOLO
    # ------------------------------------------------------------------

    def _load_yolo(
        self,
        dataset_path: Path,
        dataset_name: str,
        info: DatasetInfo,
        persistent: bool,
        shuffle: bool,
        include_all_data: bool,
    ) -> fo.Dataset:
        label_type = (
            "polylines"
            if info.task_type == TaskType.SEGMENTATION
            else "detections"
        )
        label_field = (
            "segmentations"
            if info.task_type == TaskType.SEGMENTATION
            else "detections"
        )

        yaml_path = self._find_yaml(dataset_path)
        split_names = list(info.splits.keys())
        first_split = split_names[0]

        # Import first split to create the dataset
        fo_dataset = self._import_yolo_split(
            yaml_path, first_split, dataset_name,
            persistent, shuffle, include_all_data,
            label_type, label_field,
        )
        fo_dataset.add_sample_field("split", fo.StringField)
        self._set_split_field(fo_dataset, first_split)

        # Merge remaining splits
        for split_name in split_names[1:]:
            logger.info("Loading YOLO split: %s", split_name)
            temp = self._import_yolo_split(
                yaml_path, split_name, f"{dataset_name}_{split_name}",
                False, shuffle, include_all_data,
                label_type, label_field,
            )
            self._set_split_field(temp, split_name)
            fo_dataset.merge_samples(temp)
            temp.delete()

        fo_dataset.classes = {label_field: info.class_list}
        return fo_dataset

    def _import_yolo_split(
        self,
        yaml_path: Path,
        split: str,
        name: str,
        persistent: bool,
        shuffle: bool,
        include_all_data: bool,
        label_type: str,
        label_field: str,
    ) -> fo.Dataset:
        importer = YOLOv5DatasetImporter(
            yaml_path=str(yaml_path),
            split=split,
            shuffle=shuffle,
            label_type=label_type,
            include_all_data=include_all_data,
        )
        return fo.Dataset.from_importer(
            importer,
            name=name,
            persistent=persistent,
            label_field=label_field,
            overwrite=True,
        )

    @staticmethod
    def _find_yaml(dataset_path: Path) -> Path:
        candidates = list(dataset_path.glob("*.yaml")) + list(
            dataset_path.glob("*.yml")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No data.yaml found in {dataset_path}"
            )
        return candidates[0]

    # ------------------------------------------------------------------
    # COCO
    # ------------------------------------------------------------------

    def _load_coco(
        self,
        dataset_path: Path,
        dataset_name: str,
        info: DatasetInfo,
        persistent: bool,
        shuffle: bool,
        use_polylines: bool,
    ) -> fo.Dataset:
        split_names = list(info.splits.keys())
        first_split_name = split_names[0]
        first_split = info.splits[first_split_name]

        data_path = self._coco_data_path(
            first_split, first_split_name, dataset_path
        )
        fo_dataset = self._import_coco_split(
            dataset_path, data_path, first_split.coco_ann_path,
            dataset_name, persistent, shuffle, use_polylines,
        )
        fo_dataset.add_sample_field("split", fo.StringField)
        self._set_split_field(fo_dataset, first_split_name)

        for split_name in split_names[1:]:
            logger.info("Loading COCO split: %s", split_name)
            split = info.splits[split_name]
            dp = self._coco_data_path(split, split_name, dataset_path)
            temp = self._import_coco_split(
                dataset_path, dp, split.coco_ann_path,
                f"{dataset_name}_{split_name}", False, shuffle, use_polylines,
            )
            self._set_split_field(temp, split_name)
            fo_dataset.merge_samples(temp)
            temp.delete()

        # Set classes on the correct label field
        label_field = self._detect_coco_label_field(fo_dataset)
        fo_dataset.classes = {label_field: info.class_list}

        # Remove empty detections field if segmentations exist
        if (
            label_field == "segmentations"
            and fo_dataset.has_sample_field("detections")
        ):
            fo_dataset.delete_sample_field("detections", error_level=2)

        return fo_dataset

    def _import_coco_split(
        self,
        dataset_dir: Path,
        data_path: Path,
        labels_path: Path,
        name: str,
        persistent: bool,
        shuffle: bool,
        use_polylines: bool,
    ) -> fo.Dataset:
        importer = COCODetectionDatasetImporter(
            dataset_dir=str(dataset_dir),
            data_path=str(data_path),
            labels_path=str(labels_path),
            shuffle=shuffle,
            use_polylines=use_polylines,
        )
        return fo.Dataset.from_importer(
            importer,
            name=name,
            persistent=persistent,
            overwrite=True,
        )

    @staticmethod
    def _coco_data_path(
        split: DatasetSplit, split_name: str, dataset_path: Path
    ) -> Path:
        """Return the image directory for a COCO split.

        For Roboflow-inline datasets (COCO-B), images sit beside the JSON
        so we return a relative path from the dataset root.  For standard
        COCO layouts (A/C), images are in ``images/{split}/``.
        """
        if split.coco_images_root is not None:
            try:
                return split.coco_images_root.resolve().relative_to(
                    dataset_path.resolve()
                )
            except ValueError:
                return split.coco_images_root
        return Path("images") / split_name

    @staticmethod
    def _detect_coco_label_field(dataset: fo.Dataset) -> str:
        """Return the actual label field created by the COCO importer."""
        if dataset.has_sample_field("segmentations"):
            return "segmentations"
        return "detections"

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_split_field(dataset: fo.Dataset, split_name: str) -> None:
        """Set the ``split`` field for all samples in *dataset*."""
        dataset.set_values("split", [split_name] * len(dataset))
        logger.debug(
            "Set split='%s' on %d samples", split_name, len(dataset)
        )
