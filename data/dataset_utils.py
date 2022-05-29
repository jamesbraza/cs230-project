import os
from collections.abc import Mapping as MappingCollection
from typing import List, Literal, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import tensorflow as tf

from data.clean import is_valid_image, is_valid_jfif

DATA_DIR_ABS_PATH = os.path.dirname(os.path.abspath(__file__))

SMALL_ABS_PATH = os.path.join(DATA_DIR_ABS_PATH, "clothing_dataset_small")
SMALL_TRAIN_ABS_PATH = os.path.join(SMALL_ABS_PATH, "train")
SMALL_DEV_ABS_PATH = os.path.join(SMALL_ABS_PATH, "validation")
SMALL_TEST_ABS_PATH = os.path.join(SMALL_ABS_PATH, "test")

FULL_ABS_PATH = os.path.join(DATA_DIR_ABS_PATH, "clothing_dataset_full")

SHIRTS_ABS_PATH = os.path.join(DATA_DIR_ABS_PATH, "shirts_dataset", "Dataset")


def _get_subdir_names(path: str) -> List[str]:
    """Get a sorted list of names of all subdirectories at the input path."""
    return sorted(
        [file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]
    )


SMALL_DATASET_LABELS: List[str] = _get_subdir_names(SMALL_TRAIN_ABS_PATH)
FULL_DATASET_LABELS: List[str] = [
    "undershirt",
    "hat",
    "polo",
    "shirt",
    "dress",
    "top",
    "blouse",
    "body",
    "longsleeve",
    "hoodie",
    "shoes",
    "skip",
    "outwear",
    "skirt",
    "not sure",
    "t-shirt",
    "other",
    "shorts",
    "pants",
    "blazer",
]
SHIRTS_DATASET_LABELS: List[str] = _get_subdir_names(SHIRTS_ABS_PATH)

DatasetNames = Literal["small", "full", "shirts"]


def get_full_dataset(  # noqa: C901  # pylint: disable=too-many-locals
    batch_size: int = 32,
    image_size: Tuple[int, int] = (256, 256),
    shuffle: bool = True,
    seed: Optional[int] = None,
    validation_split: float = 0.1,
    drop_remainder: bool = False,
    filter_labels: Optional[Union[Sequence[str], Mapping[str, int]]] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Convert the full clothing dataset into a tensorflow Dataset.

    Args:
        batch_size: Desired batch size.
        image_size: Desired image size.
        shuffle: If you want to shuffle the dataset.
        seed: Optional seed for shuffling.
        validation_split: Split in [0, 1] for validation data.
        drop_remainder: If you want to drop the last batch's incomplete set.
        filter_labels: Optional allow list of labels or allow mapping.
            If Sequence: use as a lookup of allowable labels.
            If Mapping: use as a lookup of allowable labels and use that label.
                This enables one to have consistent labels within an augmented
                dataset.

    Returns:
        BatchDatasets of training and validation data.
    """
    data = pd.read_csv(os.path.join(FULL_ABS_PATH, "images.csv"))
    data["image"] = data["image"].map(
        lambda x: os.path.join(
            os.path.join(FULL_ABS_PATH, "images_original"), f"{x}.jpg"
        )
    )
    valid_data_pre: List[Tuple[str, str]] = []
    for image_path, label in data[["image", "label"]].values:
        if not is_valid_image(image_path):
            continue
        if filter_labels is not None:
            label = label.lower()
            if label not in filter_labels:
                continue
        valid_data_pre.append((image_path, label))

    valid_data = pd.DataFrame(valid_data_pre, columns=["image", "label"])
    if isinstance(filter_labels, MappingCollection):
        class_name_to_label: Mapping[str, int] = filter_labels
    else:
        class_name_to_label = {
            label: i for i, label in enumerate(set(valid_data["label"]))
        }
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(valid_data["image"], dtype=tf.string),
            # Use tf.int32 (even though we could use tf.uint8) to match
            # behavior of image_dataset_from_directory
            tf.constant(
                valid_data["label"].map(class_name_to_label.__getitem__), dtype=tf.int32
            ),
        )
    )

    def _parse_function(filename, label):
        jpg_image: tf.Tensor = tf.io.decode_jpeg(tf.io.read_file(filename))
        return tf.image.resize(jpg_image, size=image_size), label

    dataset = dataset.map(_parse_function)
    if shuffle:
        dataset = dataset.shuffle(100, seed=seed, reshuffle_each_iteration=False)

    # SEE: https://stackoverflow.com/a/58452268/11163122
    validation_split_decimal = int(validation_split * 10)
    if validation_split_decimal != validation_split * 10:
        raise ValueError(
            f"Validation split {validation_split} must be a multiple of 10%."
        )

    def is_val(x, _) -> bool:
        return x % 10 < validation_split_decimal

    is_train = lambda x, y: not is_val(x, y)  # noqa: E731
    recover = lambda _, y: y  # noqa: E731
    train_ds = (
        dataset.enumerate()
        .filter(is_train)
        .map(recover)
        .batch(batch_size, drop_remainder)
    )
    train_ds.class_names = list(class_name_to_label.keys())
    val_ds = (
        dataset.enumerate()
        .filter(is_val)
        .map(recover)
        .batch(batch_size, drop_remainder)
    )
    val_ds.class_names = list(class_name_to_label.keys())
    return train_ds, val_ds


def get_dataset(
    name: DatasetNames, **kwargs
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset], List[str]]:
    """
    Get one of the three datasets discussed in the README.

    Args:
        name: Name of the dataset to fetch.
        kwargs: Keyword arguments for keras image_dataset_from_directory.
            batch_size: Default of 32.
            image_size: Default of (256, 256). Images will be resized to this,
                and pixel values will be float. Cast to uint8 for visualization.

    Returns:
        Tuple of datasets: train, validation/dev, test, labels.
    """
    if name == "small":
        paths = [SMALL_TRAIN_ABS_PATH, SMALL_DEV_ABS_PATH, SMALL_TEST_ABS_PATH]
        return (  # type: ignore[return-value]
            *tuple(
                tf.keras.preprocessing.image_dataset_from_directory(path, **kwargs)
                for path in paths
            ),
            SMALL_DATASET_LABELS,
        )
    kwargs = {**{"seed": 42, "validation_split": 0.1}, **kwargs}
    if name == "full":
        return (*get_full_dataset(**kwargs), None, FULL_DATASET_LABELS)  # type: ignore[return-value]
    directory = SHIRTS_ABS_PATH
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory, **{**{"subset": "training"}, **kwargs}
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory, **{**{"subset": "validation"}, **kwargs}
    )
    return train_ds, val_ds, None, SHIRTS_DATASET_LABELS


def get_num_classes(dataset: tf.data.Dataset) -> int:
    """Get the number of classes within a dataset."""
    return len(dataset.class_names)


def pass_class_names(
    orig_dataset: tf.data.Dataset, new_dataset: tf.data.Dataset
) -> tf.data.Dataset:
    """Pass the class_names on from one dataset to another."""
    new_dataset.class_names = orig_dataset.class_names
    return new_dataset


def get_label_overlap(
    match_ds_labels: Sequence[str], other_ds_labels: Sequence[str]
) -> Mapping[str, int]:
    """
    Get labels overlapping with a match dataset with the label index to apply.

    Args:
        match_ds_labels: Labels of the dataset we want to match/concatenate to.
        other_ds_labels: Labels of the dataset we are using for augmentation.

    Returns:
        Labels to update the "other dataset" with for concatenation with the
        "match dataset".
    """
    return {
        x: match_ds_labels.index(x) for x in other_ds_labels if x in match_ds_labels
    }
