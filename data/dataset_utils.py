import os
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd
import tensorflow as tf

DATA_DIR_ABS_PATH = os.path.dirname(os.path.abspath(__file__))

SMALL_ABS_PATH = os.path.join(DATA_DIR_ABS_PATH, "clothing_dataset_small")
SMALL_TRAIN_ABS_PATH = os.path.join(SMALL_ABS_PATH, "train")
SMALL_DEV_ABS_PATH = os.path.join(SMALL_ABS_PATH, "validation")
SMALL_TEST_ABS_PATH = os.path.join(SMALL_ABS_PATH, "test")

FULL_ABS_PATH = os.path.join(DATA_DIR_ABS_PATH, "clothing_dataset_full")

SHIRTS_ABS_PATH = os.path.join(DATA_DIR_ABS_PATH, "shirts_dataset", "Dataset")

DatasetNames = Literal["small", "full", "shirts"]


def get_full_dataset(
    batch_size: int = 32,
    image_size: Tuple[int, int] = (256, 256),
    shuffle: bool = True,
    seed: Optional[int] = None,
    validation_split: float = 0.1,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Convert the full clothing dataset into a tensorflow Dataset.

    Args:
        batch_size: Desired batch size.
        image_size: Desired image size.
        shuffle: If you want to shuffle the dataset.
        seed: Optional seed for shuffling.
        validation_split: Split in [0, 1] for validation data.

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
        try:
            with open(image_path, "rb") as fobj:
                # SEE: https://keras.io/examples/vision/image_classification_from_scratch/
                if tf.compat.as_bytes("JFIF") in fobj.peek(10):
                    valid_data_pre.append((image_path, label.lower()))
        except FileNotFoundError:
            pass
    valid_data = pd.DataFrame(valid_data_pre, columns=["image", "label"])
    class_name_to_label: Dict[str, int] = {
        label: i for i, label in enumerate(set(valid_data["label"]))
    }
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(valid_data["image"], dtype=tf.string),
            tf.constant(
                valid_data["label"].map(class_name_to_label.__getitem__), dtype=tf.uint8
            ),
        )
    )

    def _parse_function(filename, label):
        jpg_image: tf.Tensor = tf.io.decode_jpeg(tf.io.read_file(filename))
        return tf.image.resize(jpg_image, size=image_size), label

    dataset = dataset.map(_parse_function)
    if shuffle:
        dataset = dataset.shuffle(100, seed=seed, reshuffle_each_iteration=False)

    validation_split_decimal = int(validation_split * 10)
    if validation_split_decimal != validation_split * 10:
        raise ValueError(
            f"Validation split {validation_split} must be a multiple of 10%."
        )

    def is_val(x, _) -> bool:
        return x % 10 < validation_split_decimal

    is_train = lambda x, y: not is_val(x, y)  # noqa: E731
    recover = lambda _, y: y  # noqa: E731
    train_ds = dataset.enumerate().filter(is_train).map(recover).batch(batch_size)
    train_ds.class_names = list(class_name_to_label.keys())
    val_ds = dataset.enumerate().filter(is_val).map(recover).batch(batch_size)
    val_ds.class_names = list(class_name_to_label.keys())
    return train_ds, val_ds


def get_dataset(
    name: DatasetNames, **kwargs
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset]]:
    """
    Get one of the three datasets discussed in the README.

    Args:
        name: Name of the dataset to fetch.
        kwargs: Keyword arguments for keras image_dataset_from_directory.
            batch_size: Default of 32.
            image_size: Default of (256, 256). Images will be resized to this,
                and pixel values will be float. Cast to uint8 for visualization.

    Returns:
        Tuple of datasets: train, validation/dev, test.
    """
    if name == "small":
        paths = [SMALL_TRAIN_ABS_PATH, SMALL_DEV_ABS_PATH, SMALL_TEST_ABS_PATH]
        return tuple(  # type: ignore[return-value]
            tf.keras.preprocessing.image_dataset_from_directory(path, **kwargs)
            for path in paths
        )
    kwargs = {**{"seed": 42, "validation_split": 0.1}, **kwargs}
    if name == "full":
        return (*get_full_dataset(**kwargs), None)
    directory = SHIRTS_ABS_PATH
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory, **{**{"subset": "training"}, **kwargs}
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory, **{**{"subset": "validation"}, **kwargs}
    )
    return train_ds, val_ds, None


def get_num_classes(dataset: tf.data.Dataset) -> int:
    """Get the number of classes within a dataset."""
    return len(dataset.class_names)
