import os.path
from typing import Literal, Optional, Tuple

import tensorflow as tf

DATA_DIR_ABS_PATH = os.path.dirname(os.path.abspath(__file__))

SMALL_ABS_PATH = os.path.join(DATA_DIR_ABS_PATH, "clothing_dataset_small")
SMALL_TRAIN_ABS_PATH = os.path.join(SMALL_ABS_PATH, "train")
SMALL_DEV_ABS_PATH = os.path.join(SMALL_ABS_PATH, "validation")
SMALL_TEST_ABS_PATH = os.path.join(SMALL_ABS_PATH, "test")

FULL_ABS_PATH = os.path.join(DATA_DIR_ABS_PATH, "clothing_dataset_full")

SHIRTS_ABS_PATH = os.path.join(DATA_DIR_ABS_PATH, "shirts_dataset", "Dataset")

DatasetNames = Literal["small", "full", "shirts"]


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
    if name == "full":
        directory = FULL_ABS_PATH
        raise NotImplementedError("TODO: full dataset import.")
    else:
        directory = SHIRTS_ABS_PATH
    kwargs = {**{"seed": 42, "validation_split": 0.1}, **kwargs}
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
