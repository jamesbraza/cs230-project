import os.path
from typing import Any, Literal, TypeAlias

import tensorflow as tf

DATA_DIR_ABS_PATH = os.path.dirname(os.path.abspath(__file__))

SMALL_ABS_PATH = os.path.join(DATA_DIR_ABS_PATH, "clothing_dataset_small")
SMALL_TRAIN_ABS_PATH = os.path.join(SMALL_ABS_PATH, "train")
SMALL_DEV_ABS_PATH = os.path.join(SMALL_ABS_PATH, "validation")
SMALL_TEST_ABS_PATH = os.path.join(SMALL_ABS_PATH, "test")

FULL_ABS_PATH = os.path.join(DATA_DIR_ABS_PATH, "clothing_dataset_full")

SHIRTS_ABS_PATH = os.path.join(DATA_DIR_ABS_PATH, "shirts_dataset", "Dataset")

DatasetNames: TypeAlias = Literal["small", "full", "shirts"]


def get_dataset(
    name: DatasetNames, **kwargs
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset | None]:
    """
    Get one of the three datasets discussed in the README.

    Args:
        name: Name of the dataset to fetch.
        kwargs: Keyword arguments for keras image_dataset_from_directory.
            batch_size: Default of 32.
            image_size: Default of (256, 256). Images will be resized to this,
                and pixel values will be float. Cast to uint8 for visualization.

    Returns:
        Tuple of datasets: train, dev/validation, test.
    """
    if name == "small":
        paths = [SMALL_TRAIN_ABS_PATH, SMALL_DEV_ABS_PATH, SMALL_TEST_ABS_PATH]
        return tuple(
            tf.keras.preprocessing.image_dataset_from_directory(path, **kwargs)
            for path in paths
        )
    if name == "full":
        directory = FULL_ABS_PATH
    else:
        directory = SHIRTS_ABS_PATH
    kwargs: dict[str, Any] = {"validation_split": 0.2} | kwargs
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory, **({"subset": "training"} | kwargs)
    )
    dev_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory, **({"subset": "validation"} | kwargs)
    )
    return train_ds, dev_ds, None
