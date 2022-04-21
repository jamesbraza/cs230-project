import os.path
from typing import Literal

import tensorflow as tf

DIR_ABS_PATH = os.path.dirname(os.path.abspath(__file__))
SMALL_REL_PATH = "clothing_dataset_small"
FULL_REL_PATH = "clothing_dataset_full"
SHIRTS_REL_PATH = os.path.join("shirts_dataset", "Dataset")


def get_dataset(name: Literal["small", "full", "shirts"], **kwargs) -> tf.data.Dataset:
    """
    Get one of the three datasets discussed in the README.

    Args:
        name: Name of the dataset to fetch.
        kwargs: Keyword arguments for keras image_dataset_from_directory.

    Returns:
        Dataset for use within tensorflow.
    """
    if name == "small":
        directory: str = os.path.join(DIR_ABS_PATH, SMALL_REL_PATH)
    elif name == "full":
        directory = os.path.join(DIR_ABS_PATH, FULL_REL_PATH)
    else:
        directory = os.path.join(DIR_ABS_PATH, SHIRTS_REL_PATH)
    return tf.keras.preprocessing.image_dataset_from_directory(directory, **kwargs)
