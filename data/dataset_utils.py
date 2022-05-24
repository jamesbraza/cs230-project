import os.path
from typing import Dict, Literal, Optional, Tuple

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
    batch_size: int = 32, image_size: Tuple[int, int] = (256, 256)
) -> tf.data.Dataset:
    """
    Convert the full clothing dataset into a tensorflow Dataset.

    Args:
        batch_size: Desired batch size.
        image_size: Desired image size.

    Returns:
        BatchDataset corresponding with the input batch size.
    """
    data = pd.read_csv(os.path.join(FULL_ABS_PATH, "images.csv"))
    orig_images = os.path.join(FULL_ABS_PATH, "images_original")
    data["image"] = data["image"].map(lambda x: os.path.join(orig_images, f"{x}.jpg"))
    filenames: tf.Tensor = tf.constant(data["image"], dtype=tf.string)
    data["label"] = data["label"].str.lower()
    class_name_to_label: Dict[str, int] = {
        label: i for i, label in enumerate(set(data["label"]))
    }
    labels: tf.Tensor = tf.constant(
        data["label"].map(class_name_to_label.__getitem__), dtype=tf.uint8
    )
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    def _parse_function(filename, label):
        jpg_image: tf.Tensor = tf.io.decode_jpeg(tf.io.read_file(filename))
        return tf.image.resize(jpg_image, size=image_size), label

    dataset = dataset.map(_parse_function)
    batched_dataset = dataset.batch(batch_size)
    batched_dataset.class_names = list(class_name_to_label.keys())
    return batched_dataset


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
