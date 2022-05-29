"""Clean out all invalid images from a dataset."""

import os
from typing import List

import tensorflow as tf
from PIL import Image


def is_valid_jfif(path: str) -> bool:
    """
    Get if an input path is to a valid JPEG File Interchange Format (JFIF) file.

    SEE: https://keras.io/examples/vision/image_classification_from_scratch/
    """
    with open(path, "rb") as fobj:
        return tf.compat.as_bytes("JFIF") in fobj.peek(10)


def clean_images(base_folder: str, labels: List[str], dry_run: bool = True) -> None:
    """
    Clean out all corrupted images from a directory.

    Args:
        base_folder: Base directory housing subdirectories of images.
        labels: Subdirectory names corresponding with labels
        dry_run: If you don't actually want to delete the images.
    """
    for label in labels:
        folder_path = os.path.join(base_folder, label)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                continue
            if not is_valid_jfif(fpath):
                print(f"Corrupted image: {fpath}")
                if not dry_run:
                    os.remove(fpath)
            with Image.open(fpath) as im:
                try:
                    im.verify()
                except Exception as exc:
                    _ = 0
                _ = 0
            _ = 0


if __name__ == "__main__":
    from data.dataset_utils import SHIRTS_ABS_PATH, SHIRTS_DATASET_LABELS

    clean_images(SHIRTS_ABS_PATH, SHIRTS_DATASET_LABELS)
