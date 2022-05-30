"""Clean out all invalid images from a dataset."""

import os
from typing import List

import PIL
import tensorflow as tf
from PIL import Image

JPEG = "JPEG"  # For both .jpg and .jpeg


def is_valid_jfif(path: str) -> bool:
    """
    Get if an input path is to a valid JPEG File Interchange Format (JFIF) file.

    SEE: https://keras.io/examples/vision/image_classification_from_scratch/
    """
    with open(path, "rb") as fobj:
        return tf.compat.as_bytes("JFIF") in fobj.peek(10)


def is_valid_image(path: str, img_format: str = JPEG) -> bool:
    """
    Verify an image at the input path using PIL.Image.verify and Image.format.

    SEE: https://stackoverflow.com/a/48178294/11163122
    """
    try:
        with Image.open(path) as im:
            im.verify()
            return im.format == img_format
    except PIL.UnidentifiedImageError:  # Not an image (e.g. .DS_Store)
        return False


def clean_images(
    base_folder: str,
    labels: List[str],
    dry_run: bool = True,
    pil_check: bool = True,
    tensorflow_check: bool = False,
) -> None:
    """
    Clean out all corrupted JPEG images from a directory.

    Args:
        base_folder: Base directory housing subdirectories of images.
        labels: Subdirectory names corresponding with labels
        dry_run: If you don't actually want to delete the images.
        pil_check: If you want to clean per PIL.Image's suggestion
        tensorflow_check: If you want to clean per tensorflow's suggestion.
            This is worse, hence why it's default is False.
    """
    for label in labels:
        folder_path = os.path.join(base_folder, label)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                continue
            if pil_check and not is_valid_image(fpath):
                print(f"Corrupted image per PIL: {fpath}")
                if not dry_run:
                    os.remove(fpath)
            try:
                if tensorflow_check and not is_valid_jfif(fpath):
                    print(f"Corrupted image per tensorflow: {fpath}")
                    if not dry_run:
                        os.remove(fpath)
            except FileNotFoundError:
                print(
                    f"Can't check image {fpath} per tensorflow since it was "
                    f"removed after PIL check."
                )


def bulk_rename(base_folder: str, labels: List[str]) -> None:
    """Rename an entire dataset so image names are incrementing integers."""
    for label in labels:
        folder_path = os.path.join(base_folder, label)
        if not os.path.isdir(folder_path):
            continue
        i = 1
        for fname in sorted(os.listdir(folder_path)):
            fpath = os.path.join(folder_path, fname)
            if is_valid_image(fpath):
                while os.path.exists(os.path.join(folder_path, f"{i}.jpg")):
                    i += 1  # Go to next available index
                os.rename(fpath, os.path.join(folder_path, f"{i}.jpg"))
                print(f"Renamed image {fname} to {i}.jpg in directory {folder_path}.")
                i += 1


if __name__ == "__main__":
    from data.dataset_utils import (
        HOME_ABS_PATH,
        HOME_DATASET_LABELS,
        SHIRTS_ABS_PATH,
        SHIRTS_DATASET_LABELS,
    )

    bulk_rename(HOME_ABS_PATH, HOME_DATASET_LABELS)
    # clean_images(SHIRTS_ABS_PATH, SHIRTS_DATASET_LABELS)
