import glob
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from data.dataset_utils import (
    FULL_ABS_PATH,
    SHIRTS_ABS_PATH,
    SMALL_TRAIN_ABS_PATH,
    get_dataset,
)

FULL_CSV_ABS_PATH = os.path.join(FULL_ABS_PATH, "images.csv")
SHIRTS_CSV_ABS_PATH = os.path.join(SHIRTS_ABS_PATH, "data.csv")


def explore_small_dataset_raw() -> None:
    for filename in glob.iglob(f"{SMALL_TRAIN_ABS_PATH}/*/*.jpg", recursive=True):
        img: np.ndarray = mpimg.imread(filename)
        imgplot: mpimg.AxesImage = plt.imshow(img)
        print(img.shape)
        plt.show()


def explore_full_dataset_raw() -> None:
    data: pd.DataFrame = pd.read_csv(FULL_CSV_ABS_PATH)
    print(data.info())
    all_labels = set(data["label"])
    counts_labels = data["label"].value_counts(sort=True)

    filename: str
    sender_id: int
    label: str
    kids: bool
    for filename, sender_id, label, kids in data.values:  # noqa: B007
        label = label.lower()


def explore_shirts_dataset_raw() -> None:
    data: pd.DataFrame = pd.read_csv(SHIRTS_CSV_ABS_PATH)
    print(data.info())
    all_types = set(data["Type"])
    counts_types = data["Type"].value_counts(sort=True)
    all_designs = set(data["Design"])
    counts_designs = data["Design"].value_counts(sort=True)


def explore_small_dataset() -> None:
    train_ds, val_ds, test_ds = get_dataset("small")
    train_batches: tf.data.Dataset = train_ds.take(1)

    fig, ax = plt.subplots(nrows=3, ncols=3)
    for batch_images, batch_labels in train_batches:
        for i, axes in enumerate(fig.axes):
            axes.imshow(batch_images[i].numpy().astype("uint8"))
            axes.set_title(
                f"{int(batch_labels[i])}: {train_ds.class_names[batch_labels[i]]}"
            )
        fig.tight_layout()  # Prevent title overlap
        plt.show()


def explore_full_dataset() -> None:
    train_ds, val_ds, _ = get_dataset("full")
    fig, ax = plt.subplots(nrows=3, ncols=3)
    for batch_images, batch_labels in train_ds:
        for i, axes in enumerate(fig.axes):
            axes.imshow(batch_images[i].numpy().astype("uint8"))
            axes.set_title(
                f"{int(batch_labels[i])}: {train_ds.class_names[batch_labels[i]]}"
            )
        fig.tight_layout()  # Prevent title overlap
        plt.show()
        _ = 0


def explore_shirts_dataset() -> None:
    train_ds, val_ds, _ = get_dataset("shirts")


if __name__ == "__main__":
    explore_small_dataset_raw()
    explore_full_dataset_raw()
    explore_shirts_dataset_raw()
    explore_small_dataset()
    explore_full_dataset()
    explore_shirts_dataset()
