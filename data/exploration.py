import glob
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from data.dataset_utils import (
    FULL_ABS_PATH,
    FULL_DATASET_LABELS,
    SHIRTS_ABS_PATH,
    SHIRTS_DATASET_LABELS,
    SMALL_DATASET_LABELS,
    SMALL_TRAIN_ABS_PATH,
    get_dataset,
    get_full_dataset,
    get_label_overlap,
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
    train_ds, val_ds, test_ds, labels = get_dataset("small")
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
    train_ds, val_ds = get_full_dataset(
        validation_split=0.0,
        filter_labels=get_label_overlap(
            SMALL_DATASET_LABELS, other_ds_labels=FULL_DATASET_LABELS
        ),
    )
    # train_ds, val_ds, _ = get_dataset("full")
    fig, ax = plt.subplots(nrows=3, ncols=3)
    for batch_images, batch_labels in train_ds:
        for i, axes in enumerate(fig.axes):
            axes.imshow(batch_images[i].numpy().astype("uint8"))
            axes.set_title(
                f"{int(batch_labels[i])}: {train_ds.class_names[batch_labels[i]]}"
            )
        fig.tight_layout()  # Prevent title overlap
        plt.show()
        _ = 0  # Debug here


def explore_shirts_dataset() -> None:
    train_ds, val_ds, _, labels = get_dataset("shirts")


def explore_home_dataset() -> None:
    # validation_split of 0.1 yields one image in the validation set
    train_ds, val_ds, _, labels = get_dataset("home", validation_split=0.1)
    _ = 0  # Debug here


if __name__ == "__main__":
    explore_small_dataset_raw()
    explore_full_dataset_raw()
    explore_shirts_dataset_raw()
    explore_small_dataset()
    explore_full_dataset()
    explore_shirts_dataset()
    explore_home_dataset()
