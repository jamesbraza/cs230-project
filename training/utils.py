import math
import os
from datetime import datetime
from typing import Callable, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data.dataset_utils import get_num_classes, pass_class_names
from training import MODELS_DIR_ABS_PATH


def get_ts_now_as_str() -> str:
    """Get an ISO 8601-compliant timestamp for use in naming."""
    return datetime.now().isoformat().replace(":", "_")


DEFAULT_DELIM: str = "--"
DEFAULT_SAVE_NICKNAME: str = "UNNAMED"


def get_most_recent_model(
    delim: str = DEFAULT_DELIM, models_dir: str = MODELS_DIR_ABS_PATH
) -> str:
    """Get the most recent model in the input directory."""
    most_recent_file: Tuple[str, Optional[datetime]] = "", None
    for file in os.listdir(models_dir):
        abs_file = os.path.join(models_dir, file)
        if os.path.isdir(abs_file):
            try:
                ts = datetime.fromisoformat(file.replace("_", ":").split(delim)[0])
            except ValueError:
                continue
            if most_recent_file[1] is None or ts > most_recent_file[1]:
                most_recent_file = abs_file, ts
    if most_recent_file == "":
        raise LookupError(f"Didn't find a recent model in {models_dir}.")
    return most_recent_file[0]


def get_model_by_nickname(
    nickname: str = DEFAULT_SAVE_NICKNAME,
    delim: str = DEFAULT_DELIM,
    models_dir: str = MODELS_DIR_ABS_PATH,
) -> str:
    """Get a model by nickname from the input directory."""
    for file in os.listdir(models_dir):
        abs_file = os.path.join(models_dir, file)
        if os.path.isdir(abs_file):
            try:
                if file.split(delim)[1] == nickname:
                    return abs_file
            except IndexError:
                continue
    raise LookupError(f"Didn't find a model nicknamed {nickname} in {models_dir}.")


ImagesLabelsPreds = List[Tuple[tf.Tensor, int, int]]


def plot_batch_predictions(
    images_labels_preds: ImagesLabelsPreds, class_names: List[str]
) -> None:
    batch_size = len(images_labels_preds)
    subplot_size = int(math.sqrt(batch_size))
    if subplot_size**2 != batch_size:
        raise NotImplementedError(
            f"Only works for perfect-square batch sizes, "
            f"not a batch size of {batch_size}."
        )
    fig, ax = plt.subplots(nrows=subplot_size, ncols=subplot_size)
    for i, (image, label, pred) in enumerate(images_labels_preds):
        fig.axes[i].imshow(image.numpy().astype("uint8"))
        fig.axes[i].set_title(
            f"Label: {label}: {class_names[label]}\n"
            f"Pred: {pred}: {class_names[pred]}"
        )
    fig.tight_layout()  # Prevent title overlap
    plt.show()


def make_preprocessing_generator(
    dataset: tf.data.Dataset,
    num_repeat: int = -1,
    image_preprocessor: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> Iterable[Tuple[tf.Tensor, np.ndarray]]:
    """
    Make an iterator that pre-processes the input dataset for training.

    Args:
        dataset: TensorFlow dataset to preprocess.
        num_repeat: Optional count of dataset repetitions.
            Default of -1 will repeat indefinitely.
            For training data: set to the number of epochs, or -1.
        image_preprocessor: Function to pre-process images per a model's requirements.
            Default of None will not try to pre-process images.
            Examples:
            - VGG16: tf.keras.applications.vgg16.preprocess_input.
            - ResNet50: tf.keras.applications.resnet50.preprocess_input.

    Returns:
        Iterable of (image, categorical label) tuples.
    """
    num_classes = get_num_classes(dataset)
    for batch_images, batch_labels in dataset.repeat(num_repeat):
        if image_preprocessor is not None:
            batch_images = image_preprocessor(batch_images)
        yield batch_images, tf.keras.utils.to_categorical(
            batch_labels, num_classes, dtype="bool"
        )


def preprocess_dataset(
    dataset: tf.data.Dataset,
    image_preprocessor: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> tf.data.Dataset:
    """
    Preprocess the input dataset for training.

    Args:
        dataset: TensorFlow dataset to preprocess.
        image_preprocessor: Function to pre-process images per a model's requirements.
            Default of None will not try to pre-process images.
            Examples:
            - VGG16: tf.keras.applications.vgg16.preprocess_input.
            - ResNet50: tf.keras.applications.resnet50.preprocess_input.

    Returns:
        Preprocessed dataset.
    """
    num_classes = get_num_classes(dataset)

    def _preprocess(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if image_preprocessor is not None:
            x = image_preprocessor(x)
        # NOTE: one_hot is a transformation step for Tensors, so we use it here
        # over to_categorical
        # pylint: disable=no-value-for-parameter
        return x, tf.one_hot(y, depth=num_classes)

    return pass_class_names(dataset, dataset.map(_preprocess))
