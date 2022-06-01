import functools
import math
import os
from datetime import datetime
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from data.dataset_utils import get_num_classes, pass_class_names
from training import MODELS_DIR_ABS_PATH


def get_ts_now_as_str() -> str:
    """Get an ISO 8601-compliant timestamp for use in naming."""
    return datetime.now().isoformat().replace(":", "_")


DEFAULT_DELIM: str = "--"
DEFAULT_SAVE_NICKNAME: str = "UNNAMED"


def get_path_to_most_recent_model(
    delim: str = DEFAULT_DELIM, models_dir: str = MODELS_DIR_ABS_PATH
) -> str:
    """Get the most recent model's absolute path in the input directory."""
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


def get_path_to_model_by_nickname(
    nickname: str = DEFAULT_SAVE_NICKNAME,
    delim: str = DEFAULT_DELIM,
    models_dir: str = MODELS_DIR_ABS_PATH,
) -> str:
    """Get a model absolute path by nickname from the input directory."""
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


@functools.lru_cache(5)
def _get_labels_preds(
    trained_model: tf.keras.Model, dataset: tf.data.Dataset
) -> List[Tuple[int, int]]:
    """Get a pairing of label-to-pred for an entire dataset."""
    labels_preds: List[Tuple[int, int]] = []
    for images, labels in dataset.unbatch():
        preds: np.ndarray = trained_model.predict(tf.expand_dims(images, axis=0))
        pred: int = np.argmax(np.squeeze(preds))
        labels_preds.append((int(labels), pred))
    return labels_preds


def get_confusion_matrix(
    trained_model: tf.keras.Model, dataset: tf.data.Dataset
) -> np.ndarray:
    """Get a confusion matrix from a trained model and input dataset."""
    return confusion_matrix(*zip(*_get_labels_preds(trained_model, dataset)))


def plot_confusion_matrix(
    cm: np.ndarray,
    display_labels: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot a confusion matrix optionally with labels to display."""
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(xticks_rotation="vertical")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def get_classification_report(
    trained_model: tf.keras.Model,
    dataset: tf.data.Dataset,
    display_labels: Optional[Sequence[str]] = None,
    digits: int = 2,
) -> str:
    """Get a classification report (precision, recall, F1 score) using sklearn."""
    return classification_report(
        *zip(*_get_labels_preds(trained_model, dataset)),
        target_names=display_labels,
        digits=digits,
    )


def plot_softmax_visualization(
    trained_model: tf.keras.Model,
    dataset: tf.data.Dataset,
    num_rows: int,
    num_cols: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a visualization of the input model + dataset's performance.

    SEE: https://www.tensorflow.org/tutorials/keras/classification#verify_predictions
    """
    num_images = num_rows * num_cols
    fig = plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows), tight_layout=True)
    if title is not None:
        fig.suptitle(title)
    for i, (image, label) in enumerate(dataset.unbatch()):
        preds: np.ndarray = np.squeeze(
            trained_model.predict(tf.expand_dims(image, axis=0))
        )
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(
            image,
            max_pred=np.max(preds),
            pred_label=dataset.class_names[np.argmax(preds)],
            true_label=dataset.class_names[label],
        )
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_softmax_bar_graph(preds, true_label_index=label)
        if i + 1 == num_images:
            break
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_image(
    image: tf.Tensor, max_pred: float, pred_label: str, true_label: str
) -> None:
    """Plot the input image with a label using the other given information."""
    plt.grid(False)
    plt.xticks([])
    plt.xlabel(
        f"{pred_label} {max_pred * 100:2.1f}% ({true_label})",
        color="blue" if pred_label == true_label else "red",
    )
    plt.yticks([])
    plt.imshow(image.numpy().astype("uint8"))


def plot_softmax_bar_graph(preds: np.ndarray, true_label_index: int) -> None:
    """Plot the output of the softmax activation as a bar graph."""
    plt.grid(False)
    plt.xticks(range(len(preds)))
    plt.xlabel("Labels")
    plt.yticks(range(0, 100 + 1, 10))
    plt.ylim([0, 100])
    thisplot = plt.bar(range(len(preds)), preds * 100, color="grey")
    thisplot[np.argmax(preds)].set_color("red")
    thisplot[true_label_index].set_color("blue")


def plot_image_datagen(
    datagen: tf.keras.preprocessing.image.DirectoryIterator,
    num_rows: int,
    num_cols: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot a visualization of the Keras image data generator."""
    num_images = num_rows * num_cols
    fig = plt.figure(figsize=(num_cols, num_rows), tight_layout=True)
    if title is not None:
        fig.suptitle(title)
    for batch_images, _ in datagen:
        for i in range(num_images):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(batch_images[i].astype("uint8"))
        break
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
