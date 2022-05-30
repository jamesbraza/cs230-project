import collections
import functools
import os
from typing import DefaultDict, Dict, List, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from data.dataset_utils import get_dataset
from models.resnet import RESNET_IMAGE_SIZE
from models.vgg16 import VGG_IMAGE_SIZE
from training import MODELS_DIR_ABS_PATH
from training.utils import (
    ImagesLabelsPreds,
    get_path_to_model_by_nickname,
    get_path_to_most_recent_model,
    plot_batch_predictions,
)

Statistics = Dict[Tuple[int, str], Tuple[int, int]]

# Name of the persisted Model metadata to load
MODEL_NAME = get_path_to_most_recent_model()
# If you want to plot images when making predictions
PLOT_IMAGES = False
# Which model to train
MODEL: Literal["vgg16_tl", "resnet_diy", "resnet_tl"] = "resnet_tl"

if MODEL.startswith("vgg16"):
    image_size: Tuple[int, int] = VGG_IMAGE_SIZE
elif MODEL.startswith("resnet"):
    image_size = RESNET_IMAGE_SIZE
else:
    raise NotImplementedError(f"Unrecognized model: {MODEL}.")


def get_dataset_predict_stats(
    trained_model: tf.keras.Model,
    dataset: tf.data.Dataset,
    plot_images: bool = PLOT_IMAGES,
) -> Statistics:
    """Get the number correct and image count using the trained model and dataset."""
    correct_totals: DefaultDict[int, List[int]] = collections.defaultdict(
        lambda: [0, 0]
    )
    for batch_images, batch_labels in dataset:
        preds: np.ndarray = trained_model.predict(batch_images)
        labels_preds: List[Tuple[int, int]] = [
            (int(label), np.argmax(pred)) for label, pred in zip(batch_labels, preds)
        ]
        for label, pred in labels_preds:
            if label == pred:
                correct_totals[label][0] += 1  # Num correct
            correct_totals[label][1] += 1  # Total
        if plot_images:
            images_labels_preds: ImagesLabelsPreds = [
                (image, *label_pred)
                for image, label_pred in zip(batch_images, labels_preds)
            ]
            plot_batch_predictions(images_labels_preds, dataset.class_names)
    return {
        (label, dataset.class_names[label]): (correct, total)
        for label, (correct, total) in sorted(correct_totals.items())
    }


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
    cm: np.ndarray, display_labels: Optional[Sequence[str]] = None
) -> None:
    """Plot a confusion matrix optionally with labels to display."""
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(xticks_rotation="vertical")


def get_classification_report(
    trained_model: tf.keras.Model,
    dataset: tf.data.Dataset,
    display_labels: Optional[Sequence[str]] = None,
) -> str:
    """Get a classification report (precision, recall, F1 score) using sklearn."""
    return classification_report(
        *zip(*_get_labels_preds(trained_model, dataset)), target_names=display_labels
    )


def get_dataset_accuracy(
    stats: Statistics,
) -> Tuple[float, int, Dict[Tuple[int, str], float]]:
    """Get the accuracy % and total image count from a statistics dict."""
    num_correct, num_total, per_label_acc = 0, 0, {}
    for label_tup, (correct, total) in stats.items():
        num_correct += correct
        num_total += total
        per_label_acc[label_tup] = num_correct / num_total
    return num_correct / num_total, num_total, per_label_acc


# 1. Get the dataset(s)
train_ds, val_ds, test_ds, labels = get_dataset(
    "small", image_size=image_size, batch_size=16
)

# 2. Rehydrate the model
model_location = os.path.join(MODELS_DIR_ABS_PATH, MODEL_NAME)
model = tf.keras.models.load_model(model_location)

# 3. Make predictions on the dataset(s)
train_ds_stats = get_dataset_predict_stats(model, train_ds)
train_ds_summary = get_dataset_accuracy(train_ds_stats)
val_ds_stats = get_dataset_predict_stats(model, val_ds)
val_ds_summary = get_dataset_accuracy(val_ds_stats)
results: List[tuple] = [
    ("Training", *train_ds_summary),
    ("Validation", *val_ds_summary),
]
if test_ds is not None:
    test_ds_stats = get_dataset_predict_stats(model, test_ds)
    test_ds_summary = get_dataset_accuracy(test_ds_stats)
    results.append(("Test", *test_ds_summary))
for ds_name, ds_accuracy, ds_total, ds_per_label in results:
    readable_ds_per_label = {
        class_name: round(acc * 100, 2)
        for (label, class_name), acc in ds_per_label.items()
    }
    print(
        f"{ds_name} set accuracy: {ds_accuracy * 100:.2f}% correct "
        f"of {ds_total} images: {readable_ds_per_label}."
    )
conf_matrix = get_confusion_matrix(model, test_ds if test_ds else val_ds)
if PLOT_IMAGES:
    plot_confusion_matrix(conf_matrix, labels)
    plt.show()
_ = 0  # Debug here
