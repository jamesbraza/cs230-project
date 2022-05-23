import collections
import os
from typing import DefaultDict, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from data.dataset_utils import get_dataset
from models.vgg16 import VGG_IMAGE_SIZE
from training import MODELS_DIR_ABS_PATH
from training.utils import (
    ImagesLabelsPreds,
    get_most_recent_model,
    plot_batch_predictions,
)

MODEL_NAME = get_most_recent_model()
PLOT_IMAGES = False

Statistics = Dict[Tuple[int, str], Tuple[int, int]]


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
train_ds, val_ds, test_ds = get_dataset(
    "small", image_size=VGG_IMAGE_SIZE, batch_size=16
)

# 2. Rehydrate the model
model_location = os.path.join(MODELS_DIR_ABS_PATH, MODEL_NAME)
model = tf.keras.models.load_model(model_location)

# 3. Make predictions on the dataset(s)
train_ds_stats = get_dataset_predict_stats(model, train_ds)
train_ds_summary = get_dataset_accuracy(train_ds_stats)
val_ds_stats = get_dataset_predict_stats(model, val_ds)
val_ds_summary = get_dataset_accuracy(val_ds_stats)
test_ds_stats = get_dataset_predict_stats(model, test_ds)
test_ds_summary = get_dataset_accuracy(test_ds_stats)
for ds_name, ds_accuracy, ds_total, ds_per_label in [
    ("Training", *train_ds_summary),
    ("Validation", *val_ds_summary),
    ("Test", *test_ds_summary),
]:
    readable_ds_per_label = {
        class_name: round(acc * 100, 2)
        for (label, class_name), acc in ds_per_label.items()
    }
    print(
        f"{ds_name} set accuracy: {ds_accuracy * 100:.2f}% correct "
        f"of {ds_total} images: {readable_ds_per_label}."
    )
_ = 0  # Debug here
