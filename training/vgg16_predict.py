import os

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from data.dataset_utils import get_dataset
from models.vgg16 import VGG_IMAGE_SIZE
from training import MODELS_DIR_ABS_PATH
from training.utils import plot_batch_predictions

MODEL_NAME = "2022-05-15T19-17-03.103670"
PLOT_IMAGES = False


def get_dataset_accuracy(
    model: tf.keras.Model, dataset: tf.data.Dataset, plot_images: bool = PLOT_IMAGES
) -> float:
    """Get the prediction accuracy given a trained model and a dataset."""
    num_correct, total = 0, 0
    for batch_images, batch_labels in dataset:
        preds: npt.NDArray[np.float32] = model.predict(batch_images)
        labels_preds: list[tuple[int, int]] = [
            (int(label), np.argmax(pred)) for label, pred in zip(batch_labels, preds)
        ]
        num_correct += sum(label == pred for label, pred in labels_preds)
        total += len(labels_preds)
        if plot_images:
            images_labels_preds: list[tuple] = [
                (image, *label_pred)
                for image, label_pred in zip(batch_images, labels_preds)
            ]
            plot_batch_predictions(images_labels_preds, dataset.class_names)
    return num_correct / total


# 1. Get the dataset(s)
train_ds, val_ds, test_ds = get_dataset(
    "small", image_size=VGG_IMAGE_SIZE, batch_size=16
)

# 2. Rehydrate the model
model_location = os.path.join(MODELS_DIR_ABS_PATH, MODEL_NAME)
model = tf.keras.models.load_model(model_location)

# 3. Make predictions on the entire dataset
train_set_accuracy = get_dataset_accuracy(model, train_ds)
val_set_accuracy = get_dataset_accuracy(model, val_ds)
test_set_accuracy = get_dataset_accuracy(model, test_ds)
_ = 0  # Debug here
