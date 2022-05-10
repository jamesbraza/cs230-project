import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from data.dataset_utils import get_dataset
from models.vgg16 import VGG_IMAGE_SIZE
from training import MODELS_DIR_ABS_PATH

MODEL_NAME = "first"


# 1. Get the test data
_, _, test_ds = get_dataset("small", image_size=VGG_IMAGE_SIZE, batch_size=16)

# 2. Rehydrate the model
model_location = os.path.join(MODELS_DIR_ABS_PATH, MODEL_NAME)
model = tf.keras.models.load_model(model_location)

# 3. Make predictions on the entire test dataset
for batch_images, batch_labels in test_ds:
    preprocessed_images = tf.keras.applications.vgg16.preprocess_input(batch_images)
    preds: npt.NDArray[np.float] = model.predict(preprocessed_images)
    fig, ax = plt.subplots(nrows=4, ncols=4)
    for i, image in enumerate(batch_images):
        fig.axes[i].imshow(image.numpy().astype("uint8"))
        pred_label = np.argmax(preds[i])
        fig.axes[i].set_title(
            f"Label: {int(batch_labels[i])}: {test_ds.class_names[batch_labels[i]]}\n"
            f"Pred: {pred_label}: {test_ds.class_names[pred_label]}"
        )
    fig.tight_layout()  # Prevent title overlap
    plt.show()
    _ = 0  # Debug here
