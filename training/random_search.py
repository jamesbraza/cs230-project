import os
from typing import Callable, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import loguniform, uniform

from data.dataset_utils import (
    FULL_DATASET_LABELS,
    get_dataset,
    get_label_overlap,
    get_num_classes,
    pass_class_names,
)
from models.vgg16 import (
    VGG_IMAGE_SIZE,
    VGG_TOP_FC_UNITS,
    TopFCUnits,
    make_vgg16_tl_model,
)
from models.vgg16_drop import make_tl_model
from training.utils import preprocess_dataset

# Num epochs if not early stopped
MAX_NUM_EPOCHS = 48
# Number of validation set batches to check after each epoch, set None to check
# all validation batches
VALIDATION_STEPS: Optional[int] = None


def perform_random_search():

    train_loss = []
    train_accu = []
    val_loss = []
    val_accu = []

    lower_reg = 1e-4
    upper_reg = 8e-4
    lower_drop = 0.6
    upper_drop = 0.75
    size = 16

    np.random.seed(seed=3)
    rv_loguniform = loguniform.rvs(lower_reg, upper_reg, size=size)
    np.random.seed(seed=99)
    rv_uniform = np.random.uniform(lower_drop, upper_drop, size=size)

    # 1. Prepare the training data
    train_ds, val_ds, _, labels = get_dataset("small", image_size=VGG_IMAGE_SIZE)
    train_ds = preprocess_dataset(train_ds)
    train_steps_per_epoch: Optional[int] = train_ds.cardinality().numpy()
    val_ds = preprocess_dataset(val_ds)
    if VALIDATION_STEPS is None:
        val_steps_per_epoch: int = val_ds.cardinality().numpy()
    else:
        val_steps_per_epoch = VALIDATION_STEPS
    # Pre-fetch batches so the GPU has minimal downtime
    train_ds = pass_class_names(train_ds, train_ds.prefetch(tf.data.AUTOTUNE))
    val_ds = pass_class_names(val_ds, val_ds.prefetch(tf.data.AUTOTUNE))

    for i, j in zip(rv_uniform, rv_loguniform):

        # 2. Create and compile the model
        model = make_tl_model(num_classes=get_num_classes(train_ds), rand_search=(i, j))

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # 3. Perform the actual training

        initial_epoch: int = 0

        history: tf.keras.callbacks.History = model.fit(
            train_ds,
            epochs=MAX_NUM_EPOCHS,
            initial_epoch=initial_epoch,
            steps_per_epoch=train_steps_per_epoch,
            validation_data=val_ds,
            validation_steps=val_steps_per_epoch,
        )
        print(history.history)

        train_loss.append(history.history["loss"][-1])
        train_accu.append(history.history["accuracy"][-1])
        val_loss.append(history.history["val_loss"][-1])
        val_accu.append(history.history["val_accuracy"][-1])
    return (rv_uniform, rv_loguniform, train_loss, train_accu, val_loss, val_accu)


x, y, train_loss, train_accu, val_loss, val_accu = perform_random_search()

print(train_accu)
print(val_accu)
for i, val in enumerate(val_accu):
    print(f"i:{i}, val:{val}, x:{x[i]}, y:{y[i]}")


plt.yscale("log")
plt.xlim([0, 1])
plt.scatter(x, y)
plt.xlabel("dropout rate")
plt.ylabel("lambda for L2 regularization")
