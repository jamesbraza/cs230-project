import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from data.dataset_utils import (
    FULL_SMALL_LABELS,
    SMALL_DATASET_LABELS,
    get_dataset,
    get_num_classes,
)
from models.vgg16 import VGG_IMAGE_SIZE, make_tl_model
from training import CKPTS_DIR_ABS_PATH, LOG_DIR_ABS_PATH, MODELS_DIR_ABS_PATH
from training.utils import get_ts_now_as_str

# Num epochs if not early stopped
MAX_NUM_EPOCHS = 64
# Patience of EarlyStopping callback
ES_PATIENCE_EPOCHS = 8
# Number of validation set batches to check after each epoch, set None to check
# all validation batches
VALIDATION_STEPS: Optional[int] = None
# If you want to mix in the full clothing dataset
DATA_AUGMENTATION = True


def make_vgg_preprocessing_generator(
    dataset: tf.data.Dataset, num_repeat: int = -1, preprocess_image: bool = False
) -> Iterable[Tuple[tf.Tensor, np.ndarray]]:
    """
    Make an iterator that pre-processes a dataset for VGGNet training.

    Args:
        dataset: TensorFlow dataset to preprocess.
        num_repeat: Optional count of dataset repetitions.
            Default of -1 will repeat indefinitely.
            For training data: set to the number of epochs, or -1.
        preprocess_image: Set True to pre-process the image per VGG16's preprocessor.
            Default is False because this is built into the model.

    Returns:
        Iterable of (image, categorical label) tuples.
    """
    for batch_images, batch_labels in dataset.repeat(num_repeat):
        if preprocess_image:
            batch_images = tf.keras.applications.vgg16.preprocess_input(batch_images)
        yield batch_images, tf.keras.utils.to_categorical(
            batch_labels, get_num_classes(dataset), dtype="bool"
        )


# 1. Prepare the training data
train_ds, val_ds, _ = get_dataset("small", image_size=VGG_IMAGE_SIZE)
if DATA_AUGMENTATION:
    full_train_ds, _, _ = get_dataset(
        "full",
        image_size=VGG_IMAGE_SIZE,
        validation_split=0.0,
        filter_labels={
            x: SMALL_DATASET_LABELS.index(x)
            for x in FULL_SMALL_LABELS
            if x in SMALL_DATASET_LABELS
        },
    )
    class_names = train_ds.class_names
    train_ds = train_ds.concatenate(full_train_ds)
    train_ds.class_names = class_names  # Manually propagate
train_data_generator = make_vgg_preprocessing_generator(train_ds)
train_steps_per_epoch: Optional[int] = train_ds.cardinality().numpy()
if train_steps_per_epoch < 0:
    # SEE: https://github.com/tensorflow/tensorflow/issues/44933
    train_steps_per_epoch = None
val_data_generator = make_vgg_preprocessing_generator(val_ds)
if VALIDATION_STEPS is None:
    val_steps_per_epoch: int = val_ds.cardinality().numpy()
else:
    val_steps_per_epoch = VALIDATION_STEPS

# 2. Create and compile the model
model = make_tl_model(
    num_classes=get_num_classes(train_ds), top_fc_units=(128, 128, 32)
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 3. Perform the actual training
current_ts = get_ts_now_as_str()
ckpt_filename = os.path.join(
    CKPTS_DIR_ABS_PATH,
    "%s--{epoch:02d}--{loss:.2f}.hdf5" % current_ts,
)
callbacks: List[tf.keras.callbacks.Callback] = [
    tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR_ABS_PATH, histogram_freq=1),
    tf.keras.callbacks.ModelCheckpoint(ckpt_filename, save_best_only=True),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=ES_PATIENCE_EPOCHS,
        restore_best_weights=True,
        verbose=1,
    ),
]
history: tf.keras.callbacks.History = model.fit(
    train_data_generator,
    epochs=MAX_NUM_EPOCHS,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=val_data_generator,
    validation_steps=val_steps_per_epoch,
    callbacks=[callbacks],
)

# 4. Save the model for future use
model.save(os.path.join(MODELS_DIR_ABS_PATH, current_ts))
_ = 0  # Debug here
