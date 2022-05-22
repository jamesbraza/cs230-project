import os
from collections.abc import Iterable

import numpy.typing as npt
import tensorflow as tf

from data.dataset_utils import get_dataset, get_num_classes
from models.vgg16 import VGG_IMAGE_SIZE, make_tl_model
from training import CKPTS_DIR_ABS_PATH, LOG_DIR_ABS_PATH, MODELS_DIR_ABS_PATH
from training.utils import get_ts_now_as_str

MAX_NUM_EPOCHS = 16  # Num epochs if not early stopped
ES_PATIENCE_EPOCHS = 4  # EarlyStopping
VALIDATION_STEPS = 4


def make_vgg_preprocessing_generator(
    dataset: tf.data.Dataset, num_epochs: int = -1, preprocess_image: bool = False
) -> Iterable[tuple[tf.Tensor, npt.NDArray[tf.bool]]]:
    """
    Make an iterator that pre-processes a dataset for VGGNet training.

    Args:
        dataset: TensorFlow dataset to preprocess.
        num_epochs: Optional count of training epochs.
            Default of -1 will repeat indefinitely.
        preprocess_image: Set True to pre-process the image per VGG16's preprocessor.
            Default is False because this is built into the model.

    Returns:
        Iterable of (image, categorical label) tuples.
    """
    for batch_images, batch_labels in dataset.repeat(num_epochs):
        if preprocess_image:
            batch_images = tf.keras.applications.vgg16.preprocess_input(batch_images)
        yield batch_images, tf.keras.utils.to_categorical(
            batch_labels, get_num_classes(dataset), dtype="bool"
        )


# 1. Prepare the training data
train_ds, val_ds, _ = get_dataset("small", image_size=VGG_IMAGE_SIZE)
train_data_generator = make_vgg_preprocessing_generator(train_ds)
steps_per_epoch: int = train_ds.cardinality().numpy()  # Full training set
val_data_generator = make_vgg_preprocessing_generator(val_ds)

# 2. Create and compile the model
model = make_tl_model(num_classes=get_num_classes(train_ds), top_fc_units=(64, 64, 16))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 3. Perform the actual training
current_ts = get_ts_now_as_str()
ckpt_filename = os.path.join(
    CKPTS_DIR_ABS_PATH,
    "%s--{epoch:02d}--{loss:.2f}.hdf5" % current_ts,
)
callbacks: list[tf.keras.callbacks.Callback] = [
    tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR_ABS_PATH, histogram_freq=1),
    tf.keras.callbacks.ModelCheckpoint(ckpt_filename, save_best_only=True),
    tf.keras.callbacks.EarlyStopping(
        patience=ES_PATIENCE_EPOCHS, restore_best_weights=True
    ),
]
history: tf.keras.callbacks.History = model.fit(
    train_data_generator,
    epochs=MAX_NUM_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=[callbacks],
)

# 4. Save the model for future use
model.save(os.path.join(MODELS_DIR_ABS_PATH, current_ts))
_ = 0  # Debug here
