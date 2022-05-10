import os
from collections.abc import Iterator

import numpy.typing as npt
import tensorflow as tf

from data.dataset_utils import get_dataset, get_num_classes
from models.vgg16 import VGG_IMAGE_SIZE, make_tl_model
from training import CKPTS_DIR_ABS_PATH, LOG_DIR_ABS_PATH, MODELS_DIR_ABS_PATH
from training.utils import get_ts_now_as_str

NUM_EPOCHS = 10


def make_vgg_preprocessing_generator(
    dataset: tf.data.Dataset, num_epochs: int
) -> Iterator[tuple[tf.Tensor, npt.NDArray[tf.bool]]]:
    """Make an iterator that pre-processes a dataset for VGGNet training."""
    num_classes = get_num_classes(dataset)
    for batch_images, batch_labels in dataset.repeat(num_epochs):
        yield tf.keras.applications.vgg16.preprocess_input(
            batch_images
        ), tf.keras.utils.to_categorical(batch_labels, num_classes, dtype="bool")


# 1. Prepare the training data
train_ds, _, _ = get_dataset("small", image_size=VGG_IMAGE_SIZE)
train_data_generator = make_vgg_preprocessing_generator(train_ds, NUM_EPOCHS)
steps_per_epoch: int = train_ds.cardinality().numpy()

# 2. Create and compile the model
model = make_tl_model(num_classes=get_num_classes(train_ds), top_fc_units=(50, 50, 20))
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
]
model.fit(
    train_data_generator,
    epochs=NUM_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    callbacks=[callbacks],
)

# 4. Save the model for future use
model.save(os.path.join(MODELS_DIR_ABS_PATH, current_ts))
