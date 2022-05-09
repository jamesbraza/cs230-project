import os

import tensorflow as tf

from data.make import get_dataset, get_num_classes, make_vgg_preprocessing_generator
from models.vgg16 import VGG_IMAGE_SIZE, make_tl_model

NUM_EPOCHS = 10
LOG_DIR_ABS_PATH = os.getcwd()

# 1. Prepare the training data
train_ds, _, _ = get_dataset("small", image_size=VGG_IMAGE_SIZE)
num_classes = get_num_classes(train_ds)
train_data_generator = make_vgg_preprocessing_generator(train_ds, NUM_EPOCHS)
steps_per_epoch: int = train_ds.cardinality().numpy()

# 2. Create and compile the model
model = make_tl_model(num_classes)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 3. Fit the training data
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=LOG_DIR_ABS_PATH, histogram_freq=1
)
model.fit(
    train_data_generator,
    epochs=NUM_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    callbacks=[tensorboard_callback],
)
_ = 0
