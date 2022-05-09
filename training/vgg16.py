import os
from datetime import datetime

import tensorflow as tf

from data.make import get_dataset, get_num_classes, make_vgg_preprocessing_generator
from models.vgg16 import VGG_IMAGE_SIZE, make_tl_model

NUM_EPOCHS = 10
LOG_DIR_ABS_PATH = os.path.join(os.getcwd(), "logs")
CKPTS_DIR_ABS_PATH = os.path.join(os.getcwd(), "checkpoints")


# 1. Prepare the training data
train_ds, _, _ = get_dataset("small", image_size=VGG_IMAGE_SIZE)
num_classes = get_num_classes(train_ds)
train_data_generator = make_vgg_preprocessing_generator(train_ds, NUM_EPOCHS)
steps_per_epoch: int = train_ds.cardinality().numpy()

# 2. Create and compile the model
model = make_tl_model(num_classes)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 3. Fit the training data
ckpt_filename = os.path.join(
    CKPTS_DIR_ABS_PATH,
    "%s--{epoch:02d}--{loss:.2f}.hdf5" % datetime.now().isoformat().replace(":", "-"),
)
callbacks: list[tf.keras.callbacks.Callback] = [
    tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR_ABS_PATH, histogram_freq=1),
    tf.keras.callbacks.ModelCheckpoint(ckpt_filename),
]
model.fit(
    train_data_generator,
    epochs=NUM_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    callbacks=[callbacks],
)
_ = 0
