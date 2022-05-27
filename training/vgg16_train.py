import os
from typing import List, Optional

import tensorflow as tf

from data.dataset_utils import get_dataset, get_num_classes
from models.vgg16 import VGG_IMAGE_SIZE, make_tl_model
from training import CKPTS_DIR_ABS_PATH, LOG_DIR_ABS_PATH, MODELS_DIR_ABS_PATH
from training.utils import get_ts_now_as_str
from training.vgg16_utils import vgg_preprocess_dataset

# Num epochs if not early stopped
MAX_NUM_EPOCHS = 64
# Patience of EarlyStopping callback
ES_PATIENCE_EPOCHS = 8
# Number of validation set batches to check after each epoch, set None to check
# all validation batches
VALIDATION_STEPS: Optional[int] = None


# 1. Prepare the training data
train_ds, val_ds, _ = get_dataset("small", image_size=VGG_IMAGE_SIZE)
train_ds = vgg_preprocess_dataset(train_ds)
train_steps_per_epoch: Optional[int] = train_ds.cardinality().numpy()
val_ds = vgg_preprocess_dataset(val_ds)
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
    train_ds,
    epochs=MAX_NUM_EPOCHS,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps_per_epoch,
    callbacks=[callbacks],
)

# 4. Save the model for future use
model.save(os.path.join(MODELS_DIR_ABS_PATH, current_ts))
_ = 0  # Debug here
