import os
from typing import Callable, List, Literal, Optional, Tuple, Union

import tensorflow as tf

from data.dataset_utils import (
    FULL_DATASET_LABELS,
    SMALL_DATASET_LABELS,
    get_dataset,
    get_label_overlap,
    pass_class_names,
)
from models.vgg16 import VGG_IMAGE_SIZE, VGG_TOP_FC_UNITS, TopFCUnits, make_tl_model
from training import CKPTS_DIR_ABS_PATH, LOG_DIR_ABS_PATH, MODELS_DIR_ABS_PATH
from training.utils import get_ts_now_as_str, preprocess_dataset

# Num epochs if not early stopped
MAX_NUM_EPOCHS = 64
# Patience of EarlyStopping callback
ES_PATIENCE_EPOCHS = 8
# Number of validation set batches to check after each epoch, set None to check
# all validation batches
VALIDATION_STEPS: Optional[int] = None
# If you want to mix in the full clothing dataset
DATA_AUGMENTATION = True
# Set to the last checkpoint if you want to resume training,
# or leave as None to begin anew
LAST_CHECKPOINT: Optional[str] = None
# Which model to train
MODEL: Literal["vgg16"] = "vgg16"

if MODEL == "vgg16":
    image_size: Tuple[int, int] = VGG_IMAGE_SIZE
    model_factory: Callable[..., tf.keras.Model] = make_tl_model
    top_fc_units: Union[TopFCUnits, int] = (
        *[u for u in VGG_TOP_FC_UNITS[:-1]],
        len(SMALL_DATASET_LABELS),
    )
else:
    raise NotImplementedError(f"Unrecognized model: {MODEL}.")

# 1. Prepare the training data
train_ds, val_ds, _ = get_dataset("small", image_size=image_size)
train_ds = preprocess_dataset(train_ds)
train_steps_per_epoch: Optional[int] = train_ds.cardinality().numpy()
if DATA_AUGMENTATION:
    full_train_ds, _, _ = get_dataset(
        "full",
        image_size=image_size,
        validation_split=0.0,
        filter_labels=get_label_overlap(
            SMALL_DATASET_LABELS, other_ds_labels=FULL_DATASET_LABELS
        ),
    )
    full_train_ds = preprocess_dataset(full_train_ds)
    train_ds = pass_class_names(train_ds, train_ds.concatenate(full_train_ds))
    if train_ds.cardinality().numpy() == tf.data.experimental.UNKNOWN_CARDINALITY:
        train_steps_per_epoch = None
val_ds = preprocess_dataset(val_ds)
if VALIDATION_STEPS is None:
    val_steps_per_epoch: int = val_ds.cardinality().numpy()
else:
    val_steps_per_epoch = VALIDATION_STEPS
# Pre-fetch batches so the GPU has minimal downtime
train_ds = pass_class_names(train_ds, train_ds.prefetch(tf.data.AUTOTUNE))
val_ds = pass_class_names(val_ds, val_ds.prefetch(tf.data.AUTOTUNE))

# 2. Create and compile the model
model = model_factory(top_fc_units=top_fc_units)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 3. Perform the actual training
current_ts = get_ts_now_as_str()
checkpoint_delim = "--"
ckpt_filename = os.path.join(
    CKPTS_DIR_ABS_PATH,
    checkpoint_delim.join(["%s", "{epoch:02d}", "{loss:.2f}.hdf5"]) % current_ts,
)
initial_epoch: int = 0
if LAST_CHECKPOINT is not None:  # Recover from checkpoint
    initial_epoch = int(LAST_CHECKPOINT.split(checkpoint_delim)[1])
    model.load_weights(os.path.join(CKPTS_DIR_ABS_PATH, LAST_CHECKPOINT))
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
    initial_epoch=initial_epoch,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps_per_epoch,
    callbacks=[callbacks],
)

# 4. Save the model for future use
model.save(os.path.join(MODELS_DIR_ABS_PATH, current_ts))
_ = 0  # Debug here
