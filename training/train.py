import os
from typing import Callable, List, Literal, Optional, Tuple, Union

import tensorflow as tf

from data.dataset_utils import (
    FULL_DATASET_LABELS,
    SMALL_TRAIN_ABS_PATH,
    get_dataset,
    get_label_overlap,
    pass_class_names,
)
from models.resnet import RESNET_IMAGE_SIZE, make_resnet_diy_model, make_resnet_tl_model
from models.vgg16 import (
    VGG_IMAGE_SIZE,
    VGG_TOP_FC_UNITS,
    TopFCUnits,
    make_vgg16_tl_model,
)
from training import CKPTS_DIR_ABS_PATH, LOG_DIR_ABS_PATH, MODELS_DIR_ABS_PATH
from training.utils import (
    DEFAULT_DELIM,
    DEFAULT_SAVE_NICKNAME,
    get_ts_now_as_str,
    preprocess_dataset,
)

# Num epochs if not early stopped
MAX_NUM_EPOCHS = 64
# Patience of EarlyStopping callback
ES_PATIENCE_EPOCHS = 8
# Number of validation set batches to check after each epoch, set None to check
# all validation batches
VALIDATION_STEPS: Optional[int] = None
# 0: no data augmentation
# 1: if you want to use ImageDataGenerator to slightly randomize images
# 2: if you want to mix in the full clothing dataset
DATA_AUGMENTATION: Literal[0, 1, 2] = 0
# Set to the last checkpoint if you want to resume training,
# or leave as None to begin anew
LAST_CHECKPOINT: Optional[str] = None
# Set to a nickname for the save file to help facilitate reuse
SAVE_NICKNAME: str = "RESNET-TL"
# Which model to train
MODEL: Literal["vgg16_tl", "resnet_diy", "resnet_tl"] = "resnet_tl"

if MODEL.startswith("vgg16"):
    image_size: Tuple[int, int] = VGG_IMAGE_SIZE
    model_factory: Callable[..., tf.keras.Model] = make_vgg16_tl_model
elif MODEL.startswith("resnet"):
    image_size = RESNET_IMAGE_SIZE
    if MODEL == "resnet_diy":
        model_factory = make_resnet_diy_model
    else:
        model_factory = make_resnet_tl_model
else:
    raise NotImplementedError(f"Unrecognized model: {MODEL}.")

# 1. Prepare the training data
train_ds, val_ds, _, labels = get_dataset("small", image_size=image_size)
train_ds = preprocess_dataset(train_ds)
train_steps_per_epoch: Optional[int] = train_ds.cardinality().numpy()
if DATA_AUGMENTATION == 1:  # Data aug via ImageDataGenerator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20.0,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )
    train_datagen = datagen.flow_from_directory(
        SMALL_TRAIN_ABS_PATH, target_size=image_size, seed=42
    )
elif DATA_AUGMENTATION == 2:  # Data aug via merging with full dataset
    full_train_ds, _, _, _ = get_dataset(
        "full",
        image_size=image_size,
        validation_split=0.0,
        filter_labels=get_label_overlap(labels, other_ds_labels=FULL_DATASET_LABELS),
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

# TODO: figure out how to consolidate this if statements with the above
if MODEL.startswith("vgg16"):
    top_fc_units: Union[TopFCUnits, int] = (*VGG_TOP_FC_UNITS[:-1], len(labels))
elif MODEL.startswith("resnet"):
    top_fc_units = len(labels)
else:
    raise NotImplementedError(f"Unrecognized model: {MODEL}.")

# 2. Create and compile the model
model = model_factory(top_fc_units=top_fc_units)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 3. Perform the actual training
current_ts = get_ts_now_as_str()
ckpt_filename = os.path.join(
    CKPTS_DIR_ABS_PATH,
    DEFAULT_DELIM.join(["%s", SAVE_NICKNAME, "{epoch:02d}", "{loss:.2f}.hdf5"])
    % current_ts,
)
initial_epoch: int = 0
if LAST_CHECKPOINT is not None:  # Recover from checkpoint
    initial_epoch = int(LAST_CHECKPOINT.split(DEFAULT_DELIM)[1])
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
    train_ds if DATA_AUGMENTATION != 1 else train_datagen,
    epochs=MAX_NUM_EPOCHS,
    initial_epoch=initial_epoch,
    steps_per_epoch=train_steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps_per_epoch,
    callbacks=[callbacks],
)

# 4. Save the model for future use
model.save(
    os.path.join(MODELS_DIR_ABS_PATH, f"{current_ts}{DEFAULT_DELIM}{SAVE_NICKNAME}")
)
_ = 0  # Debug here
