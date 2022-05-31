import os
from typing import Callable, List, Literal, Optional, Tuple, Union

import tensorflow as tf
import numpy as np

from scipy.stats import loguniform, uniform

from data.dataset_utils import (
    FULL_DATASET_LABELS,
    get_dataset,
    get_label_overlap,
    pass_class_names,
    get_num_classes
)
from models.vgg16_drop import make_tl_model
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
MAX_NUM_EPOCHS = 1
# Patience of EarlyStopping callback
ES_PATIENCE_EPOCHS = 24
# Number of validation set batches to check after each epoch, set None to check
# all validation batches
VALIDATION_STEPS: Optional[int] = None
# If you want to mix in the full clothing dataset
DATA_AUGMENTATION = False
# Set to the last checkpoint if you want to resume training,
# or leave as None to begin anew
LAST_CHECKPOINT: Optional[str] = None
# Set to a nickname for the save file to help facilitate reuse
SAVE_NICKNAME: str = DEFAULT_SAVE_NICKNAME
# Which model to train

def perform_random_search():

    train_loss = []
    train_accu = []
    val_loss = []
    val_accu = []

    lower_reg = 1e-5
    upper_reg = 1
    lower_drop = 0.1
    upper_drop = 0.8
    size = 16

    np.random.seed(seed=3)
    rv_loguniform = loguniform.rvs(lower_reg, upper_reg,size=size)
    np.random.seed(seed=99)
    rv_uniform = np.random.uniform(lower_drop, upper_drop, size=size)

    print(rv_loguniform)
    print(rv_uniform)

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

    for i,j in zip(rv_uniform,rv_loguniform):

        # 2. Create and compile the model
        model = make_tl_model(num_classes=get_num_classes(train_ds), rand_search=(i,j))

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
            tf.keras.callbacks.ModelCheckpoint(ckpt_filename, save_best_only=False),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=ES_PATIENCE_EPOCHS,
                restore_best_weights=False,
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


        train_loss.append(history.history['loss'][-1])
        train_accu.append(history.history['accuracy'][-1])
        val_loss.append(history.history['val_loss'][-1])
        val_accu.append(history.history['val_accuracy'][-1])

    return (rv_uniform, rv_loguniform, train_loss, train_accu, val_loss, val_accu)

   
