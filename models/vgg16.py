"""VGG docs: https://keras.io/api/applications/vgg/."""

from typing import Tuple

import tensorflow as tf

VGG_IMAGE_SIZE = (224, 224)
VGG_IMAGE_SHAPE = (*VGG_IMAGE_SIZE, 3)  # RGB
VGG_TOP_FC_UNITS = (4096, 4096, 1000)
NUM_EPOCHS = 10


def make_tl_model(
    num_classes: int, top_fc_units: Tuple[int, ...] = VGG_TOP_FC_UNITS
) -> tf.keras.Model:
    """
    Make a VGG16 model given a number of classes and FC units.

    Args:
        num_classes: Number of classes for the final softmax layer.
        top_fc_units: Number of units to use in each of the top FC layers.
            Default is three FC layers per the VGGNet paper.

    Returns:
        Model created.
    """
    tf.random.set_seed(0)
    base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
    base_model.trainable = False  # Freeze the model
    dense_layers = [
            tf.keras.layers.Dense(top_fc_units[0], activation="relu"),
            tf.keras.layers.Dense(top_fc_units[1], activation="relu"),
            tf.keras.layers.Dense(top_fc_units[2], activation="relu")
    ]
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=VGG_IMAGE_SHAPE),  # Specify input size
            tf.keras.layers.Lambda(tf.keras.applications.vgg16.preprocess_input),
            base_model,
            tf.keras.layers.Flatten(),
            *dense_layers,
            # Last layer matches number of classes
            tf.keras.layers.Dense(num_classes, activation="softmax",
                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        ]
    )
