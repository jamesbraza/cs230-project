"""VGG docs: https://keras.io/api/applications/vgg/."""

from typing import Tuple

import tensorflow as tf

RES_IMAGE_SIZE = (224, 224)
RES_IMAGE_SHAPE = (*RES_IMAGE_SIZE, 3)  # RGB
RES_TOP_FC_UNITS = 1000


def make_tl_model(
    num_classes: int, top_fc_units: Tuple[int, ...] = RES_TOP_FC_UNITS
) -> tf.keras.Model:
    """
    Make a ResNet50 V2 model given a number of classes and FC units.

    Args:
        num_classes: Number of classes for the final softmax layer.
        top_fc_units: Number of units to use in each of the top FC layers.
            Default is three FC layers.

    Returns:
        Model created.
    """
    base_model = tf.keras.applications.ResNet50V2(weights="imagenet", include_top=False)
    base_model.trainable = False  # Freeze the model
    
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=RES_IMAGE_SHAPE),  # Specify input size
            tf.keras.layers.Lambda(tf.keras.applications.resnet_v2.preprocess_input),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"),
            tf.keras.layers.Dense(top_fc_units, activation="relu"),
            # Last layer matches number of classes
            tf.keras.layers.Dense(num_classes, activation="softmax")
        ]
    )
