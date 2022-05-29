"""VGG docs: https://keras.io/api/applications/vgg/."""

from typing import Tuple

import tensorflow as tf

TopFCUnits = Tuple[int, ...]

VGG_IMAGE_SIZE = (224, 224)  # Channels last format
VGG_IMAGE_SHAPE = (*VGG_IMAGE_SIZE, 3)  # RGB
VGG_TOP_FC_UNITS: TopFCUnits = (4096, 4096, 1000)  # From the paper to match ImageNet


def make_tl_model(top_fc_units: TopFCUnits = VGG_TOP_FC_UNITS) -> tf.keras.Model:
    """
    Make a VGG16 model given a number of classes and FC units.

    Args:
        top_fc_units: Number of units to use in each of the top FC layers.
            Default is three FC layers per the VGGNet paper.
            Last value in the tuple should match your number of classes.

    Returns:
        VGGNet model created.
    """
    base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
    base_model.trainable = False  # Freeze the model
    dense_layers = [
        tf.keras.layers.Dense(units=units, activation="relu", name=f"fc{i+1}")
        for i, units in enumerate(top_fc_units[:-1])
    ]
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=VGG_IMAGE_SHAPE),  # Specify input size
            tf.keras.layers.Lambda(
                function=tf.keras.applications.vgg16.preprocess_input,
                name="preprocess_images",
            ),
            base_model,
            tf.keras.layers.Flatten(),
            *dense_layers,
            # Last layer matches number of classes
            tf.keras.layers.Dense(
                units=top_fc_units[-1], activation="softmax", name="predictions"
            ),
        ],
        name="diy_tl_vgg16",
    )
