"""VGG docs: https://keras.io/api/applications/vgg/."""
import os
from typing import Optional, Tuple

import tensorflow as tf

from training import MODELS_DIR_ABS_PATH
from training.utils import get_path_to_model_by_nickname

TopFCUnits = Tuple[int, ...]

VGG_IMAGE_SIZE = (224, 224)  # Channels last format
VGG_IMAGE_SHAPE = (*VGG_IMAGE_SIZE, 3)  # RGB
VGG_TOP_FC_UNITS: TopFCUnits = (4096, 4096, 1000)  # From the paper to match ImageNet


def make_vgg16_tl_model(
    top_fc_units: TopFCUnits = VGG_TOP_FC_UNITS,
    base_model: Optional[tf.keras.Model] = None,
) -> tf.keras.Model:
    """
    Make a VGG16 model given a number of classes and FC units.

    Args:
        top_fc_units: Number of units to use in each of the top FC layers.
            Default is three FC layers per the VGGNet paper.
            Last value in the tuple should match your number of classes.
        base_model: Optional base model to use for transfer learning.
            If left as default of None, use Keras VGG16 trained on ImageNet.

    Returns:
        VGGNet model created.
    """
    if base_model is None:
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
        name="tl_vgg16",
    )


def load_vgg_model(filepath: str, include_top: bool = True) -> tf.keras.Model:
    """
    Load a VGG model optionally including the top softmax layer.

    Args:
        filepath: Absolute path to the saved model.
        include_top: If you want to include the top three FC layers.
            Set False (non-default) when you want to do transfer learning.

    Returns:
        VGG model loaded.
    """
    model: tf.keras.Model = tf.keras.models.load_model(filepath)
    if not include_top:
        model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    return model


if __name__ == "__main__":
    loaded_model = load_vgg_model(
        os.path.join(MODELS_DIR_ABS_PATH, get_path_to_model_by_nickname("BASELINE"))
    )
    vgg16_tl_model = make_vgg16_tl_model((4096, 4096, 10))
    _ = 0  # Debug here
