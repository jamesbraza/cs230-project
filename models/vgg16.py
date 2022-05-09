"""VGG docs: https://keras.io/api/applications/vgg/."""

import tensorflow as tf

VGG_IMAGE_SIZE = (224, 224)
VGG_IMAGE_SHAPE = (*VGG_IMAGE_SIZE, 3)  # RGB
VGG_TOP_FC_UNITS = [4096, 4096, 1000]
NUM_EPOCHS = 10


def make_tl_model(num_classes: int) -> tf.keras.Model:
    """Make a VGG16 model specified for a given number of classes."""
    base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
    base_model.trainable = False  # Freeze the model
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=VGG_IMAGE_SHAPE),  # Specify input size
            base_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(20, activation="relu"),
            tf.keras.layers.Dense(
                num_classes, activation="softmax"
            ),  # Match label count
        ]
    )
