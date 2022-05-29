from typing import Tuple

import tensorflow as tf
from tensorflow.python.keras.engine.keras_tensor import KerasTensor


def make_identity_block(
    x: KerasTensor,
    filters: Tuple[int, int, int],
    stage: int,
    block: int,
) -> KerasTensor:
    """
    Make a "bottleneck" building block for a ResNet.

    Args:
        x: Model under construction to place the identity block on top of.
        filters: Three-tuple of the three Conv2D's output dimensionality.
        stage: Stage number for layer names.
        block: Block number for layer names, where blocks are stage sub-parts.

    Returns:
        Further constructed model.
    """
    base_name = f"conv{stage}_block{block}"
    x_nonshortcut = x_shortcut = x
    for layer in [
        tf.keras.layers.Conv2D(
            filters=filters[0],
            kernel_size=(1, 1),
            name=f"{base_name}_1_conv",
        ),
        tf.keras.layers.BatchNormalization(axis=3, name=f"{base_name}_1_bn"),
        tf.keras.layers.Activation(activation="relu", name=f"{base_name}_1_relu"),
        tf.keras.layers.Conv2D(
            filters=filters[1],
            kernel_size=(3, 3),
            padding="same",
            name=f"{base_name}_2_conv",
        ),
        tf.keras.layers.BatchNormalization(axis=3, name=f"{base_name}_2_bn"),
        tf.keras.layers.Activation(activation="relu", name=f"{base_name}_2_relu"),
        tf.keras.layers.Conv2D(
            filters=filters[2],
            kernel_size=(1, 1),
            name=f"{base_name}_3_conv",
        ),
        tf.keras.layers.BatchNormalization(axis=3, name=f"{base_name}_3_bn"),
    ]:
        x_nonshortcut = layer(x_nonshortcut)
    x = tf.keras.layers.Add(name=f"{base_name}_add")([x_shortcut, x_nonshortcut])
    return tf.keras.layers.Activation(activation="relu", name=f"{base_name}_out")(x)


def make_convolutional_block(
    x: KerasTensor,
    filters: Tuple[int, int, int],
    stage: int,
    block: int,
    first_strides: Tuple[int, int] = (1, 1),
) -> KerasTensor:
    """
    Make a "convolutional" building block for a ResNet.

    Args:
        x: Model under construction to place the identity block on top of.
        filters: Three-tuple of the three Conv2D's output dimensionality.
        stage: Stage number for layer names.
        block: Block number for layer names, where blocks are stage sub-parts.
        first_strides: Strides to use in the first convolutional layer.

    Returns:
        Further constructed model.
    """
    base_name = f"conv{stage}_block{block}"
    x_nonshortcut = x_shortcut = x
    for layer in [
        tf.keras.layers.Conv2D(
            filters=filters[2],
            kernel_size=(1, 1),
            strides=first_strides,
            name=f"{base_name}_0_conv",
        ),
        tf.keras.layers.BatchNormalization(axis=3, name=f"{base_name}_0_bn"),
    ]:
        x_shortcut = layer(x_shortcut)
    for layer in [
        tf.keras.layers.Conv2D(
            filters=filters[0],
            kernel_size=(1, 1),
            strides=first_strides,
            name=f"{base_name}_1_conv",
        ),
        tf.keras.layers.BatchNormalization(axis=3, name=f"{base_name}_1_bn"),
        tf.keras.layers.Activation(activation="relu", name=f"{base_name}_1_relu"),
        tf.keras.layers.Conv2D(
            filters=filters[1],
            kernel_size=(3, 3),
            padding="same",
            name=f"{base_name}_2_conv",
        ),
        tf.keras.layers.BatchNormalization(axis=3, name=f"{base_name}_2_bn"),
        tf.keras.layers.Activation(activation="relu", name=f"{base_name}_2_relu"),
        tf.keras.layers.Conv2D(
            filters=filters[2],
            kernel_size=(1, 1),
            name=f"{base_name}_3_conv",
        ),
        tf.keras.layers.BatchNormalization(axis=3, name=f"{base_name}_3_bn"),
    ]:
        x_nonshortcut = layer(x_nonshortcut)
    x = tf.keras.layers.Add(name=f"{base_name}_add")([x_shortcut, x_nonshortcut])
    return tf.keras.layers.Activation(activation="relu", name=f"{base_name}_out")(x)
