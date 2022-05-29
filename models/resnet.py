import os
from typing import NamedTuple, Optional, Tuple, TypedDict

import tensorflow as tf
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from models.resnet_utils import make_convolutional_block, make_identity_block
from training import MODELS_DIR_ABS_PATH
from training.utils import get_model_by_nickname

RESNET_IMAGE_SIZE = (224, 224)  # Channels last format
RESNET_IMAGE_SHAPE = (*RESNET_IMAGE_SIZE, 3)  # RGB
RESNET_TOP_FC_UNITS = 1000  # From the paper to match ImageNet


class Conv2DConfig(TypedDict):
    """Group parameters to pass to a Conv2D."""

    filters: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]


class MaxPool2DConfig(TypedDict):
    """Group parameters to pass to a MaxPool2D."""

    pool_size: Tuple[int, int]
    strides: Tuple[int, int]


class ConvBlockConfig(NamedTuple):
    """Group parameters to pass into a convolution helper function."""

    filters: Tuple[int, int, int]
    first_strides: Tuple[int, int]
    num_blocks: int


# These all come from the ResNet paper
CONV_1_CONV = Conv2DConfig(filters=64, kernel_size=(7, 7), strides=(2, 2))
CONV_2_MAXPOOL = MaxPool2DConfig(pool_size=(3, 3), strides=(2, 2))
CONV_2_CONV_FILTERS = ConvBlockConfig((64, 64, 256), (1, 1), 3)
CONV_3_CONV_FILTERS = ConvBlockConfig((128, 128, 512), (2, 2), 4)
CONV_4_CONV_FILTERS = ConvBlockConfig((256, 256, 1024), (2, 2), 6)
CONV_5_CONV_FILTERS = ConvBlockConfig((512, 512, 2048), (2, 2), 3)


def make_resnet_diy_model(top_fc_units: int = RESNET_TOP_FC_UNITS) -> tf.keras.Model:
    """
    Make a ResNet model given a number of classes and FC units.

    Args:
        top_fc_units: Number of units to use in the top FC layer.
            Default is one FC layer per the ResNet paper.

    Returns:
        ResNet model created.
    """

    def _conv1(x_: KerasTensor, config: Conv2DConfig = CONV_1_CONV) -> KerasTensor:
        base_name = "conv1"
        for layer in [
            tf.keras.layers.ZeroPadding2D(padding=(3, 3), name=f"{base_name}_pad"),
            tf.keras.layers.Conv2D(**config, name=f"{base_name}_conv"),
            tf.keras.layers.BatchNormalization(axis=3, name=f"{base_name}_bn"),
            tf.keras.layers.Activation(activation="relu", name=f"{base_name}_relu"),
        ]:
            x_ = layer(x_)
        return x_

    def _conv2(
        x_: KerasTensor,
        max_pool_config: MaxPool2DConfig = CONV_2_MAXPOOL,
        conv_block_config: ConvBlockConfig = CONV_2_CONV_FILTERS,
    ) -> KerasTensor:
        pool_base_name = "pool1"
        for layer in [
            tf.keras.layers.ZeroPadding2D(name=f"{pool_base_name}_pad"),
            tf.keras.layers.MaxPool2D(**max_pool_config, name=f"{pool_base_name}_pool"),
        ]:
            x_ = layer(x_)
        return _convx(x_, conv_block_config, stage=2)

    def _convx(
        x_: KerasTensor,
        conv_block_config: ConvBlockConfig,
        stage: int,
    ) -> KerasTensor:
        x_ = make_convolutional_block(
            x_,
            conv_block_config.filters,
            stage=stage,
            block=1,
            first_strides=conv_block_config.first_strides,
        )
        for i in range(2, conv_block_config.num_blocks + 1):
            x_ = make_identity_block(
                x_, conv_block_config.filters, stage=stage, block=i
            )
        return x_

    x_input = tf.keras.Input(shape=RESNET_IMAGE_SHAPE)
    x: KerasTensor = tf.keras.layers.Lambda(
        tf.keras.applications.resnet50.preprocess_input, name="preprocess_images"
    )(x_input)
    x = _conv1(x)
    x = _conv2(x)
    x = _convx(x, CONV_3_CONV_FILTERS, 3)
    x = _convx(x, CONV_4_CONV_FILTERS, 4)
    x = _convx(x, CONV_5_CONV_FILTERS, 5)
    for layer in [
        tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"),
        # Last layer matches number of classes
        tf.keras.layers.Dense(
            units=top_fc_units, activation="softmax", name="predictions"
        ),
    ]:
        x = layer(x)
    return tf.keras.Model(inputs=x_input, outputs=x, name="diy_resnet50")


def make_resnet_tl_model(
    top_fc_units: int = RESNET_TOP_FC_UNITS, base_model: Optional[tf.keras.Model] = None
) -> tf.keras.Model:
    """
    Make a ResNet transfer-learned model given a number of classes and FC units.

    Args:
        top_fc_units: Number of units to use in the top FC layer.
            Default is one FC layer per the ResNet paper.
        base_model: Optional base model to use for transfer learning.
            If left as default of None, use Keras ResNet50 trained on ImageNet.

    Returns:
        ResNet model created.
    """
    if base_model is None:
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False
        )
    base_model.trainable = False  # Freeze the model
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=RESNET_IMAGE_SHAPE),  # Specify input size
            tf.keras.layers.Lambda(
                function=tf.keras.applications.vgg16.preprocess_input,
                name="preprocess_images",
            ),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"),
            # Last layer matches number of classes
            tf.keras.layers.Dense(
                units=top_fc_units, activation="softmax", name="predictions"
            ),
        ],
        name="tl_resnet50",
    )


def load_resnet_model(filepath: str, include_top: bool = True) -> tf.keras.Model:
    """
    Load a ResNet model optionally including the top softmax layer.

    Args:
        filepath: Absolute path to the saved model.
        include_top: If you want to include the top global average pooling and FC layer.
            Set False (non-default) when you want to do transfer learning.

    Returns:
        ResNet model loaded.
    """
    model: tf.keras.Model = tf.keras.models.load_model(filepath)
    if not include_top:
        model = tf.keras.Model(inputs=model.input, outputs=model.layers[-1].output)
    return model


if __name__ == "__main__":
    loaded_model = load_resnet_model(
        os.path.join(MODELS_DIR_ABS_PATH, get_model_by_nickname("RESNET-TL"))
    )
    resnet_diy_model = make_resnet_diy_model(10)
    resnet_tl_model = make_resnet_tl_model(10)
    _ = 0  # Debug here
