from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf

from data.dataset_utils import get_num_classes, pass_class_names


def make_vgg_preprocessing_generator(
    dataset: tf.data.Dataset, num_repeat: int = -1, preprocess_image: bool = False
) -> Iterable[Tuple[tf.Tensor, np.ndarray]]:
    """
    Make an iterator that pre-processes a dataset for VGGNet training.

    Args:
        dataset: TensorFlow dataset to preprocess.
        num_repeat: Optional count of dataset repetitions.
            Default of -1 will repeat indefinitely.
            For training data: set to the number of epochs, or -1.
        preprocess_image: Set True to pre-process the image per VGG16's preprocessor.
            Default is False because this is built into the model.

    Returns:
        Iterable of (image, categorical label) tuples.
    """
    num_classes = get_num_classes(dataset)
    for batch_images, batch_labels in dataset.repeat(num_repeat):
        if preprocess_image:
            batch_images = tf.keras.applications.vgg16.preprocess_input(batch_images)
        yield batch_images, tf.keras.utils.to_categorical(
            batch_labels, num_classes, dtype="bool"
        )


def vgg_preprocess_dataset(
    dataset: tf.data.Dataset, preprocess_image: bool = False
) -> tf.data.Dataset:
    """
    Preprocess the input dataset for a VGG16 network.

    Args:
        dataset: TensorFlow dataset to preprocess.
        preprocess_image: Set True to pre-process the image per VGG16's preprocessor.
            Default is False because this is built into the model.

    Returns:
        Preprocessed dataset.
    """
    num_classes = get_num_classes(dataset)

    def _preprocess(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if preprocess_image:
            x = tf.keras.applications.vgg16.preprocess_input(x)
        # NOTE: one_hot is a transformation step for Tensors, so we use it here
        # over to_categorical
        # pylint: disable=no-value-for-parameter
        return x, tf.one_hot(y, depth=num_classes)

    return pass_class_names(dataset, dataset.map(_preprocess))
