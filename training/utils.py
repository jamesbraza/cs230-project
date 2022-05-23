import math
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf


def get_ts_now_as_str() -> str:
    """Get an ISO 8601-compliant timestamp for use in naming."""
    return datetime.now().isoformat().replace(":", "-")


ImagesLabelsPreds = List[Tuple[tf.Tensor, int, int]]


def plot_batch_predictions(
    images_labels_preds: ImagesLabelsPreds, class_names: List[str]
) -> None:
    batch_size = len(images_labels_preds)
    subplot_size = int(math.sqrt(batch_size))
    if subplot_size**2 != batch_size:
        raise NotImplementedError(
            f"Only works for perfect-square batch sizes, "
            f"not a batch size of {batch_size}."
        )
    fig, ax = plt.subplots(nrows=subplot_size, ncols=subplot_size)
    for i, (image, label, pred) in enumerate(images_labels_preds):
        fig.axes[i].imshow(image.numpy().astype("uint8"))
        fig.axes[i].set_title(
            f"Label: {label}: {class_names[label]}\n"
            f"Pred: {pred}: {class_names[pred]}"
        )
    fig.tight_layout()  # Prevent title overlap
    plt.show()
