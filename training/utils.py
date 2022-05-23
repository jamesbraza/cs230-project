import math
import os
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf

from training import MODELS_DIR_ABS_PATH


def get_ts_now_as_str() -> str:
    """Get an ISO 8601-compliant timestamp for use in naming."""
    return datetime.now().isoformat().replace(":", "_")


def get_most_recent_model(models_dir: str = MODELS_DIR_ABS_PATH) -> str:
    """Get the most recent model in the input directory."""
    most_recent_file: Tuple[str, Optional[datetime]] = "", None
    for file in os.listdir(models_dir):
        abs_file = os.path.join(models_dir, file)
        if os.path.isdir(abs_file):
            try:
                ts = datetime.fromisoformat(file.replace("_", ":"))
            except ValueError:
                continue
            if most_recent_file[1] is None or ts > most_recent_file[1]:
                most_recent_file = abs_file, ts
    if most_recent_file == "":
        raise LookupError(f"Didn't find a model in {models_dir}.")
    return most_recent_file[0]


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
