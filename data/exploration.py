import os.path

import pandas as pd

DIR_ABS_PATH = os.path.dirname(os.path.abspath(__file__))
FULL_CSV_REL_PATH = os.path.join("clothing_dataset_full", "images.csv")
SHIRTS_CSV_REL_PATH = os.path.join("shirts_dataset", "Dataset", "data.csv")


def explore_full_dataset() -> None:
    data: pd.DataFrame = pd.read_csv(FULL_CSV_REL_PATH)
    print(data.info())
    all_labels = set(data["label"])
    counts_labels = data["label"].value_counts(sort=True)


def explore_shirts_dataset() -> None:
    data: pd.DataFrame = pd.read_csv(SHIRTS_CSV_REL_PATH)
    print(data.info())
    all_types = set(data["Type"])
    counts_types = data["Type"].value_counts(sort=True)
    all_designs = set(data["Design"])
    counts_designs = data["Design"].value_counts(sort=True)


if __name__ == "__main__":
    explore_full_dataset()
    explore_shirts_dataset()
