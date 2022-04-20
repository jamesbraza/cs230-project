import pandas as pd

FULL_CSV_FILE = "clothing_dataset_full/images.csv"
SHIRTS_CSV_FILE = "shirts_dataset/Dataset/data.csv"


def explore_full_dataset() -> None:
    data: pd.DataFrame = pd.read_csv(FULL_CSV_FILE)
    print(data.info())
    all_labels = set(data["label"])
    counts_labels = data["label"].value_counts(sort=True)


def explore_shirts_dataset() -> None:
    data: pd.DataFrame = pd.read_csv(SHIRTS_CSV_FILE)
    print(data.info())
    all_types = set(data["Type"])
    counts_types = data["Type"].value_counts(sort=True)
    all_designs = set(data["Design"])
    counts_designs = data["Design"].value_counts(sort=True)


if __name__ == "__main__":
    explore_full_dataset()
    explore_shirts_dataset()
