# data

Here is information on various datasets we used.  They are all available on [Kaggle](https://www.kaggle.com/).

Downloading the datasets can be made easy using the [`kaggle-api`](https://github.com/Kaggle/kaggle-api).

## Small Clothing Dataset

We utilized version 2 of [clothing dataset small][1], which holds a
train, validation, and test set.  There are 10 different classes available.

```bash
kaggle datasets download -p clothing_dataset_small --unzip abdelrahmansoltan98/clothing-dataset-small
```

The `.jpg` images in this dataset seem to always have a height or width of 400-px,
and sometimes both are 400-px.

## Full Clothing Dataset

We utilized version 1 of [Clothing dataset (full, high resolution)][2], which holds images, and pre-compressed images.  The pre-compressed images have their aspect ratio preserved with a 400-px width.

```bash
kaggle datasets download -p clothing_dataset_full --unzip agrigorev/clothing-dataset-full
```

## Shirts Dataset

We utilized version 2 of the [Clothing Dataset][3], which holds 2779 images of various men's shirts.
The labels can be found from a `data.csv` inside, or alternately from the sub-folder names themselves.
The image dimensions are not standardized, for example:

- Some shirts were 1200 x 1200.
- Some polos were 72 x 72.
- Some jackets were 931 x 1200.

Furthermore, there were many images that were not natively JPEG,
so we rolled out the `data/clean` module to remove them for training.

```bash
kaggle datasets download -p shirts_dataset --unzip gabrielalbertin/clothing-dataset
```

[1]: https://www.kaggle.com/datasets/abdelrahmansoltan98/clothing-dataset-small
[2]: https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full
[3]: https://www.kaggle.com/datasets/gabrielalbertin/clothing-dataset
