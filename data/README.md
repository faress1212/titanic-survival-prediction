# Data

This project uses the classic Titanic passenger dataset, loaded automatically
via seaborn's built-in dataset loader (`seaborn.load_dataset("titanic")`).

No manual download is required — `src/data/load_data.py` fetches the data
directly at runtime.

In case you want to cache a local copy of the dataset (e.g. downloaded
from [Kaggle's Titanic competition](https://www.kaggle.com/c/titanic/data)).
If you place a `train.csv` here, you can adapt `load_data()` to read from
`data/raw/train.csv` instead of using seaborn.
