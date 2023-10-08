import os

import pandas as pd
from pandas.core.frame import DataFrame


class Dataset:
    def __new__(cls, path: str) -> DataFrame:
        dataset = pd.read_csv(path)
        if 'cls' in dataset.columns:
            dataset.task = 'classification'
        else:
            dataset.task = 'regression'
        dataset.name = os.path.basename(path)
        dataset.path = os.path.relpath(path, 'datasets')
        return dataset


# if __name__ == '__main__':
#     data = Dataset(path='datasets/projekt1/classification/data.simple.train.100.csv')
#     print(data.name)
#     print(data.path)
#     print(data.y)
#     if data.task == 'classification':
#         print(data.cls)
#     print(data.x[4])
#     print(data.y[5])
#     if data.task == 'classification':
#         print(data.cls[6])
#     print(type(data))
