import os

import pandas as pd


class Dataset:
    def __new__(cls, path: str):
        dataset = pd.read_csv(path)
        if 'cls' in dataset.columns:
            dataset.task = 'classification'
        else:
            dataset.task = 'regression'
        dataset.dataset_name = os.path.basename(path)
        return dataset


# if __name__ == '__main__':
#     data = Dataset(path='datasets/projekt1/classification/data.simple.train.100.csv')
#     print(data.x)
#     print(data.y)
#     if data.task == 'classification':
#         print(data.cls)
#     print(data.x[4])
#     print(data.y[5])
#     if data.task == 'classification':
#         print(data.cls[6])
#     print(type(data))
#     print(data.data_name)
