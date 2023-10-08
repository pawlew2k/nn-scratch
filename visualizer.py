from matplotlib import pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame


class Visualizer:
    @staticmethod
    def show_dataset(dataset: DataFrame, savefig=False):
        plt.xlabel('x')
        plt.ylabel('y')
        try:
            dataset_name = dataset.dataset_name
            plt.title(f"Dataset: '{dataset_name}'")
        except AttributeError:
            dataset_name = 'no_title'

        if dataset.task == 'regression':
            x = dataset.x
            y = dataset.y
            plt.scatter(x, y, marker='.', s=1, edgecolors='red')
            plt.legend(['observation'])
        elif dataset.task == 'classification':
            for label in pd.unique(dataset.cls):
                x = dataset.loc[dataset['cls'] == label].x
                y = dataset.loc[dataset['cls'] == label].y
                plt.scatter(x, y, marker='.', s=20, label=label)
            plt.legend(title='labels:')
        else:
            raise Exception('Task in dataset undefined')

        if savefig:
            path = f'plots/{dataset_name.rsplit(sep=".", maxsplit=1)[0]}.jpg'
            plt.savefig(fname=path, dpi=300)
            plt.clf()
        else:
            plt.show()


# if __name__ == '__main__':
#     from dataset import Dataset
#
#     data = Dataset(path='datasets/projekt1/regression/data.activation.train.1000.csv')
#     Visualizer.show_dataset(data)
#
#     data = Dataset(path='datasets/projekt1/classification/data.three_gauss.train.500.csv')
#     Visualizer.show_dataset(data, savefig=True)
