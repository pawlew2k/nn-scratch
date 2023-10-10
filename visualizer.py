from utils.weights_visualizator import VisualizeNN as VisNN

import os

from matplotlib import pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame

from neural_net import NeuralNet


class Visualizer:
    @staticmethod
    def show_dataset(dataset: DataFrame, savefig: bool = False):
        plt.xlabel('x')
        plt.ylabel('y')
        if hasattr(dataset, 'name'):
            plt.title(f"Dataset: '{dataset.name}'")

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
            if hasattr(dataset, 'path'):
                path = f'plots/{dataset.path.rsplit(sep=".", maxsplit=1)[0]}.jpg'
                Visualizer.save_fig(path)
            else:
                raise Exception('Dataset has no attributes: path')
        else:
            plt.show()

    @staticmethod
    def show_net_weights(net: NeuralNet, savefig: bool = False, path: str = None):
        weights = [layer.get_weights().T for layer in net.layers]
        print(net.net_structure)
        print(weights)
        # Draw the Neural Network with weights
        network = VisNN.DrawNN(net.net_structure, weights)
        network.draw(savefig, path)

    @staticmethod
    def save_fig(path: str):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(fname=path, dpi=300)
        plt.clf()


# if __name__ == '__main__':
#     net = NeuralNet([3, 3, 2], "SIGMOID")
#     net.train(1, [[1.0, 2.0, 3.0]])
#     Visualizer.show_net_weights(net, savefig=False, path='plots/net_weights/example.jpg')
#     print(net)
#
#
#     from dataset import Dataset
#
#     data = Dataset(path='datasets/projekt1/regression/data.activation.train.1000.csv')
#     Visualizer.show_dataset(data)
#
#     data = Dataset(path='datasets/projekt1/classification/data.three_gauss.train.500.csv')
#     Visualizer.show_dataset(data, savefig=True)
