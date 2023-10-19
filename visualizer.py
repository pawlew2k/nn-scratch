from dataset import Dataset
from nn_functions import *
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
        weights = [layer.get_weights() for layer in net.layers]
        # Draw the Neural Network with weights
        network = VisNN.DrawNN(net.net_structure, weights)
        network.draw(savefig, path)

    @staticmethod
    def show_gradients(net: NeuralNet, savefig: bool = False, path: str = None):
        # print(net.layers[0].gradient.shape)
        # print(net.layers[0].gradient[0:-1, 0:-1])
        gradients = [layer.get_gradient() * 100 for layer in net.layers]
        # Draw the Neural Network with weights
        print(gradients)
        network = VisNN.DrawNN(net.net_structure, gradients)
        network.draw(savefig, path)

    @staticmethod
    def save_fig(path: str):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(fname=path, dpi=300)
        plt.clf()


if __name__ == '__main__':
    net = NeuralNet([(1, ""), (4, SIGMOID), (4, SIGMOID), (1, SOFTMAX)], MSE)  # cubic

    # data loading and preparation
    data = Dataset(path='datasets/projekt1/regression/data.cube.train.1000.csv')
    data = data.to_numpy()
    x = np.atleast_2d(data[:, 0]).T
    y = np.atleast_2d(data[:, 1]).T
    sigmoid_normalized = min_max_normalize(y, y.min(), y.max())
    tanh_normalized = min_max_normalize(y, y.min(), y.max(), (-1, 1))

    test_data = Dataset(path='datasets/projekt1/regression/data.cube.test.1000.csv')
    test_data = test_data.to_numpy()
    test_x = np.atleast_2d(test_data[:, 0]).T
    test_y = np.atleast_2d(test_data[:, 1]).T

    # net = NeuralNet([(1, ""), (4, SIGMOID), (1, LINEAR)], MSE, learning_rate=0.001) # activation
    # net = NeuralNet([(1, ""), (1, LINEAR)], MSE)  # linear

    # create neural network
    model = NeuralNet([(1, ""), (4, SIGMOID), (4, SIGMOID), (1, LINEAR)], MSE)  # cubic

    # train neural network
    print("[INFO] training network...")
    for i in range(10):
        model.train(x, sigmoid_normalized, epochs=10, learning_rate=0.01)
        # print(net)
        # print(net.net_structure)
        Visualizer.show_gradients(model)
    # cubic
    # net.train(x, sigmoid_normalized, epochs=2000, learning_rate=0.01)

    # evaluate network
    print("[INFO] evaluating network...")
    prepare_test_x = np.atleast_2d(test_x)
    prepare_test_x = np.c_[prepare_test_x, np.ones((prepare_test_x.shape[0]))]

    predictions = model.predict(prepare_test_x)

    test_y_sigmoid_normalized = min_max_normalize(test_y, y.min(), y.max())
    test_y_tanh_normalized = min_max_normalize(test_y, y.min(), y.max(), (-1, 1))
    print(f"loss: {2 * model.loss(test_y_sigmoid_normalized, predictions) / len(test_y)}")
#
#
#     from dataset import Dataset
#
#     data = Dataset(path='datasets/projekt1/regression/data.activation.train.1000.csv')
#     Visualizer.show_dataset(data)
#
#     data = Dataset(path='datasets/projekt1/classification/data.three_gauss.train.500.csv')
#     Visualizer.show_dataset(data, savefig=True)
