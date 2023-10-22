from dataset import Dataset
from nn.nn_functions import *
from utils.weights_visualizator import VisualizeNN as VisNN

import os

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from pandas.core.frame import DataFrame

from nn.neural_net import NeuralNet, TaskType, TrainingReport


def report_mapper(report: TrainingReport):
    return report.epoch, report.loss, report.mse, report.precision, report.recall, report.f1


class Visualizer:
    @staticmethod
    def show_dataset(dataset: DataFrame, savefig: bool = False, hold_plot=False):
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
        elif not hold_plot:
            plt.show()

    @staticmethod
    def show_net_weights(net: NeuralNet, savefig: bool = False, path: str = None):
        weights = [layer.get_weights() for layer in net.layers]
        network = VisNN.DrawNN(net.net_structure, weights)
        network.draw(savefig, path)

    @staticmethod
    def show_gradients(net: NeuralNet, savefig: bool = False, path: str = None):
        gradients = [layer.get_gradient() for layer in net.layers]
        network = VisNN.DrawNN(net.net_structure, gradients)
        network.draw(savefig, path)

    @staticmethod
    def show_prediction(net: NeuralNet, dataset: DataFrame, savefig: bool = False, path: str = None,
                        include_bias: bool = True, display_information=None):
        Visualizer.show_dataset(dataset, hold_plot=True)
        if hasattr(dataset, 'name'):
            if display_information is not None:
                plt.title(f"Prediction dataset: '{display_information}'")
            else:
                plt.title(f"Prediction dataset: '{dataset.name}'")
        if dataset.task == 'regression':
            x = dataset.x
            prepared_x = [np.atleast_2d(datum) for datum in x]
            if include_bias:
                prepared_x = [np.c_[datum, np.ones((datum.shape[0]))] for datum in prepared_x]
            # print(prepared_x)
            y = [net.predict(datum) for datum in prepared_x]
            y_normalized = reverse_min_max_normalize(np.asarray(y), dataset.y)
            plt.scatter(x, y_normalized, marker='.', s=1, edgecolors='green')
            plt.legend(['observation', 'prediction'])

        elif dataset.task == 'classification':
            colors = list(mcolors.TABLEAU_COLORS.values())
            for label in pd.unique(dataset.cls):
                x = dataset.loc[dataset['cls'] == label].x
                y = dataset.loc[dataset['cls'] == label].y
                x_y = np.atleast_2d(list(zip(x, y)))
                if include_bias:
                    x_y = np.c_[x_y, np.ones((x_y.shape[0]))]
                predicted_labels = net.predict(x_y, task='classification')
                print(predicted_labels)
                for x_datum, y_datum, pred_label in zip(x, y, predicted_labels):
                    if pred_label != label:
                        plt.scatter(x_datum, y_datum, marker='.', s=20, color=colors[pred_label - 1])
                        plt.scatter(x_datum, y_datum, marker='x', s=40, color='red', linewidths=0.6)
        else:
            raise Exception('Task in dataset undefined')

        if savefig:
            if path:
                Visualizer.save_fig(path)
            else:
                raise Exception('Path needed to save')
        else:
            plt.show()

    @staticmethod
    def show_metrics(net: NeuralNet, savefig: bool = False, path: str = None, display_information=None):
        if display_information is not None:
            plt.title(f"Prediction dataset: '{display_information}'")

        epochs = []
        loss = []
        mse = []
        precision = []
        recall = []
        f1 = []

        for report in net.training_report:
            epochs.append(report.epoch)
            loss.append(report.loss)
            mse.append(report.mse)
            precision.append(report.precision)
            recall.append(report.recall)
            f1.append(report.f1)

        plt.plot(epochs, loss)

        if net.task_type == TaskType.REGRESSION:
            plt.plot(epochs, mse)
        elif net.task_type == TaskType.CLASSIFICATION:
            plt.plot(epochs, f1)

        else:
            raise Exception('Task in model is undefined')

        if savefig:
            if path:
                Visualizer.save_fig(path)
            else:
                raise Exception('Path needed to save')
        else:
            plt.show()

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
    data = Dataset(path='../datasets/projekt1/regression/data.cube.train.1000.csv')
    data = data.to_numpy()
    x = np.atleast_2d(data[:, 0]).T
    y = np.atleast_2d(data[:, 1]).T
    sigmoid_normalized = min_max_normalize(y, y.min(), y.max())
    tanh_normalized = min_max_normalize(y, y.min(), y.max(), (-1, 1))

    test_data = Dataset(path='../datasets/projekt1/regression/data.cube.test.1000.csv')
    test_data = test_data.to_numpy()
    test_x = np.atleast_2d(test_data[:, 0]).T
    test_y = np.atleast_2d(test_data[:, 1]).T

    # net = NeuralNet([(1, ""), (4, SIGMOID), (1, LINEAR)], MSE, learning_rate=0.001) # activation
    # net = NeuralNet([(1, ""), (1, LINEAR)], MSE)  # linear

    # create neural network
    model = NeuralNet([(1, ""), (4, SIGMOID), (4, SIGMOID), (1, LINEAR)], MSE)  # cubic

    # train neural network
    print("[INFO] training network...")
    # for i in range(10):
    #     model.train(x, sigmoid_normalized, epochs=10, learning_rate=0.01)
    # print(net)
    # print(net.net_structure)
    # Visualizer.show_gradients(model)
    model.train(x, sigmoid_normalized, epochs=100, learning_rate=0.01)
    test_data = Dataset(path='../datasets/projekt1/regression/data.cube.test.1000.csv')
    Visualizer.show_prediction(model, test_data, savefig=True, path='../plots/predict_regression/example.jpg')
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
