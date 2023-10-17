import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from dataset import Dataset
from neural_net import NeuralNet
from nn_functions import SIGMOID, LINEAR, MSE, min_max_normalize


def regression():
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

    # net = NeuralNet([(1, ""), (4, SIGMOID), (1, LINEAR)], MSE, 0.001) # activation
    # net = NeuralNet([(1, ""), (1, LINEAR)], MSE, 0.001) # linear

    # create neural network
    model = NeuralNet([(1, ""), (16, SIGMOID), (16, SIGMOID), (1, LINEAR)], MSE)  # cubic

    # train neural network
    print("[INFO] training network...")
    model.train(x, sigmoid_normalized, epochs=2000, learning_rate=0.01)

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


#### CLASSIFICATION

def classification():
    #
    data = Dataset(path='datasets/projekt1/classification/data.three_gauss.test.100.csv')
    data = data.to_numpy()
    train_x = np.atleast_2d(data[:, :2])
    train_y = data[:, -1]

    test_data = Dataset(path='datasets/projekt1/classification/data.three_gauss.test.100.csv')
    test_data = test_data.to_numpy()
    test_x = test_data[:, :2]
    test_y = test_data[:, -1]

    # convert classification from integers to vectors
    train_y = LabelBinarizer().fit_transform(train_y)
    test_y = LabelBinarizer().fit_transform(test_y)

    # create neural network
    model = NeuralNet([(train_x.shape[1], ""), (16, SIGMOID), (train_y.shape[1], SIGMOID)], MSE)

    # train neural network
    print("[INFO] training network...")
    model.train(train_x, train_y, epochs=1000, learning_rate=0.01)

    # evaluate network
    print("[INFO] evaluating network...")
    prepare_test_x = np.atleast_2d(test_x)
    prepare_test_x = np.c_[prepare_test_x, np.ones((prepare_test_x.shape[0]))]

    predictions = model.predict(prepare_test_x)

    predictions = predictions.argmax(axis=1)
    print(classification_report(test_y.argmax(axis=1), predictions))


if __name__ == '__main__':
    regression()
