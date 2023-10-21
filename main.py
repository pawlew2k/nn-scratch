import warnings

from classification import classification
from dataset import Dataset
from neural_net import NeuralNet
from nn_functions import SIGMOID, LINEAR, MSE, TANH, MSLE, MAE, RELU, SOFTMAX, CROSS_ENTROPY
from regression import regression


def activation_regression():
    train_path = 'datasets/projekt1/regression/data.activation.train.500.csv'
    test_path = 'datasets/projekt1/regression/data.activation.test.500.csv'
    hidden_function = SIGMOID
    last_layer_function = LINEAR
    loss_function = MSE
    include_bias = False

    # create neural network
    model = NeuralNet([(1, ""), (8, hidden_function), (1, last_layer_function)], loss_function,
                      include_bias=include_bias)
    regression(train_path, test_path, model, hidden_function, epochs=300, include_bias=include_bias)


def cube_regression():
    train_path = 'datasets/projekt1/regression/data.cube.train.500.csv'
    test_path = 'datasets/projekt1/regression/data.cube.test.500.csv'
    hidden_function = SIGMOID
    last_layer_function = LINEAR
    loss_function = MSE
    include_bias = True

    # create neural network
    model = NeuralNet([(1, ""), (32, hidden_function), (16, hidden_function), (1, last_layer_function)], loss_function,
                      include_bias=include_bias)
    regression(train_path, test_path, model, hidden_function=hidden_function, epochs=1000, include_bias=include_bias)


def linear_regression():
    train_path = 'datasets/projekt1-oddanie/regression/data.linear.train.500.csv'
    test_path = 'datasets/projekt1-oddanie/regression/data.linear.test.500.csv'
    hidden_function = SIGMOID
    last_layer_function = LINEAR
    loss_function = MSE
    include_bias = True

    # create neural network
    model = NeuralNet([(1, ""), (8, hidden_function), (1, last_layer_function)], loss_function,
                      include_bias=include_bias)
    regression(train_path, test_path, model, hidden_function=hidden_function, epochs=500, include_bias=include_bias)


def multimodal_regression():
    train_path = 'datasets/projekt1-oddanie/regression/data.multimodal.train.500.csv'
    test_path = 'datasets/projekt1-oddanie/regression/data.multimodal.test.500.csv'
    hidden_function = SIGMOID
    last_layer_function = LINEAR
    loss_function = MSE
    include_bias = True

    # create neural network
    model = NeuralNet([(1, ""), (128, hidden_function), (64, hidden_function), (32, hidden_function),
                       (16, hidden_function), (1, last_layer_function)], loss_function,
                      include_bias=include_bias)
    regression(train_path, test_path, model, hidden_function=hidden_function, epochs=1000, include_bias=include_bias)


def square_regression():
    train_path = 'datasets/projekt1-oddanie/regression/data.square.train.500.csv'
    test_path = 'datasets/projekt1-oddanie/regression/data.square.test.500.csv'
    hidden_function = SIGMOID
    last_layer_function = LINEAR
    loss_function = MSE
    include_bias = True

    # create neural network
    model = NeuralNet([(1, ""), (32, hidden_function),
                       (16, hidden_function), (1, last_layer_function)], loss_function,
                      include_bias=include_bias)
    regression(train_path, test_path, model, hidden_function=hidden_function, epochs=1000, include_bias=include_bias)


def simple_classification():
    train_path = 'datasets/projekt1/classification/data.simple.train.500.csv'
    test_path = 'datasets/projekt1/classification/data.simple.test.500.csv'
    hidden_function = RELU
    last_layer_function = SOFTMAX
    loss_function = CROSS_ENTROPY
    include_bias = True

    # create neural network
    model = NeuralNet([(2, ""), (4, hidden_function), (2, last_layer_function)], loss_function,
                      include_bias=include_bias)
    classification(train_path, test_path, model, hidden_function=hidden_function, epochs=10, include_bias=include_bias)


def three_gauss_classification():
    train_path = 'datasets/projekt1/classification/data.three_gauss.train.500.csv'
    test_path = 'datasets/projekt1/classification/data.three_gauss.test.500.csv'
    hidden_function = RELU
    last_layer_function = SOFTMAX
    loss_function = CROSS_ENTROPY
    include_bias = True

    # create neural network
    model = NeuralNet([(2, ""), (16, hidden_function), (3, last_layer_function)], loss_function,
                      include_bias=include_bias)
    classification(train_path, test_path, model, hidden_function=hidden_function, epochs=400, include_bias=include_bias)


if __name__ == '__main__':
    warnings.warn_explicit()
    # activation_regression()
    # linear_regression()
    # square_regression()
    # cube_regression()
    # multimodal_regression()

    # classification()
    simple_classification()
    # three_gauss_classification()

# examples
