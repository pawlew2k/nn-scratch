from nn.neural_net import NeuralNet
from nn.nn_functions import SIGMOID, LINEAR, MSE, RELU, SOFTMAX, CROSS_ENTROPY
from nn.regression import regression


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


if __name__ == '__main__':
    linear_regression()
    multimodal_regression()
    square_regression()