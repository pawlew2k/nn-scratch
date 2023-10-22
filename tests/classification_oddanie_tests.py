from nn.classification import classification, check_classification_dimensions
from nn.neural_net import NeuralNet
from nn.nn_functions import SIGMOID, LINEAR, MSE, RELU, SOFTMAX, CROSS_ENTROPY
from nn.regression import regression

def circles_classification():
    train_path = 'datasets/projekt1-oddanie/classification/data.circles.train.500.csv'
    test_path = 'datasets/projekt1-oddanie/classification/data.circles.test.500.csv'
    hidden_function = SIGMOID
    last_layer_function = SOFTMAX
    loss_function = CROSS_ENTROPY
    include_bias = True

    x_dim, y_dim = check_classification_dimensions(train_path)

    # create neural network
    model = NeuralNet(
        [(x_dim, ""), (16, hidden_function), (8, hidden_function), (y_dim, last_layer_function)], loss_function, include_bias=include_bias)
    classification(train_path, test_path, model, hidden_function=hidden_function, epochs=500,
                   include_bias=include_bias)


def noisy_xor_classification():
    train_path = 'datasets/projekt1-oddanie/classification/data.noisyXOR.train.500.csv'
    test_path = 'datasets/projekt1-oddanie/classification/data.noisyXOR.test.500.csv'
    hidden_function = RELU
    last_layer_function = SOFTMAX
    loss_function = CROSS_ENTROPY
    include_bias = True

    x_dim, y_dim = check_classification_dimensions(train_path)

    # create neural network
    model = NeuralNet([(x_dim, ""), (16, hidden_function), (y_dim, last_layer_function)], loss_function,
                      include_bias=include_bias)
    classification(train_path, test_path, model, hidden_function=hidden_function, epochs=400, include_bias=include_bias)


def xor_classification():
    train_path = 'datasets/projekt1-oddanie/classification/data.XOR.train.500.csv'
    test_path = 'datasets/projekt1-oddanie/classification/data.XOR.test.500.csv'
    hidden_function = RELU
    last_layer_function = SOFTMAX
    loss_function = CROSS_ENTROPY
    include_bias = True

    x_dim, y_dim = check_classification_dimensions(train_path)

    # create neural network
    model = NeuralNet([(x_dim, ""), (16, hidden_function), (y_dim, last_layer_function)], loss_function,
                      include_bias=include_bias)
    classification(train_path, test_path, model, hidden_function=hidden_function, epochs=400, include_bias=include_bias)