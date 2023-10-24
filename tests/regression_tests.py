import time

from nn.neural_net import NeuralNet, TaskType
from nn.nn_functions import SIGMOID, LINEAR, MSE, RELU, SOFTMAX, CROSS_ENTROPY, TANH, MAE
from nn.regression import regression

plots_default_path = '../plots/predict_regression'


def activation_regression(hf: str = SIGMOID, lf=MSE, lr=0.01, epochs=300):
    train_path = 'datasets/projekt1/regression/data.activation.train.500.csv'
    test_path = 'datasets/projekt1/regression/data.activation.test.500.csv'
    hidden_function = hf
    last_layer_function = LINEAR
    loss_function = lf
    include_bias = True

    plot_path = plots_default_path + '/activation/example.jpg'


    # create neural network
    model = NeuralNet([(1, ""), (8, hidden_function), (8, hidden_function), (1, last_layer_function)], loss_function,
                      include_bias=include_bias, task_type=TaskType.REGRESSION)
    regression(train_path, test_path, model, hidden_function, epochs=epochs, include_bias=include_bias,
               plot_path=plot_path, display_information=f"Activation: {lf}", learning_rate=lr)


def cube_regression(hf: str = SIGMOID, lf=MSE, lr=0.01, epochs=300):
    train_path = 'datasets/projekt1/regression/data.cube.train.500.csv'
    test_path = 'datasets/projekt1/regression/data.cube.test.500.csv'
    hidden_function = hf
    last_layer_function = LINEAR
    loss_function = lf
    include_bias = True

    plot_path = plots_default_path + '/cube/example.jpg'

    # create neural network
    model = NeuralNet([(1, ""), (32, hidden_function), (16, hidden_function), (1, last_layer_function)], loss_function,
                      include_bias=include_bias, task_type=TaskType.REGRESSION)
    regression(train_path, test_path, model, hidden_function=hidden_function, epochs=epochs, include_bias=include_bias,
               plot_path=plot_path, savefig=False, display_information=f"Cube: {lf}", learning_rate=lr)
