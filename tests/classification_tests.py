from nn.classification import classification, check_classification_dimensions
from nn.neural_net import NeuralNet, TaskType
from nn.nn_functions import SIGMOID, LINEAR, MSE, RELU, SOFTMAX, CROSS_ENTROPY, HINGE, TANH
from nn.regression import regression

plots_default_path = '../plots/predict_classification'


def simple_classification(hf: str = SIGMOID, ll: str = SOFTMAX, lf: str = CROSS_ENTROPY, lr=0.01, epochs=300):
    train_path = 'datasets/projekt1/classification/data.simple.train.500.csv'
    test_path = 'datasets/projekt1/classification/data.simple.test.500.csv'
    hidden_function = hf
    last_layer_function = ll
    loss_function = lf
    include_bias = True

    plot_path = plots_default_path + '/simple/example.jpg'

    # create neural network
    model = NeuralNet([(2, ""), (4, hidden_function), (2, last_layer_function)], loss_function,
                      include_bias=include_bias, task_type=TaskType.CLASSIFICATION)
    classification(train_path, test_path, model, loss_function=loss_function, epochs=epochs, include_bias=include_bias,
                   plot_path=plot_path, display_information=f"Simple: {ll} + {lf}", learning_rate=lr)


def three_gauss_classification(hf: str = SIGMOID, ll: str = SOFTMAX, lf: str = CROSS_ENTROPY, lr=0.01, epochs=300):
    train_path = 'datasets/projekt1/classification/data.three_gauss.train.500.csv'
    test_path = 'datasets/projekt1/classification/data.three_gauss.test.500.csv'
    hidden_function = hf
    last_layer_function = ll
    loss_function = lf
    include_bias = True

    plot_path = plots_default_path + '/three_gauss/example.jpg'

    # create neural network
    model = NeuralNet([(2, ""), (16, hidden_function), (3, last_layer_function)], loss_function,
                      include_bias=include_bias, task_type=TaskType.CLASSIFICATION)
    classification(train_path, test_path, model, loss_function=loss_function, epochs=epochs, include_bias=include_bias,
                   plot_path=plot_path, display_information=f"Three gauss: {ll} + {lf}", learning_rate=lr)


if __name__ == '__main__':
    simple_classification()
    three_gauss_classification()