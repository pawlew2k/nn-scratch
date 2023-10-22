from dataset import Dataset
from nn.neural_net import NeuralNet
from nn.nn_functions import *
from nn.nn_serializer import serialize_model
from visualization.visualizer import Visualizer

default_plot_path = '../plots/predict_regression/example.jpg'


# REGRESSION
def regression(train_path: str, test_path: str, model: NeuralNet, hidden_function: str,
               epochs: int = 1000, learning_rate: float = 0.01, include_bias: bool = True,
               plot_path: str = default_plot_path, savefig: bool = False,
               display_information=None, model_path: str = '/models/predict_regression/example.json',
               save_model: bool = True):
    # load data
    train_x, train_y, test_x, test_y = prepare_data_for_regression(train_path, test_path, hidden_function, include_bias)

    # train neural network
    print("[INFO] training network...")
    model.train(train_x, train_y, epochs=epochs, learning_rate=learning_rate, include_bias=include_bias)

    # save model
    if save_model:
        serialize_model(model, model_path)

    # predict and evaluate network
    return predict_and_evaluate_regression(model, test_path, test_x, test_y, include_bias=include_bias,
                                           plot_path=plot_path, savefig=savefig,
                                           display_information=display_information)


def prepare_data_for_regression(train_path: str, test_path: str, hidden_function: str = SIGMOID,
                                include_bias: bool = True):
    # data loading and preparation
    data = Dataset(path=train_path)
    data = data.to_numpy()
    x = np.atleast_2d(data[:, 0]).T
    y = np.atleast_2d(data[:, 1]).T

    test_data = Dataset(path=test_path)
    test_data = test_data.to_numpy()
    test_x = np.atleast_2d(test_data[:, 0]).T
    test_y = np.atleast_2d(test_data[:, 1]).T

    test_x = np.atleast_2d(test_x)
    if include_bias:
        test_x = np.c_[test_x, np.ones((test_x.shape[0]))]

    # normalizing data
    y_min = y.min()
    y_max = y.max()

    if hidden_function == SIGMOID:
        y = min_max_normalize(y, y_min, y_max)
        test_y = min_max_normalize(test_y, y_min, y_max)
    elif hidden_function == TANH:
        y = min_max_normalize(y, y_min, y_max, (-1, 1))
        test_y = min_max_normalize(test_y, y_min, y_max, (-1, 1))

    return x, y, test_x, test_y


def predict_and_evaluate_regression(model, test_path, test_x, test_y, include_bias: bool = True,
                                    plot_path: str = default_plot_path, savefig: bool = False,
                                    display_information=None):
    print("[INFO] evaluating network...")
    Visualizer.show_prediction(model,
                               Dataset(path=test_path),
                               savefig=savefig, path=plot_path,
                               include_bias=include_bias, display_information=display_information)

    predictions = model.predict(test_x)
    print(f"loss: {model.loss(test_y, predictions)}")

    return predictions
