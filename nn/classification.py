from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from dataset import Dataset
from nn.neural_net import NeuralNet
from nn.nn_functions import *
from visualization.visualizer import Visualizer


# CLASSIFICATION
def classification(train_path: str, test_path: str, model: NeuralNet, hidden_function: str,
                   epochs: int = 1000, learning_rate: float = 0.01, include_bias: bool = True,
                   plot_path: str = '../plots/predict_classification/example.jpg', savefig: bool = False,
                   display_information=None):
    # load data
    train_x, train_y, test_x, test_y = prepare_data_for_classification(train_path, test_path, include_bias=include_bias)

    # train neural network
    print("[INFO] training network...")
    model.train(train_x, train_y, epochs=epochs, learning_rate=learning_rate, include_bias=include_bias)

    # predict and evaluate network
    return predict_and_evaluate_classification(model, test_path, test_x, test_y, include_bias=include_bias,
                                               plot_path=plot_path, savefig=savefig,
                                               display_information=display_information)


def prepare_data_for_classification(train_path: str, test_path: str, include_bias: bool = True):
    # data loading and preparation
    data = Dataset(path=train_path)

    data = data.to_numpy()
    train_x = np.atleast_2d(data[:, :2])
    train_y = data[:, -1]

    test_data = Dataset(path=test_path)
    test_data = test_data.to_numpy()
    test_x = test_data[:, :2]
    test_y = test_data[:, -1]

    # convert classification from integers to vectors
    train_y = LabelBinarizer().fit_transform(train_y)
    if train_y.shape[1] == 1:
        train_y = np.hstack((1 - train_y, train_y))
    test_y = LabelBinarizer().fit_transform(test_y)
    if test_y.shape[1] == 1:
        test_y = np.hstack((1 - test_y, test_y))

    test_x = np.atleast_2d(test_x)

    if include_bias:
        test_x = np.c_[test_x, np.ones((test_x.shape[0]))]

    return train_x, train_y, test_x, test_y


def predict_and_evaluate_classification(model, test_path, test_x, test_y, include_bias: bool = True,
                                        plot_path: str = '../plots/predict_classification/example.jpg',
                                        savefig: bool = False, display_information=None):
    print("[INFO] evaluating network...")
    Visualizer.show_prediction(model,
                               Dataset(path=test_path),
                               savefig=savefig, path=plot_path,
                               include_bias=include_bias,
                               display_information=display_information)

    predictions = model.predict(test_x)
    print(f"loss: {model.loss(test_y, predictions)}")

    predictions = predictions.argmax(axis=1)
    print(classification_report(test_y.argmax(axis=1), predictions))

    return predictions


def check_classification_dimensions(path):
    data = Dataset(path=path)
    x_dim = np.atleast_2d(data.head(1).iloc[:, :-1]).shape[1]
    y_dim = data.iloc[:, -1].nunique()
    return x_dim, y_dim
