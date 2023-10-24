import time

from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelBinarizer

from dataset import Dataset
from nn.neural_net import NeuralNet
from nn.nn_functions import *
from nn.nn_serializer import serialize_model
from visualization.visualizer import Visualizer


# CLASSIFICATION
def classification(train_path: str, test_path: str, model: NeuralNet, loss_function: str = "None",
                   epochs: int = 1000, learning_rate: float = 0.01, include_bias: bool = True,
                   plot_path: str = '../plots/predict_classification/example.jpg', savefig: bool = False,
                   display_information=None, model_path: str = '/models/predict_classification/example.json',
                   save_model: bool = False):
    # load data
    train_x, train_y, test_x, test_y = prepare_data_for_classification(train_path, test_path, include_bias=include_bias,
                                                                       loss_function=loss_function)

    # train neural network
    print("[INFO] training network...")
    start = time.time()
    model.train(train_x, train_y, epochs=epochs, learning_rate=learning_rate, include_bias=include_bias, gradient_descent="mini-batch")
    end = time.time()
    print(f"time: {end - start}")

    # save model
    if save_model:
        serialize_model(model, model_path)

    # predict and evaluate network
    return predict_and_evaluate_classification(model, test_path, test_x, test_y, include_bias=include_bias,
                                               plot_path=plot_path, savefig=savefig,
                                               display_information=display_information)


def prepare_data_for_classification(train_path: str, test_path: str, include_bias: bool = True,
                                    loss_function: str = "None"):
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

    if loss_function == HINGE:
        test_y = np.where(test_y == 0, -1, 1)
        train_y = np.where(train_y == 0, -1, 1)

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
    loss = model.loss(test_y, predictions)
    print(f"loss: {loss}")

    predictions = predictions.argmax(axis=1)
    f1 = f1_score(test_y.argmax(axis=1), predictions, average='micro')
    print()

    Visualizer.show_metrics(model,
                            savefig=False, path=plot_path.replace('predict_classification', 'metrics_classification'),
                            display_information=display_information, loss=loss, f1=f1)

    return predictions


def check_classification_dimensions(path):
    data = Dataset(path=path)
    x_dim = np.atleast_2d(data.head(1).iloc[:, :-1]).shape[1]
    y_dim = data.iloc[:, -1].nunique()
    return x_dim, y_dim
