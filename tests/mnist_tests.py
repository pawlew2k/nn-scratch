import time

import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from nn.neural_net import NeuralNet
from nn.nn_deserializer import deserialize_model
from nn.nn_functions import TANH, LINEAR, MSE, SIGMOID, SOFTMAX, CROSS_ENTROPY, RELU
from nn.nn_serializer import serialize_model


#### MNIST CLASSIFICATION
def mnist_classification():
    digits = datasets.load_digits()
    data = digits.data.astype("float")
    data = (data - data.min()) / (data.max() - data.min())

    (train_x, test_x, train_y, test_y) = train_test_split(data,
                                                          digits.target, test_size=0.25, random_state=42)

    train_y = LabelBinarizer().fit_transform(train_y)
    test_y = LabelBinarizer().fit_transform(test_y)

    # train the network
    print("[INFO] training network...")
    model = NeuralNet([(train_x.shape[1], ""), (32, SIGMOID), (16, SIGMOID), (10, SOFTMAX)], CROSS_ENTROPY)

    model.train(train_x, train_y, epochs=2000, learning_rate=0.01)

    # save model
    serialize_model(model, '/models/mnist/mnist.json')

    # evaluate the network
    print("[INFO] evaluating network...")
    prepare_test_x = np.atleast_2d(test_x)
    prepare_test_x = np.c_[prepare_test_x, np.ones((prepare_test_x.shape[0]))]

    predictions = model.predict(prepare_test_x)
    predictions = predictions.argmax(axis=1)
    print(classification_report(test_y.argmax(axis=1), predictions, digits=4))


def mnist_decode_and_predict():
    # deserialize model
    model = deserialize_model('/models/mnist/mnist.json')

    digits = datasets.load_digits()
    data = digits.data.astype("float")
    data = (data - data.min()) / (data.max() - data.min())

    (train_x, test_x, train_y, test_y) = train_test_split(data,
                                                          digits.target, test_size=0.25, random_state=42)

    test_y = LabelBinarizer().fit_transform(test_y)


    # evaluate the network
    print("[INFO] evaluating network...")
    prepare_test_x = np.atleast_2d(test_x)
    prepare_test_x = np.c_[prepare_test_x, np.ones((prepare_test_x.shape[0]))]

    predictions = model.predict(prepare_test_x)
    predictions = predictions.argmax(axis=1)
    print(classification_report(test_y.argmax(axis=1), predictions, digits=4))