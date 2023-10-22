import time

import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from nn.neural_net import NeuralNet
from nn.nn_functions import TANH, LINEAR, MSE, SIGMOID


#### MNIST CLASSIFICATION
def mnist_classification():
    # np.seterr(invalid='raise')
    # load the MNIST dataset and apply min/max scaling to scale the
    # pixel intensity values to the range [0, 1] (each image is
    # represented by an 8 x 8 = 64-dim feature vector)
    print("[INFO] loading MNIST (sample) dataset...")
    digits = datasets.load_digits()
    data = digits.data.astype("float")
    data = (data - data.min()) / (data.max() - data.min())
    print("[INFO] samples: {}, dim: {}".format(data.shape[0],
                                               data.shape[1]))

    # construct the training and testing splits
    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      digits.target, test_size=0.25)
    # convert the labels from integers to vectors
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)

    # train the network
    print("[INFO] training network...")
    # nn = NeuralNet([trainX.shape[1], 32, 16, 10], )
    model = NeuralNet([(trainX.shape[1], ""), (32, SIGMOID), (16, SIGMOID), (10, SIGMOID)], MSE)
    print("[INFO] {}".format(model))

    start = time.time()
    model.train(trainX, trainY, epochs=2000, learning_rate=0.1)
    end = time.time()
    print(f"elapsed time {end - start}")

    # evaluate the network
    print("[INFO] evaluating network...")
    prepare_test_x = np.atleast_2d(testX)
    prepare_test_x = np.c_[prepare_test_x, np.ones((prepare_test_x.shape[0]))]

    predictions = model.predict(prepare_test_x)
    predictions = predictions.argmax(axis=1)
    print(classification_report(testY.argmax(axis=1), predictions))


#### REGRESION

def mnist_regression():
    # np.seterr(invalid='raise')
    # load the MNIST dataset and apply min/max scaling to scale the
    # pixel intensity values to the range [0, 1] (each image is
    # represented by an 8 x 8 = 64-dim feature vector)
    print("[INFO] loading MNIST (sample) dataset...")
    digits = datasets.load_digits()
    data = digits.data.astype("float")
    data = (data - data.min()) / (data.max() - data.min())
    print("[INFO] samples: {}, dim: {}".format(data.shape[0],
                                               data.shape[1]))

    # construct the training and testing splits
    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      digits.target, test_size=0.25)
    # convert the labels from integers to vectors
    # trainY = LabelBinarizer().fit_transform(trainY)
    # testY = LabelBinarizer().fit_transform(testY)

    # train the network
    print("[INFO] training network...")
    # nn = NeuralNet([trainX.shape[1], 32, 16, 10], )
    model = NeuralNet([(trainX.shape[1], ""), (32, TANH), (16, TANH), (1, LINEAR)], MSE)
    print("[INFO] {}".format(model))

    start = time.time()
    model.train(trainX, trainY, epochs=2000, learning_rate=0.1)
    end = time.time()
    print(f"elapsed time {end - start}")

    # evaluate the network
    print("[INFO] evaluating network...")

    prepare_test_x = np.atleast_2d(testX)
    prepare_test_x = np.c_[prepare_test_x, np.ones((prepare_test_x.shape[0]))]

    predictions = model.predict(prepare_test_x)
    # predictions = predictions.argmax(axis=1)
    # print(classification_report(testY.argmax(axis=1), predictions))
    print(classification_report(testY, predictions))
