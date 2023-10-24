import time

import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from nn.neural_net import NeuralNet
from nn.nn_functions import TANH, LINEAR, MSE, SIGMOID, SOFTMAX, CROSS_ENTROPY, RELU
from nn.nn_serializer import serialize_model


#### MNIST CLASSIFICATION
def mnist_classification():
    digits = datasets.load_digits()
    data = digits.data.astype("float")
    data = (data - data.min()) / (data.max() - data.min())

    # construct the training and testing splits
    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      digits.target, test_size=0.25)
    # convert the labels from integers to vectors
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)

    # train the network
    print("[INFO] training network...")
    model = NeuralNet([(trainX.shape[1], ""), (32, SIGMOID), (16, SIGMOID), (10, SOFTMAX)], CROSS_ENTROPY)
    print("[INFO] {}".format(model))

    start = time.time()
    model.train(trainX, trainY, epochs=5000, learning_rate=0.005)
    end = time.time()
    print(f"elapsed time {end - start}")

    # save model
    serialize_model(model, '/models/mnist/mnist.json')

    # evaluate the network
    print("[INFO] evaluating network...")
    prepare_test_x = np.atleast_2d(testX)
    prepare_test_x = np.c_[prepare_test_x, np.ones((prepare_test_x.shape[0]))]

    predictions = model.predict(prepare_test_x)
    predictions = predictions.argmax(axis=1)
    print(classification_report(testY.argmax(axis=1), predictions, digits=4))