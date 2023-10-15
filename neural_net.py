# Elementy do zaimplementowania
# możliwość zainicjowania (powtarzalnego) procesu uczenia z zadanym ziarnem generatora liczb losowych
# łatwa konfiguracja liczby warstw w sieci i neuronów w warstwie, obecności biasów
# (w dniu oddania będzie trzeba szybko dostosować architekturę sieci)
# wizualizacja zbioru uczącego i efektów klasyfikacji oraz regresji - (raczej matplotlib)
# wizualizacja błędu propagowanego w kolejnych iteracjach uczenia (na każdej z wag) (raczej matplotlib)
# wizualizacja wartości wag w kolejnych iteracjach uczenia (może networkx)
import time

import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

from dataset import Dataset
from nn_functions import ACTIVATION_FUNCTION_DICT, ACTIVATION_FUNCTION_DERIVATIVE_DICT, LOSS_FUNCTION_DICT, \
    LOSS_FUNCTION_DERIVATIVE_DICT, WEIGHT_HEURISTICS, SIGMOID, MSE, SOFTMAX, CROSS_ENTROPY, RELU, TANH, LINEAR, MAE, \
    MSLE


# Elementy do zbadania
# Wpływ funkcji aktywacji na skuteczność działania sieci – sprawdzić funkcję sigmoidalną i dwie inne, dowolne,
# funkcje aktywacji wewnątrz sieci. Uwaga: funkcja aktywacji na wyjściu musi być dobrana odpowiednio do rodzaju problemu.
# Wpływ liczby warstw ukrytych w sieci i ich liczności. Zbadać różne liczby warstw od 0 do 4, kilka różnych architektur
# Wpływ miary błędu na wyjściu sieci na skuteczność uczenia. Sprawdzić dwie miary błędu dla klasyfikacji i dwie dla regresji.


class Layer:
    def __init__(self, in_size: int, out_size: int, activ_func: str, is_last=False):
        self.activ_func = ACTIVATION_FUNCTION_DICT.get(activ_func, ACTIVATION_FUNCTION_DICT["RELU"])
        self.activ_func_deriv = ACTIVATION_FUNCTION_DERIVATIVE_DICT.get(activ_func,
                                                                        ACTIVATION_FUNCTION_DERIVATIVE_DICT["RELU"])

        ## bias as last value in weights => [weights, bias]
        # weight_heuristic = WEIGHT_HEURISTICS.get(activ_func, WEIGHT_HEURISTICS["RELU"])
        self.weights = np.random.randn(in_size + 1,
                                       out_size + (0 if is_last else 1))  # * weight_heuristic(in_size, out_size)
        # self.biases = np.random.randn(1, out_size) #* weight_heuristic(in_size, out_size)
        self.weights /= np.sqrt(in_size)
        # self.biases /= np.sqrt(in_size)

        # ### TESTING DATA
        # if out_size == 2:
        #     weights = np.array([[1.0, 3.0], [2.0, 4.0]])
        #     biases = np.array([[5.0, 6.0]])
        # else:
        #     weights = np.array([[2.0], [1.0]])
        #     biases = np.array([[3.0]])
        #
        # self.weights = weights
        #
        # self.biases = biases
        #
        # ### END OF TESTING DATA

        # output after activation function
        self.outputs = np.zeros(out_size)

        # deltas for layer
        self.delta = np.array([0])

    def __str__(self):
        w = self.get_weights()
        # b = self.get_biases()
        return ""
        # return '\n'.join([f'W_{i}={w[i]}, b_{i}={b[0]}' for i in range(w.shape[0])])

    def get_weights(self):
        return self.weights

    # def get_biases(self):
    #     return self.biases


class NeuralNet:
    def __init__(self, layers: list[(int, str)], loss_func: str, rate: float = 0.001, seed: int = 42):
        self.learning_rate = rate
        self.loss = LOSS_FUNCTION_DICT.get(loss_func, LOSS_FUNCTION_DICT["MSE"])
        self.loss_deriv = LOSS_FUNCTION_DERIVATIVE_DICT.get(loss_func, LOSS_FUNCTION_DERIVATIVE_DICT["MSE"])
        # init Tensors
        np.random.seed(seed)
        self.layers: list[Layer] = []
        for i in range(1, len(layers)):
            self.layers.append(Layer(layers[i - 1][0], layers[i][0], layers[i][1], i == len(layers) - 1))

    def __str__(self):
        layers = []
        for i, layer in enumerate(self.layers):
            # print(type(layer))
            # print(layer.weights, 'xxxxxxxxx')
            layers.append(f'Layer_{i}:\n{str(layer)}\n')
            # print(layer, '-------')
        return ''.join(layers)

    def train(self, training_data, target_values, epochs: int = 1000):
        displayUpdate = 100
        training_data = np.c_[training_data, np.ones((training_data.shape[0]))]

        for epoch in range(epochs):
            for (inputs, target) in zip(training_data, target_values):
                inputs = np.atleast_2d(inputs)
                target = np.atleast_2d(target)
                out = inputs

                # Feed forward
                for layer in self.layers:
                    before_activation = out.dot(layer.weights)
                    # before_activation += layer.biases
                    out = layer.activ_func(before_activation)
                    layer.outputs = out

                # print(out)
                # Back propagation

                # last layer
                # loss = self.loss(target, out)
                # print(f"loss on epoch {epoch}: {loss}")

                loss_derivative = self.loss_deriv(target, out)
                deriv = self.layers[-1].activ_func_deriv(self.layers[-1].outputs)
                self.layers[-1].delta = loss_derivative * deriv

                # weight change
                layer_input = np.array(inputs)
                if len(self.layers) > 1:
                    layer_input = self.layers[-2].outputs
                layer_input = np.atleast_2d(layer_input).T
                d_ = np.atleast_2d(self.layers[-1].delta)
                dotted = layer_input.dot(d_)
                change = self.learning_rate * dotted
                new_weights = self.layers[-1].weights - change
                self.layers[-1].weights = new_weights

                for i in range(len(self.layers) - 2, -1, -1):
                    weights_t = self.layers[i + 1].weights.T
                    delta_prev_layer = self.layers[i + 1].delta
                    dot = delta_prev_layer.dot(weights_t)
                    deriv_ = self.layers[i].activ_func_deriv(self.layers[i].outputs)
                    self.layers[i].delta = dot * deriv_

                    layer_input = inputs
                    if i != 0:
                        layer_input = self.layers[i - 1].outputs

                    layer_input = np.atleast_2d(layer_input).T

                    self.layers[i].weights -= self.learning_rate * layer_input.dot(np.atleast_2d(self.layers[i].delta))
                    # bias_change = self.learning_rate * self.layers[i].delta
                    # new_biases = self.layers[i].biases - bias_change
                    # self.layers[i].biases = new_biases

            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(training_data, target_values)
                print("[INFO] epoch={}, loss={:.7f}".format(
                    epoch + 1, loss))

    def predict(self, x, bias=True):
        p = np.atleast_2d(x)

        if bias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in self.layers:
            p = layer.activ_func(np.dot(p, layer.weights))

        return p

    def calculate_loss(self, x, targets):
        # make predictions for the input data points then compute
        # the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(x, False)
        loss = self.loss(targets, predictions)
        # return the loss
        return loss


# if __name__ == '__main__':
#     # net = NeuralNet([(3, ""), (2, "LINEAR"), (1, "LINEAR")], "MSE")
#     # net.train(1, [[1.0, 2.0, 3.0]], [[1.0]])
#
#     ### TESTING DATA
#     # net = NeuralNet([(2, ""), (2, "LINEAR"), (1, "LINEAR")], "MSE")
#     # net = NeuralNet([(2, ""), (2, "LINEAR")], "MSE")
#
#     net = NeuralNet([(1, ""), (4, "SIGMOID"), (4, "SIGMOID"), (1, "SIGMOID")], "MSE")
#     # X = [[0, 0], [0, 1], [1, 0], [1, 1]]
#     # y = [[0], [1], [1], [0]]
#     # X = [[0, 1]]
#     # y = [[1]]
#
#     data = Dataset(path='datasets/projekt1/regression/data.cube.test.100.csv')
#     data = data.to_numpy()
#     # X = np.atleast_2d(data[:, 0]).T
#     # y = np.atleast_2d(data[:, 1]).T
#
#     X = [[-5.], [-4.99, ], [-4.98], [-4.97]]
#     y = [[-1253], [-1247.368296], [-1241.753168], [-1236.154592]]
#
#     net.train(10000, X, y)
#     print(net)

#### CLASSIFICATION

if __name__ == '__main__':
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
    nn = NeuralNet([(trainX.shape[1], ""), (32, SIGMOID), (16, SIGMOID), (10, SIGMOID)], MSLE, 0.01)
    print("[INFO] {}".format(nn))

    start = time.time()
    nn.train(trainX, trainY, 2000)
    end = time.time()
    print(f"elapsed time {end - start}")

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = nn.predict(testX)
    predictions = predictions.argmax(axis=1)
    print(classification_report(testY.argmax(axis=1), predictions))

#### REGRESION

# if __name__ == '__main__':
#     # np.seterr(invalid='raise')
#     # load the MNIST dataset and apply min/max scaling to scale the
#     # pixel intensity values to the range [0, 1] (each image is
#     # represented by an 8 x 8 = 64-dim feature vector)
#     print("[INFO] loading MNIST (sample) dataset...")
#     digits = datasets.load_digits()
#     data = digits.data.astype("float")
#     data = (data - data.min()) / (data.max() - data.min())
#     print("[INFO] samples: {}, dim: {}".format(data.shape[0],
#                                                data.shape[1]))
#
#     # construct the training and testing splits
#     (trainX, testX, trainY, testY) = train_test_split(data,
#                                                       digits.target, test_size=0.25)
#     # convert the labels from integers to vectors
#     # trainY = LabelBinarizer().fit_transform(trainY)
#     # testY = LabelBinarizer().fit_transform(testY)
#
#     # train the network
#     print("[INFO] training network...")
#     # nn = NeuralNet([trainX.shape[1], 32, 16, 10], )
#     nn = NeuralNet([(trainX.shape[1], ""), (32, RELU), (16, TANH), (1, LINEAR)], MSE, 0.1)
#     print("[INFO] {}".format(nn))
#
#     start = time.time()
#     nn.train(trainX, trainY, 2000)
#     end = time.time()
#     print(f"elapsed time {end - start}")
#
#     # evaluate the network
#     print("[INFO] evaluating network...")
#     predictions = nn.predict(testX)
#     # predictions = predictions.argmax(axis=1)
#     # print(classification_report(testY.argmax(axis=1), predictions))
#     print(classification_report(testY, predictions))
