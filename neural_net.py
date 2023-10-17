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
from sklearn.metrics import classification_report, explained_variance_score
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
        weight_heuristic = WEIGHT_HEURISTICS.get(activ_func, WEIGHT_HEURISTICS["RELU"])(in_size, out_size)
        self.weights = np.random.randn(in_size + 1,
                                       out_size + (0 if is_last else 1))  # * weight_heuristic(in_size, out_size)

        ### V2 biases as another array
        # self.weights = np.random.randn(in_size,
        #                                out_size)
        # self.biases = np.random.randn(1, out_size) #* weight_heuristic(in_size, out_size)
        # self.biases /= np.sqrt(in_size)

        self.weights *= weight_heuristic

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
    def __init__(self, layers: list[(int, str)], loss_func: str, seed: int = 42, gradient_normalization=True):
        self.learning_rate = 0.001
        self.gradient_normalization = gradient_normalization
        self.loss = LOSS_FUNCTION_DICT.get(loss_func, LOSS_FUNCTION_DICT["MSE"])
        self.loss_deriv = LOSS_FUNCTION_DERIVATIVE_DICT.get(loss_func, LOSS_FUNCTION_DERIVATIVE_DICT["MSE"])
        # init Layers
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

    def train(self, training_data, target_values, epochs: int = 1000, learning_rate=0.001, dynamic_learning_rate=False,
              learning_rate_decrease_speed=5000, display_update=10):

        self.learning_rate = learning_rate
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
                gradient = layer_input.dot(d_)

                if self.gradient_normalization:
                    np.clip(gradient, -1, 1, gradient)
                    # normalization = np.linalg.norm(gradient, axis=0, keepdims=True)
                    # gradient /= normalization

                change = self.learning_rate * gradient
                new_weights = self.layers[-1].weights - change
                self.layers[-1].weights = new_weights

                # # bias change
                # bias_change = self.learning_rate * self.layers[-1].delta
                # new_biases = self.layers[-1].biases - bias_change
                # self.layers[-1].biases = new_biases

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

                    gradient = layer_input.dot(np.atleast_2d(self.layers[i].delta))

                    if self.gradient_normalization:
                        np.clip(gradient, -1, 1, gradient)
                        # normalization = np.linalg.norm(gradient, axis=0, keepdims=True)
                        # gradient /= normalization

                    weight_change = self.learning_rate * gradient
                    self.layers[i].weights -= weight_change

                    # # bias change
                    # bias_change = self.learning_rate * self.layers[i].delta
                    # new_biases = self.layers[i].biases - bias_change
                    # self.layers[i].biases = new_biases

            # should decrease learning rate when further down the calculations
            if dynamic_learning_rate:  # and self.learning_rate > 0.0004:
                self.learning_rate /= (1 + epoch / learning_rate_decrease_speed)

            if epoch == 0 or (epoch + 1) % display_update == 0:
                loss = self.calculate_loss(training_data, target_values)
                print("[INFO] epoch={}, loss={:.7f}, rate={:.7f}".format(
                    epoch + 1, loss, self.learning_rate))

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


def reverse_min_max_normalize(normalized_data: np.ndarray, input_data: np.ndarray):
    data = normalized_data.copy()
    min_val = np.min(input_data)
    max_val = np.max(input_data)
    return data * (max_val - min_val) + min_val


def min_max_normalize(input_data: np.ndarray, min_val, max_val):
    data = input_data.copy()
    return (data - min_val) / (max_val - min_val)


def z_score_normalize(input_data: np.ndarray, mean, std_dev):
    data = input_data.copy()
    standardized_data = (data - mean) / std_dev
    return standardized_data


def regression():
    # net = NeuralNet([(3, ""), (2, "LINEAR"), (1, "LINEAR")], "MSE")
    # net.train(1, [[1.0, 2.0, 3.0]], [[1.0]])

    ### TESTING DATA
    # net = NeuralNet([(1, ""), (4, SIGMOID), (1, LINEAR)], MSE, 0.001) # activation
    # net = NeuralNet([(1, ""), (1, LINEAR)], MSE, 0.001) # linear

    net = NeuralNet([(1, ""), (16, SIGMOID), (16, SIGMOID), (1, LINEAR)], MSE, gradient_normalization=False)

    data = Dataset(path='datasets/projekt1/regression/data.cube.train.10000.csv')
    data = data.to_numpy()
    x = np.atleast_2d(data[:, 0]).T
    y = np.atleast_2d(data[:, 1]).T
    normalized_y = min_max_normalize(y, y.min(), y.max())

    test_data = Dataset(path='datasets/projekt1/regression/data.cube.test.10000.csv')
    test_data = test_data.to_numpy()
    test_x = np.atleast_2d(test_data[:, 0]).T
    test_y = np.atleast_2d(test_data[:, 1]).T

    net.train(x, normalized_y, epochs=1000, learning_rate=0.05)
    # , dynamic_learning_rate=True)
    # , learning_rate_decrease_speed=20000, dynamic_learning_rate=True)
    print(net)

    predictions = net.predict(test_x)
    test_y_normalized = min_max_normalize(test_y, y.min(), y.max())
    print(f"score: {explained_variance_score(test_y_normalized, predictions)}")
    print(f"loss: {net.loss(test_y_normalized, predictions)}")


#### CLASSIFICATION

def classification():
    data = Dataset(path='datasets/projekt1/classification/data.three_gauss.test.100.csv')
    data = data.to_numpy()
    trainX = np.atleast_2d(data[:, :2])
    trainY = data[:, -1]
    test_data = Dataset(path='datasets/projekt1/classification/data.three_gauss.test.100.csv')
    test_data = test_data.to_numpy()
    testX = test_data[:, :2]
    testY = test_data[:, -1]

    # convert the labels from integers to vectors
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)

    # train the network
    print("[INFO] training network...")
    nn = NeuralNet([(trainX.shape[1], ""), (16, SIGMOID), (trainY.shape[1], SIGMOID)], MSE)
    print("[INFO] {}".format(nn))

    # start = time.time()
    nn.train(trainX, trainY, epochs=1000,
             learning_rate=0.01)  # , dynamic_learning_rate=True), learning_rate_decrease_speed=20000)
    # end = time.time()
    # print(f"elapsed time {end - start}")

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = nn.predict(testX)
    predictions = predictions.argmax(axis=1)
    print(classification_report(testY.argmax(axis=1), predictions))


if __name__ == '__main__':
    regression()
