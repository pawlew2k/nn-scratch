# Elementy do zaimplementowania
# możliwość zainicjowania (powtarzalnego) procesu uczenia z zadanym ziarnem generatora liczb losowych
# łatwa konfiguracja liczby warstw w sieci i neuronów w warstwie, obecności biasów
# (w dniu oddania będzie trzeba szybko dostosować architekturę sieci)
# wizualizacja zbioru uczącego i efektów klasyfikacji oraz regresji - (raczej matplotlib)
# wizualizacja błędu propagowanego w kolejnych iteracjach uczenia (na każdej z wag) (raczej matplotlib)
# wizualizacja wartości wag w kolejnych iteracjach uczenia (może networkx)

import numpy as np

from nn_functions import ACTIVATION_FUNCTION_DICT, ACTIVATION_FUNCTION_DERIVATIVE_DICT, LOSS_FUNCTION_DICT, \
    LOSS_FUNCTION_DERIVATIVE_DICT, WEIGHT_HEURISTICS


# Elementy do zbadania
# Wpływ funkcji aktywacji na skuteczność działania sieci – sprawdzić funkcję sigmoidalną i dwie inne, dowolne,
# funkcje aktywacji wewnątrz sieci. Uwaga: funkcja aktywacji na wyjściu musi być dobrana odpowiednio do rodzaju problemu.
# Wpływ liczby warstw ukrytych w sieci i ich liczności. Zbadać różne liczby warstw od 0 do 4, kilka różnych architektur
# Wpływ miary błędu na wyjściu sieci na skuteczność uczenia. Sprawdzić dwie miary błędu dla klasyfikacji i dwie dla regresji.


class Layer:
    def __init__(self, in_size: int, out_size: int, activ_func: str):
        self.activ_func = ACTIVATION_FUNCTION_DICT.get(activ_func, ACTIVATION_FUNCTION_DICT["RELU"])
        self.activ_func_deriv = ACTIVATION_FUNCTION_DERIVATIVE_DICT.get(activ_func,
                                                                        ACTIVATION_FUNCTION_DERIVATIVE_DICT["RELU"])

        ## bias as last value in weights => [weights, bias]
        weight_heuristic = WEIGHT_HEURISTICS.get(activ_func, WEIGHT_HEURISTICS["RELU"])
        self.weights = np.random.randn(in_size + 1, out_size) * weight_heuristic(in_size, out_size)

        ### TESTING DATA
        # if out_size == 2:
        #     weights = np.array([[1.0, 3.0], [2.0, 4.0], [5.0, 6.0]])
        # else:
        #     weights = np.array([[2.0], [1.0], [3.0]])
        #
        # self.weights = weights

        # output before activation function
        self.before_activation = np.zeros(in_size)
        # output after activation function
        self.outputs = np.zeros(out_size)

        # deltas for layer
        self.delta = np.array([0])

    def __str__(self):
        w = self.get_weights()
        b = self.get_biases()
        return '\n'.join([f'W_{i}={w[i]}, b_{i}={b[i]}' for i in range(w.shape[0])])

    def get_weights(self):
        return self.weights[:-1, :]

    def get_biases(self):
        return self.weights[-1, :]


class NeuralNet:
    def __init__(self, layers: list[(int, str)], loss_func: str, rate: float = 0.1, seed: int = 42):
        self.learning_rate = rate
        self.loss = LOSS_FUNCTION_DICT.get(loss_func, LOSS_FUNCTION_DICT["MSE"])
        self.loss_deriv = LOSS_FUNCTION_DERIVATIVE_DICT.get(loss_func, LOSS_FUNCTION_DERIVATIVE_DICT["MSE"])
        # init Tensors
        np.random.seed(seed)
        self.layers: list[Layer] = []
        for i in range(1, len(layers)):
            self.layers.append(Layer(layers[i - 1][0], layers[i][0], layers[i][1]))

    def __str__(self):
        layers = []
        for i, layer in enumerate(self.layers):
            # print(type(layer))
            # print(layer.weights, 'xxxxxxxxx')
            layers.append(f'Layer_{i}:\n{str(layer)}\n')
            # print(layer, '-------')
        return ''.join(layers)

    def train(self, epochs: int, training_data: list[list[float]], target_values: list[list[float]]):
        for epoch in range(epochs):
            for (inputs, target) in zip(training_data, target_values):
                out = np.array(inputs)

                # Feed forward
                for layer in self.layers:
                    out = np.append(out, 1.0)
                    layer.before_activation = out.dot(layer.weights)
                    out = layer.activ_func(layer.before_activation)
                    layer.outputs = out

                # print(out)
                # Back propagation

                # last layer
                loss = self.loss(np.array(target), out)
                print(f"loss on epoch {epoch}: {loss}")

                loss_derivative = self.loss_deriv(np.array(target), out)
                deriv = self.layers[-1].activ_func_deriv(self.layers[-1].before_activation)
                self.layers[-1].delta = loss_derivative * deriv
                outputs_T = self.layers[-1].outputs.T
                dotted = outputs_T.dot(self.layers[-1].delta)
                change = self.learning_rate * dotted
                self.layers[-1].weights -= change

                for i in range(len(self.layers) - 2, -1, -1):
                    weights_t = self.layers[i].weights.T
                    delta_prev_layer = self.layers[i + 1].delta
                    dot = delta_prev_layer.dot(weights_t)
                    self.layers[i].delta = dot * self.layers[i].activ_func_deriv(self.layers[i].before_activation) # TODO: tutaj cos nie dziala nadal

                    self.layers[i].weights -= self.learning_rate * self.layers[i].outputs.T.dot(self.layers[i].delta)

    def predict(self, data: list[list[float]]):
        # predict the outcome
        pass


if __name__ == '__main__':
    net = NeuralNet([(3, ""), (2, "LINEAR"), (1, "LINEAR")], "MSE")
    net.train(1, [[1.0, 2.0, 3.0]], [[1.0]])

    ### TESTING DATA
    # net = NeuralNet([(2, ""), (2, "LINEAR"), (1, "LINEAR")], "MSE")
    # net.train(1, [[10, 20]], [[1.0]])
    print(net)
