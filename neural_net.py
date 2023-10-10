# Elementy do zaimplementowania
# możliwość zainicjowania (powtarzalnego) procesu uczenia z zadanym ziarnem generatora liczb losowych
# łatwa konfiguracja liczby warstw w sieci i neuronów w warstwie, obecności biasów
# (w dniu oddania będzie trzeba szybko dostosować architekturę sieci)
# wizualizacja zbioru uczącego i efektów klasyfikacji oraz regresji - (raczej matplotlib)
# wizualizacja błędu propagowanego w kolejnych iteracjach uczenia (na każdej z wag) (raczej matplotlib)
# wizualizacja wartości wag w kolejnych iteracjach uczenia (może networkx)

import numpy as np

from nn_functions import ACTIVATION_FUNCTION_DICT, ACTIVATION_FUNCTION_DERIVATIVE_DICT


# Elementy do zbadania
# Wpływ funkcji aktywacji na skuteczność działania sieci – sprawdzić funkcję sigmoidalną i dwie inne, dowolne,
# funkcje aktywacji wewnątrz sieci. Uwaga: funkcja aktywacji na wyjściu musi być dobrana odpowiednio do rodzaju problemu.
# Wpływ liczby warstw ukrytych w sieci i ich liczności. Zbadać różne liczby warstw od 0 do 4, kilka różnych architektur
# Wpływ miary błędu na wyjściu sieci na skuteczność uczenia. Sprawdzić dwie miary błędu dla klasyfikacji i dwie dla regresji.


class Layer:
    def __init__(self, in_size: int, out_size: int, activ_func: str):
        self.activ_func = ACTIVATION_FUNCTION_DICT.get(activ_func, ACTIVATION_FUNCTION_DICT["RELU"])
        self.activ_func_d = ACTIVATION_FUNCTION_DERIVATIVE_DICT.get(activ_func,
                                                                    ACTIVATION_FUNCTION_DERIVATIVE_DICT["RELU"])
        self.perceptron_values = np.zeros(out_size)
        weight_heuristic = np.sqrt(2 / (in_size + out_size))
        # bias as last value in weights => [weights, bias]
        self.weights = np.random.randn(out_size, in_size + 1) * weight_heuristic
        # self.bias = np.random.random(out_size) * weight_heuristic

    def __str__(self):
        w = self.weights
        b = self.get_biases()
        return '\n'.join([f'W_{i}={W[i]}, b_{i}={b[i]}' for i in range(w.shape[0])])


class NeuralNet:
    def __init__(self, layers: list[(int, str)], rate: float = 0.1, seed: int = 42):
        self.learning_rate = rate
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

    def train(self, epochs: int, training_data: list[list[float]]):
        for _ in range(epochs):
            for inputs in training_data:
                out = np.array(inputs)

                # Feed forward
                for layer in self.layers:
                    out = np.append(out, 1.0)
                    out = layer.activ_func(layer.weights.dot(out))
                    layer.perceptron_values = out

                # print(out)
                # Back propagation
                # TODO: implement back propagation after 2nd lecture

                #w = w - self.learning_rate * gradient(w)

    def predict(self, data: list[list[float]]):
        # predict the outcome
        pass


if __name__ == '__main__':
    net = NeuralNet([(3, ""), (3, "RELU"), (1, "RELU")])
    net.train(1, [[1.0, 2.0, 3.0]])
    # print(net)
