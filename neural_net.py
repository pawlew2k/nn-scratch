# Elementy do zaimplementowania
# możliwość zainicjowania (powtarzalnego) procesu uczenia z zadanym ziarnem generatora liczb losowych
# łatwa konfiguracja liczby warstw w sieci i neuronów w warstwie, obecności biasów
# (w dniu oddania będzie trzeba szybko dostosować architekturę sieci)
# wizualizacja zbioru uczącego i efektów klasyfikacji oraz regresji - (raczej matplotlib)
# wizualizacja błędu propagowanego w kolejnych iteracjach uczenia (na każdej z wag) (raczej matplotlib)
# wizualizacja wartości wag w kolejnych iteracjach uczenia
from enum import Enum

import numpy as np


# Elementy do zbadania
# Wpływ funkcji aktywacji na skuteczność działania sieci – sprawdzić funkcję sigmoidalną i dwie inne, dowolne,
# funkcje aktywacji wewnątrz sieci. Uwaga: funkcja aktywacji na wyjściu musi być dobrana odpowiednio do rodzaju problemu.
# Wpływ liczby warstw ukrytych w sieci i ich liczności. Zbadać różne liczby warstw od 0 do 4, kilka różnych architektur
# Wpływ miary błędu na wyjściu sieci na skuteczność uczenia. Sprawdzić dwie miary błędu dla klasyfikacji i dwie dla regresji.

class ActivationFunction(Enum):
    RELU = (lambda x: np.max(0, x))
    SIGMOID = (lambda x: 1 / (1 + np.exp(-x)))
    TANH = (lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))
    BINARY_STEP = (lambda x: x)
    LINEAR = (lambda x: 1 if x >= 0 else 0)


class Tensor:
    def __init__(self, in_size: int, out_size: int):
        weight_heuristic = np.sqrt(2 / (in_size + out_size))
        # bias as last value in weights => [weights, bias]
        self.weights = np.random.randn(out_size, in_size + 1) * weight_heuristic
        # self.bias = np.random.random(out_size) * weight_heuristic


class NeuralNet:

    def __init__(self, layers: list[int], activ_func: ActivationFunction, seed: int = 42):
        # init Tensors
        np.random.seed(seed)
        self.layers: list[Tensor] = []
        for i in range(1, len(layers)):
            self.layers.append(Tensor(layers[i - 1], layers[i]))

    def train(self, epochs: int, training_data: list[list[float]]):
        for _ in range(epochs):
            for inputs in training_data:
                out = np.array(inputs)

                # Feed forward
                for layer in self.layers:
                    out = np.append(out, 1.0)
                    out = layer.weights.dot(out)

                print(out)
                # Back propagation
                # TODO: implement back propagation after 2nd lecture

    def predict(self, data: list[list[float]]):
        # predict the outcome
        pass


if __name__ == '__main__':
    net = NeuralNet([3, 3, 2], ActivationFunction.RELU)
    net.train(1, [[1.0, 2.0, 3.0]])
