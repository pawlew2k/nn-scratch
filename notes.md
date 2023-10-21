# Notes

## Elementy do zaimplementowania:

- [x] możliwość zainicjowania (powtarzalnego) procesu uczenia z zadanym ziarnem generatora liczb losowych
- [x] łatwa konfiguracja liczby warstw w sieci i neuronów w warstwie, obecności biasów (w dniu oddania będzie trzeba
  szybko dostosować architekturę sieci)
- [ ] wizualizacja zbioru uczącego i efektów klasyfikacji oraz regresji - (raczej matplotlib)
- [ ] wizualizacja błędu propagowanego w kolejnych iteracjach uczenia (na każdej z wag) (raczej matplotlib)
- [ ] wizualizacja wartości wag w kolejnych iteracjach uczenia (może networkx)

## Elementy do zbadania

- [ ] Wpływ funkcji aktywacji na skuteczność działania sieci – sprawdzić funkcję sigmoidalną i dwie inne, dowolne,
- [ ] funkcje aktywacji wewnątrz sieci. Uwaga: funkcja aktywacji na wyjściu musi być dobrana odpowiednio do rodzaju
  problemu.
- [ ] Wpływ liczby warstw ukrytych w sieci i ich liczności. Zbadać różne liczby warstw od 0 do 4, kilka różnych
  architektur
- [ ] Wpływ miary błędu na wyjściu sieci na skuteczność uczenia. Sprawdzić dwie miary błędu dla klasyfikacji i dwie dla
  regresji.

## Q&A

- Łatwa konfiguracja ... obecności biasów -> czy chodzi po prostu o obecność, czy w jakiś sposób
  konfigurację początkową biasów
    - Ma być zmienna, która ma ustalać czy mają być biasy czy nie
- wizualizacja błędu propagowanego w kolejnych iteracjach uczenia (na każdej z wag) - czy chodzi o
  MSE (jak wygląda los w kolejnych iteracjach), czy o coś innego?
- kilka różnych architektur -> czy chodzi jedynie o ilość warstw i liczebność perceptronów w
  warstwach
- drop rate - czy powinniśmy używać, aby zapobiegać overfittingowi?
- czy można używać gotowych metryk, czy trzeba je implementować samemu?

## Gradient descent

https://medium.com/@sami.benbrahim/gradient-descent-from-scratch-an-overview-of-gd-variants-f558da269a5f

- we have chosen stochastic (SGD) for 1st task

## Exploding gradient problem in regression

we have encountered a exploding gradient problem that resulted in NaN values in loss

-> it can be resolved by scaling input/output models to min-max normalized (for sigmoid) or z-score normalized (for
tanh)

## Working regression:

### ACTIVATION

net = NeuralNet([(1, ""), (4, SIGMOID), (1, LINEAR)], MSE, learning_rate=0.001)
net.train(x, sigmoid_normalized, epochs=2000, learning_rate=0.01)

### LINEAR

net = NeuralNet([(1, ""), (1, LINEAR)], MSE)

### CUBIC

model = NeuralNet([(1, ""), (16, SIGMOID), (16, SIGMOID), (1, LINEAR)], MSE)  # cubic

    # create neural network
    # model = NeuralNet([(train_x.shape[1], ""), (16, hidden_function), (train_y.shape[1], last_layer_function)],
    #                   loss_function, include_bias=include_bias)

