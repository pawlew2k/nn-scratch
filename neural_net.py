import numpy as np

from nn_functions import ACTIVATION_FUNCTION_DICT, ACTIVATION_FUNCTION_DERIVATIVE_DICT, LOSS_FUNCTION_DICT, \
    LOSS_FUNCTION_DERIVATIVE_DICT, WEIGHT_HEURISTICS, SIGMOID, MSE, SOFTMAX, CROSS_ENTROPY, RELU, TANH, LINEAR, MAE, \
    MSLE

class Layer:
    def __init__(self, in_size: int, out_size: int, activ_func: str, is_last=False):
        self.activ_func = ACTIVATION_FUNCTION_DICT.get(activ_func, ACTIVATION_FUNCTION_DICT[RELU])
        self.activ_func_deriv = ACTIVATION_FUNCTION_DERIVATIVE_DICT.get(activ_func,
                                                                        ACTIVATION_FUNCTION_DERIVATIVE_DICT[RELU])

        ## bias as last value in weights => [weights, bias]
        weight_heuristic = WEIGHT_HEURISTICS.get(activ_func, WEIGHT_HEURISTICS[RELU])(in_size, out_size)
        self.weights = np.random.randn(in_size + 1,
                                       out_size + (0 if is_last else 1)) * weight_heuristic

        # output after activation function
        self.outputs = np.zeros(out_size)

        # deltas for layer
        self.delta = np.array([0])

    def __str__(self):
        w = self.get_weights()
        return '\n'.join([f'W_{i}={w[i]}' for i in range(w.shape[0])])

    def get_weights(self):
        return self.weights


class NeuralNet:
    def __init__(self, layers: list[(int, str)], loss_func: str, seed: int = 42):
        self.learning_rate = 0.001
        self.loss = LOSS_FUNCTION_DICT.get(loss_func, LOSS_FUNCTION_DICT["MSE"])
        self.loss_deriv = LOSS_FUNCTION_DERIVATIVE_DICT.get(loss_func, LOSS_FUNCTION_DERIVATIVE_DICT["MSE"])

        # initialize Layers
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

    # train neural network on traning_data
    def train(self, training_data, target_values, epochs: int = 1000, learning_rate=0.001, dynamic_learning_rate=False,
              learning_rate_decrease=5000, display_update=10, gradient_normalization=False):

        self.learning_rate = learning_rate
        training_data = np.c_[training_data, np.ones((training_data.shape[0]))]

        for epoch in range(epochs):
            for (inputs, target) in zip(training_data, target_values):
                inputs = np.atleast_2d(inputs)
                target = np.atleast_2d(target)

                # Feed forward
                self.feed_forward(inputs.copy())

                # Back propagation
                self.backpropagation(inputs, target, gradient_normalization)

            # decrease learning rate when further down the calculations
            if dynamic_learning_rate:
                self.learning_rate /= (1 + epoch / learning_rate_decrease)

            self.show_update_mid_training(epoch, display_update, training_data, target_values)

    def feed_forward(self, out):
        for layer in self.layers:
            out = layer.activ_func(out.dot(layer.weights))
            layer.outputs = out

    def backpropagation(self, inputs, target, gradient_normalization):
        # updating last layer
        loss_function_derivative = self.loss_deriv(target, self.layers[-1].outputs)
        activation_function_derivative = self.layers[-1].activ_func_deriv(self.layers[-1].outputs)
        self.layers[-1].delta = loss_function_derivative * activation_function_derivative

        # weight change in last layer
        layer_input = self.layers[-2].outputs
        if len(self.layers) == 1:
            layer_input = inputs
        layer_input = np.atleast_2d(layer_input).T

        gradient = layer_input.dot(np.atleast_2d(self.layers[-1].delta))

        if gradient_normalization:
            normalization = np.linalg.norm(gradient, axis=0, keepdims=True)
            gradient /= normalization

        self.layers[-1].weights -= self.learning_rate * gradient

        # updating hidden layers
        for i in range(len(self.layers) - 2, -1, -1):
            weights_t = self.layers[i + 1].weights.T
            delta_prev_layer = self.layers[i + 1].delta
            self.layers[i].delta = delta_prev_layer.dot(weights_t) * self.layers[i].activ_func_deriv(
                self.layers[i].outputs)

            # getting inputs for layer
            layer_input = self.layers[i - 1].outputs
            if i == 0:
                layer_input = inputs
            layer_input = np.atleast_2d(layer_input).T

            gradient = layer_input.dot(np.atleast_2d(self.layers[i].delta))

            # normalize gradient
            if gradient_normalization:
                normalization = np.linalg.norm(gradient, axis=0, keepdims=True)
                gradient /= normalization

            # update weights
            self.layers[i].weights -= self.learning_rate * gradient

    # show update during training
    def show_update_mid_training(self, epoch, display_update, training_data, target_values):
        if epoch == 0 or (epoch + 1) % display_update == 0:
            targets = np.atleast_2d(target_values)
            predictions = self.predict(training_data)
            loss = self.loss(targets, predictions)
            print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    # predict an output based on input x
    def predict(self, x):
        prediction = np.atleast_2d(x)

        for layer in self.layers:
            prediction = layer.activ_func(np.dot(prediction, layer.weights))

        return prediction
