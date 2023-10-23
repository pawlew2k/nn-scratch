from enum import Enum

from sklearn.metrics import f1_score

import nn.nn_functions
from nn.nn_functions import *


class TaskType(Enum):
    CLASSIFICATION = "CLASSIFICATION",
    REGRESSION = "REGRESSION"


class TrainingReport:
    def __init__(self, epoch, loss, f1=None, mse=None):
        self.epoch = epoch
        self.loss = loss
        self.f1 = f1
        self.mse = mse


class Layer:
    def __init__(self, in_size: int, out_size: int, activ_func: str, is_last: bool = False, include_bias: bool = True):
        self.activ_func_name = activ_func
        self.activ_func = ACTIVATION_FUNCTION_DICT[activ_func]
        self.activ_func_deriv = ACTIVATION_FUNCTION_DERIVATIVE_DICT[activ_func]
        self.is_last = is_last
        self.include_bias = include_bias

        bias_addition = 1 if include_bias else 0

        ## bias as last value in weights => [weights, bias]
        weight_heuristic = WEIGHT_HEURISTICS[activ_func](in_size, out_size)
        self.weights = np.random.randn(in_size + bias_addition,
                                       out_size + (0 if is_last else bias_addition)) * weight_heuristic

        # output after activation function
        self.outputs = np.zeros(out_size)

        # deltas for layer
        self.delta = np.array([0])
        self.gradient = np.array([0])

    def __str__(self):
        w = self.get_weights()
        return '\n'.join([f'W_{i}={w[i]}' for i in range(w.shape[0])])

    def get_weights(self):
        return self.weights[0:-1, 0:-1] if self.include_bias and not self.is_last else self.weights[0:-1, :]

    def get_gradient(self):
        return self.gradient[0:-1, 0:-1] if self.include_bias and not self.is_last else self.gradient[0:-1, :]


class NeuralNet:
    def __init__(self, layers: list[(int, str)], loss_func: str, seed: int = 42, include_bias: bool = True,
                 task_type: TaskType = None):
        self.net_structure = list(zip(*layers))[0]
        self.learning_rate = 0.001
        self.loss = LOSS_FUNCTION_DICT[loss_func]
        self.loss_deriv = LOSS_FUNCTION_DERIVATIVE_DICT[loss_func]
        self.loss_name = loss_func

        self.training_report: list[TrainingReport] = []
        self.task_type = task_type

        # initialize Layers
        np.random.seed(seed)
        self.layers: list[Layer] = []
        is_last_layer = False
        for i in range(1, len(layers)):
            if i == len(layers) - 1:
                is_last_layer = True
            self.layers.append(Layer(in_size=layers[i - 1][0], out_size=layers[i][0],
                                     activ_func=layers[i][1], is_last=is_last_layer, include_bias=include_bias))

    def __str__(self):
        layers = []
        for i, layer in enumerate(self.layers):
            layers.append(f'Layer_{i}:\n{str(layer)}\n')
        return ''.join(layers)

    # train neural network on training_data
    def train(self, training_data, target_values, epochs: int = 1000, learning_rate=0.001, dynamic_learning_rate=False,
              learning_rate_decrease=5000, display_update=10, gradient_normalization=False, include_bias: bool = True):

        self.learning_rate = learning_rate

        if include_bias:
            training_data = np.c_[training_data, np.ones((training_data.shape[0]))]

        for epoch in range(epochs):
            for inputs, target in zip(training_data, target_values):
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

    # feed forward step with input data x
    def feed_forward(self, x, update_layer_outputs=True):
        for layer in self.layers:
            x = layer.activ_func(x.dot(layer.weights))
            if update_layer_outputs:
                layer.outputs = x

        return x

    def backpropagation(self, inputs, target, gradient_normalization):
        # updating last layer
        if self.loss_name == CROSS_ENTROPY:
            self.layers[-1].delta = delta_softmax_cross_entropy(target, self.layers[-1].outputs)
        else:
            loss_function_derivative = self.loss_deriv(target, self.layers[-1].outputs)
            activation_function_derivative = self.layers[-1].activ_func_deriv(self.layers[-1].outputs)
            self.layers[-1].delta = loss_function_derivative * activation_function_derivative

        # weight change in last layer
        layer_input = inputs
        if len(self.layers) != 1:
            layer_input = self.layers[-2].outputs
        layer_input = np.atleast_2d(layer_input).T

        gradient = layer_input.dot(np.atleast_2d(self.layers[-1].delta))

        if gradient_normalization:
            normalization = np.linalg.norm(gradient, axis=0, keepdims=True)
            gradient /= normalization

        self.layers[-1].gradient = np.asarray(gradient)

        self.layers[-1].weights -= self.learning_rate * gradient

        # updating hidden layers
        for i in range(len(self.layers) - 2, -1, -1):
            weights_t = self.layers[i + 1].weights.T
            delta_prev_layer = self.layers[i + 1].delta
            self.layers[i].delta = delta_prev_layer.dot(weights_t) * self.layers[i].activ_func_deriv(
                self.layers[i].outputs)

            # getting inputs for layer
            layer_input = inputs
            if i != 0:
                layer_input = self.layers[i - 1].outputs
            layer_input = np.atleast_2d(layer_input).T

            gradient = layer_input.dot(np.atleast_2d(self.layers[i].delta))

            # normalize gradient
            if gradient_normalization:
                normalization = np.linalg.norm(gradient, axis=0, keepdims=True)
                gradient /= normalization

            self.layers[i].gradient = np.asarray(gradient)

            # update weights
            self.layers[i].weights -= self.learning_rate * gradient

    # show update during training
    def show_update_mid_training(self, epoch, display_update, training_data, target_values):
        targets = np.atleast_2d(target_values)
        predictions = self.predict(training_data, return_probs=True)
        loss = self.loss(targets, predictions)
        mse = nn.nn_functions.LOSS_FUNCTION_DICT[MSE](targets, predictions)

        report = TrainingReport(loss=loss, epoch=epoch, mse=mse)

        if self.task_type == TaskType.CLASSIFICATION:
            targets_classes = target_values.argmax(axis=1)
            predictions_classes = predictions.argmax(axis=1)
            report.f1 = f1_score(targets_classes, predictions_classes, average='macro')

        self.training_report.append(report)

        if epoch == 0 or (epoch + 1) % display_update == 0:
            print(f'[INFO] epoch={epoch + 1}, loss={loss:.7f}')

    # predict an output based on input x
    def predict(self, x, task: str = None, return_probs=False):
        prediction = np.atleast_2d(x)
        output = self.feed_forward(prediction, update_layer_outputs=False)
        if task == 'classification' and not return_probs:
            return np.argmax(np.asarray(output), axis=1) + 1
        return output
