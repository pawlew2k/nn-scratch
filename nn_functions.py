from typing import Callable

import numpy as np

# constant function names
RELU = "RELU"
SIGMOID = "SIGMOID"
TANH = "TANH"
LINEAR = "LINEAR"
BINARY_STEP = "BINARY_STEP"
SOFTMAX = "SOFTMAX"
MSE = "MSE"
MAE = "MAE"
CROSS_ENTROPY = "CROSS_ENTROPY"
NLL = "NLL"


def relu(arr: np.ndarray):
    return np.array([x if x > 0 else 0.0 for x in arr])


def sigmoid(arr: np.ndarray):
    return 1 / (1 + np.exp(-arr))


def tanh(arr: np.ndarray):
    return (np.exp(arr) - np.exp(-arr)) / (np.exp(arr) + np.exp(-arr))


def binary_step(arr: np.ndarray):
    return np.array([1 if x >= 0 else 0 for x in arr])


def softmax(arr: np.ndarray):
    exps = np.exp(arr)
    return exps / np.sum(exps)


ACTIVATION_FUNCTION_DICT: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "RELU": lambda x: relu(x),
    "SIGMOID": lambda x: sigmoid(x),
    "TANH": lambda x: tanh(x),
    "LINEAR": lambda x: x,
    "BINARY_STEP": lambda x: binary_step(x),
    "SOFTMAX": lambda x: softmax(x)
}


def relu_derivative(arr: np.ndarray):
    return np.array([1.0 if x > 0 else 0.0 for x in arr])


def sigmoid_derivative(arr: np.ndarray):
    return arr * (1 - arr)


def tanh_derivative(arr: np.ndarray):
    return 1 - arr ** 2


def softmax_derivative(arr: np.ndarray):
    return arr * (1 - arr)


# input to activation function derivative is the output of layer (sums that went through activation function)
ACTIVATION_FUNCTION_DERIVATIVE_DICT: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "RELU": lambda x: relu_derivative(x),
    "SIGMOID": lambda x: sigmoid_derivative(x),
    "TANH": lambda x: tanh_derivative(x),
    "LINEAR": lambda x: np.ones(len(x)),
    "BINARY_STEP": lambda x: np.zeros(len(x)),
    "SOFTMAX": lambda x: softmax_derivative(x)
}

LOSS_FUNCTION_DICT: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    # MEAN SQUARED ERROR
    "MSE": lambda actual, target: np.sum(((actual - target) ** 2) / (len(actual))),
    # MEAN ABSOLUTE ERROR
    "MAE": lambda actual, target: np.sum((np.abs(actual - target)) / (len(actual))),
    "CROSS_ENTROPY": lambda actual, target: -np.sum(actual * np.log2(target)),
    # NEGATIVE LOG LIKELIHOOD
    "NLL": lambda actual: 1 - np.max(np.log(actual))
}

LOSS_FUNCTION_DERIVATIVE_DICT: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "MSE": lambda target, actual: target - actual,
    "MAE": lambda target, actual: np.ndarray([1 if a > t else -1 for t, a in zip(target, actual)]),
    "CROSS_ENTROPY": lambda target, actual: np.ndarray([-t / a for t, a in zip(target, actual)]),
}


def xavier_normalized_heuristic(in_size, out_size):
    return np.sqrt(6 / (in_size + out_size + 2))


def he_heuristic(in_size):
    return np.sqrt(2 / (in_size + 1))


WEIGHT_HEURISTICS: dict[str, Callable[[int, int], float]] = {
    "RELU": lambda in_size, out_size: he_heuristic(in_size),
    "SIGMOID": lambda in_size, out_size: xavier_normalized_heuristic(in_size, out_size),
    "TANH": lambda in_size, out_size: xavier_normalized_heuristic(in_size, out_size),
    "LINEAR": lambda in_size, out_size: xavier_normalized_heuristic(in_size, out_size),
    "BINARY_STEP": lambda in_size, out_size: xavier_normalized_heuristic(in_size, out_size),
    "SOFTMAX": lambda in_size, out_size: xavier_normalized_heuristic(in_size, out_size)
}
