from typing import Callable

import numpy as np

# ACTIVATION FUNCTION NAMES
RELU = "RELU"
SIGMOID = "SIGMOID"
TANH = "TANH"
LINEAR = "LINEAR"
BINARY_STEP = "BINARY_STEP"
SOFTMAX = "SOFTMAX"

# ERROR FUNCTION NAMES
MSE = "MSE"
MSLE = "MSLE"
MAE = "MAE"
CROSS_ENTROPY = "CROSS_ENTROPY"
NLL = "NLL"


def relu(arr: np.ndarray):
    return arr * (arr > 0)


def sigmoid(arr: np.ndarray):
    return 1 / (1 + np.exp(-arr))


def tanh(arr: np.ndarray):
    return (np.exp(arr) - np.exp(-arr)) / (np.exp(arr) + np.exp(-arr))


def binary_step(arr: np.ndarray):
    return 1. * (arr > 0)


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
    return 1. * (arr > 0)


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
    "LINEAR": lambda x: np.ones_like(x),
    "BINARY_STEP": lambda x: np.zeros_like(x),
    "SOFTMAX": lambda x: softmax_derivative(x)
}

LOSS_FUNCTION_DICT: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    # LOSS FUNCTIONS FOR REGRESSION
    # MEAN SQUARED ERROR
    "MSE": lambda target, actual: 0.5 * np.sum((target - actual) ** 2),
    # MEAN ABSOLUTE ERROR
    "MAE": lambda target, actual: np.sum(np.abs(actual - target)),
    # MEAN SQUARED LOGARITHMIC ERROR
    "MSLE": lambda target, actual: 0.5 * np.sum((np.log(target + 1) - np.log(actual + 1)) ** 2),
    # LOSS FUNCTIONS FOR CLASSIFICATION
    "CROSS_ENTROPY": lambda target, actual: -np.sum(target * np.log(actual)),
    # NEGATIVE LOG LIKELIHOOD
    "NLL": lambda target, actual: -np.sum(target * np.log(actual))
}


def cross_entropy_derivative(target, actual):
    result = -target / actual
    return result


def mae_derivative(target, actual):
    return np.where(actual - target > 0, 1, -1)


def msle_derivative(target, actual):
    return (np.log(actual) - np.log(target + 1)) / (actual + 1)


LOSS_FUNCTION_DERIVATIVE_DICT: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "MSE": lambda target, actual: actual - target,
    "MAE": lambda target, actual: mae_derivative(target, actual),
    "MSLE": lambda target, actual: msle_derivative(target, actual),
    "CROSS_ENTROPY": lambda target, actual: cross_entropy_derivative(target, actual)
}


def xavier_heuristic(in_size):
    return 1 / np.sqrt(in_size)


def xavier_normalized_heuristic(in_size, out_size):
    return np.sqrt(6 / (in_size + out_size))


def he_heuristic(in_size):
    return np.sqrt(2 / in_size)


WEIGHT_HEURISTICS: dict[str, Callable[[int, int], float]] = {
    "RELU": lambda in_size, out_size: he_heuristic(in_size),
    "SIGMOID": lambda in_size, out_size: xavier_heuristic(in_size),
    "TANH": lambda in_size, out_size: xavier_heuristic(in_size),
    "LINEAR": lambda in_size, out_size: he_heuristic(in_size),
    "BINARY_STEP": lambda in_size, out_size: xavier_heuristic(in_size),
    "SOFTMAX": lambda in_size, out_size: xavier_heuristic(in_size)
}
