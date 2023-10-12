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
    return np.array([1 / (1 + np.exp(-x)) for x in arr])


def tanh(arr: np.ndarray):
    return np.array([(np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) for x in arr])


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
    return np.array([np.exp(-x) / (1 + np.exp(-x)) ** 2 for x in arr])


def tanh_derivative(arr: np.ndarray):
    return np.array([4 / (np.exp(x) + np.exp(-x)) ** 2 for x in arr])


def softmax_derivative(arr: np.ndarray):
    return np.array([softmax(x)(1 - softmax(x)) for x in arr])


ACTIVATION_FUNCTION_DERIVATIVE_DICT: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "RELU": lambda x: relu_derivative(x),
    "SIGMOID": lambda x: sigmoid_derivative(x),
    "TANH": lambda x: tanh_derivative(x),
    "LINEAR": lambda x: np.array([1.0 for _ in x]),
    "BINARY_STEP": lambda x: np.array([0.0 for _ in x]),
    "SOFTMAX": lambda x: softmax_derivative(x)
}

LOSS_FUNCTION_DICT: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    # MEAN SQUARED ERROR
    "MSE": lambda target, actual: np.sum([((t - a) ** 2) / 2 for t, a in zip(target, actual)]),
    # MEAN ABSOLUTE ERROR
    "MAE": lambda target, actual: np.sum([np.abs(t - a) for t, a in zip(target, actual)]),
    "CROSS_ENTROPY": lambda target, actual: -np.sum([t * np.log(a) for t, a in zip(target, actual)]),
    # NEGATIVE LOG LIKELIHOOD
    "NLL": lambda target, actual: -np.sum(
        [t * np.log(a) + (1 - t) * np.log(1 - a) for t, a in zip(target, actual)])
}

LOSS_FUNCTION_DERIVATIVE_DICT: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "MSE": lambda target, actual: target - actual,
    "MAE": lambda target, actual: np.ndarray([1 if a > t else -1 for t, a in zip(target, actual)]),
    "CROSS_ENTROPY": lambda target, actual: np.ndarray([-t / a for t, a in zip(target, actual)]),
    "NLL": lambda target, actual: np.ndarray([(a - t) / (a - a ** 2) for t, a in zip(target, actual)])
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
