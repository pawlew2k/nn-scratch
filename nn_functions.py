from typing import Callable

import numpy as np


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

ACTIVATION_FUNCTION_DERIVATIVE_DICT = {
    "RELU": lambda x: 1.0 if 1 > 0 else 0.0,
    "SIGMOID": lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2,
    "TANH": lambda x: 4 / (np.exp(x) + np.exp(-x)) ** 2,
    "LINEAR": lambda x: 1.0,
    "BINARY_STEP": lambda x: 0.0,
    "SOFTMAX": lambda x:
}

REGRESSION_LOSS_FUNCTION_DICT = {
    # MEAN SQUARED ERROR
    "MSE": lambda target, actual: (target - actual) ** 2,
    # MEAN ABSOLUTE ERROR
    "MAE": lambda target, actual: np.abs(target - actual)
}

REGRESSION_LOSS_FUNCTION_DERIVATIVE_DICT = {
    "MSE": lambda target, actual: -2 * (target - actual),
    "MAE": lambda target, actual: 1 if actual > target else -1
}

CLASSIFICATION_LOSS_FUNCTION_DICT = {
    "CROSS_ENTROPY": lambda target, actual: -np.sum([t * np.log(a) for t, a in zip(target, actual)])
}

CLASSIFICATION_LOSS_FUNCTION_DERIVATIVE_DICT = {
    "CROSS_ENTROPY": lambda target, actual: np.array([a - t] for t, a in zip(target, actual))
}
