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
HINGE = "HINGE"

# EPSILON TO ADD TO LOG OR DIVIDE WHEN INVALID VALUE WOULD BE ENCOUNTERED
EPS = 1e-5


def raise_(ex):
    raise ex


LEAKY_RELU_CONST = 0.01


def relu(arr: np.ndarray):
    return np.where(arr > 0, arr, LEAKY_RELU_CONST * arr)


def sigmoid(arr: np.ndarray):
    return 1 / (1 + np.exp(-arr))


def tanh(arr: np.ndarray):
    return (np.exp(arr) - np.exp(-arr)) / (np.exp(arr) + np.exp(-arr))


def binary_step(arr: np.ndarray):
    return 1. * (arr > 0)


def softmax(arr: np.ndarray):
    shift_arr = arr - np.max(arr)  # numerically stable way
    exps = np.exp(shift_arr)
    return exps / np.sum(exps, axis=1, keepdims=True)


ACTIVATION_FUNCTION_DICT: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    RELU: lambda x: relu(x),
    SIGMOID: lambda x: sigmoid(x),
    TANH: lambda x: tanh(x),
    LINEAR: lambda x: x,
    BINARY_STEP: lambda x: binary_step(x),
    SOFTMAX: lambda x: softmax(x)
}


def relu_derivative(arr: np.ndarray):
    return np.where(arr > 0, 1, LEAKY_RELU_CONST)


def sigmoid_derivative(arr: np.ndarray):
    return arr * (1 - arr)


def tanh_derivative(arr: np.ndarray):
    return 1 - arr ** 2


def softmax_derivative(arr: np.ndarray):
    return arr * (1 - arr)


# input to activation function derivative is the output of layer (sums that went through activation function)
ACTIVATION_FUNCTION_DERIVATIVE_DICT: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    RELU: lambda x: relu_derivative(x),
    SIGMOID: lambda x: sigmoid_derivative(x),
    TANH: lambda x: tanh_derivative(x),
    LINEAR: lambda x: np.ones_like(x),
    BINARY_STEP: lambda x: np.zeros_like(x),
    SOFTMAX: lambda x: softmax_derivative(x)
}


def msle(target, actual):
    t = np.where(np.logical_and(- EPS < target, target < EPS), EPS, target)
    a = np.where(np.logical_and(- EPS < actual, actual < EPS), EPS, actual)
    return np.mean((np.log(t) - np.log(a)) ** 2)


def cross_entropy(target, actual):
    a = np.where(np.logical_and(- EPS < actual, actual < EPS), EPS, actual)
    return np.mean(-np.sum(target * np.log(a), axis=1))


LOSS_FUNCTION_DICT: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    # MEAN SQUARED ERROR
    MSE: lambda target, actual: np.mean((target - actual) ** 2),
    # MEAN ABSOLUTE ERROR
    MAE: lambda target, actual: np.mean(np.abs(actual - target)),
    # MEAN SQUARED LOGARITHMIC ERROR
    MSLE: lambda target, actual: msle(target, actual),
    # CROSS ENTROPY
    CROSS_ENTROPY: lambda target, actual: cross_entropy(target, actual),
    # HINGE LOSS
    HINGE: lambda target, actual: np.mean(np.mean(np.maximum(0, 1 - target * actual), axis=1))
}


def mse_derivative(target, actual):
    return 2 * (actual - target)


def mae_derivative(target, actual):
    return np.where(actual - target > 0, 1, -1)


def msle_derivative(target, actual):
    t = np.where(np.logical_and(- EPS < target, target < EPS), EPS, target)
    a = np.where(np.logical_and(- EPS < actual, actual < EPS), EPS, actual)
    return 2 * (np.log(a) - np.log(t)) / a


def delta_softmax_cross_entropy(target, actual):
    return actual - target


LOSS_FUNCTION_DERIVATIVE_DICT: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    MSE: lambda target, actual: mse_derivative(target, actual),
    MAE: lambda target, actual: mae_derivative(target, actual),
    MSLE: lambda target, actual: msle_derivative(target, actual),
    CROSS_ENTROPY: lambda target, actual: lambda x: raise_(Exception("PLAIN CROSS_ENTROPY DERIVATIVE NOT ALLOWED")),
    HINGE: lambda target, actual: np.where(actual * target < 1, -target / actual.size, 0)
}


def xavier_heuristic(in_size):
    return 1 / np.sqrt(in_size)


def xavier_normalized_heuristic(in_size, out_size):
    return np.sqrt(6 / (in_size + out_size))


def he_heuristic(in_size):
    return np.sqrt(2 / in_size)


WEIGHT_HEURISTICS: dict[str, Callable[[int, int], float]] = {
    RELU: lambda in_size, out_size: he_heuristic(in_size),
    SIGMOID: lambda in_size, out_size: xavier_heuristic(in_size),
    TANH: lambda in_size, out_size: xavier_heuristic(in_size),
    LINEAR: lambda in_size, out_size: he_heuristic(in_size),
    BINARY_STEP: lambda in_size, out_size: xavier_heuristic(in_size),
    SOFTMAX: lambda in_size, out_size: xavier_heuristic(in_size)
}


def reverse_min_max_normalize(normalized_data: np.ndarray, input_data: np.ndarray, feature_range=(0, 1)):
    data = normalized_data.copy()
    data = (data - feature_range[0]) / (feature_range[1] - feature_range[0])
    min_val = np.min(input_data)
    max_val = np.max(input_data)
    return data * (max_val - min_val) + min_val


def min_max_normalize(input_data: np.ndarray, min_val, max_val, feature_range=(0, 1)):
    data = input_data.copy()
    data_std = (data - min_val) / (max_val - min_val)
    data_scl = data_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    return data_scl
