import numpy as np


def softmax(output: np.ndarray):
    exps = np.exp(output)
    return exps / np.sum(exps)


ACTIVATION_FUNCTION_DICT = {
    "RELU": lambda x: x if x > 0 else 0.0,
    "SIGMOID": lambda x: 1 / (1 + np.exp(-x)),
    "TANH": lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),
    "LINEAR": lambda x: x,
    "BINARY_STEP": lambda x: 1 if x >= 0 else 0,
    "SOFTMAX": lambda x: softmax(x)
}

def softmax_derivative(x, i, j):
    if i == j:
        return np.exp()

ACTIVATION_FUNCTION_DERIVATIVE_DICT = {
    "RELU": lambda x: x if 1 > 0 else 0.0,
    "SIGMOID": lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2,
    "TANH": lambda x: 4 / (np.exp(x) + np.exp(-x)) ** 2,
    "LINEAR": lambda x: 1,
    "BINARY_STEP": lambda x: 0,
    "SOFTMAX": lambda x, i, j:
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
    "CROSS_ENTROPY": lambda target, actual: np.sum([t * np.log(a) for t, a in zip(target, actual)])
}

CLASSIFICATION_LOSS_FUNCTION_DERIVATIVE_DICT = {
    "CROSS_ENTROPY": lambda target, actual: np.array([a - t] for t, a in zip(target, actual))
}
