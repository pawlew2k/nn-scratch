from nn.nn_functions import RELU, SOFTMAX, CROSS_ENTROPY
from tests.classification_tests import simple_classification


def examples():
    simple_classification(RELU, ll=SOFTMAX, lf=CROSS_ENTROPY, lr=0.001, epochs=1000)
    # simple_classification(TANH, ll=SOFTMAX, lf=CROSS_ENTROPY, lr=0.0001, epochs=1000)
    # three_gauss_classification(SOFTMAX, ll=SOFTMAX, lf=CROSS_ENTROPY)
    # three_gauss_classification(RELU, ll=SOFTMAX, lf=CROSS_ENTROPY, lr=0.000005, epochs=2000) #learning_rate=0.001 #we need quite small loss for relu
    # three_gauss_classification(TANH, ll=TANH, lf=HINGE)
    # #
    # cube_regression(SIGMOID, lf=MSE)
    # cube_regression(TANH, lf=MAE, lr=0.01, epochs=300)
    # cube_regression(TANH, lf=MAE, lr=0.001, epochs=300)
    # activation_regression(SIGMOID, lf=MAE, lr=0.01, epochs=300)
    # activation_regression(SIGMOID, lf=MSE, lr=0.001, epochs=300)