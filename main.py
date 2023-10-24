from dataset import Dataset
from nn.nn_functions import SIGMOID, TANH, SOFTMAX, CROSS_ENTROPY, HINGE, MSE, MAE, RELU
from tests.classification_mass_tests import classification_mass_tests
from tests.classification_tests import three_gauss_classification, simple_classification
from tests.mnist_tests import mnist_classification
from tests.regression_tests import cube_regression, activation_regression

if __name__ == '__main__':
    # simple_classification(RELU, ll=SOFTMAX, lf=CROSS_ENTROPY)
    # simple_classification(SOFTMAX, ll=SOFTMAX, lf=CROSS_ENTROPY)
    # three_gauss_classification(SOFTMAX, ll=SOFTMAX, lf=CROSS_ENTROPY)
    # three_gauss_classification(RELU, ll=SOFTMAX, lf=CROSS_ENTROPY, lr=0.001, epochs=100) #learning_rate=0.001 #we need quite small loss for relu
    # three_gauss_classification(TANH, ll=TANH, lf=HINGE)
    # #
    # cube_regression(SIGMOID, lf=MSE)
    # cube_regression(TANH, lf=MSE)
    # activation_regression(SIGMOID, lf=MSE)

    mnist_classification()
