from dataset import Dataset
from nn.nn_functions import SIGMOID, TANH, SOFTMAX, CROSS_ENTROPY, HINGE, MSE, MAE
from tests.classification_mass_tests import classification_mass_tests
from tests.classification_tests import three_gauss_classification, simple_classification

if __name__ == '__main__':
    simple_classification(TANH, ll=SOFTMAX, lf=CROSS_ENTROPY)
    # three_gauss_classification(SIGMOID, ll=SOFTMAX, lf=CROSS_ENTROPY)
    # three_gauss_classification(TANH, ll=TANH, lf=HINGE)
    #
    # cube_regression(SIGMOID, lf=MSE)
    # cube_regression(TANH, lf=MSE)
    # activation_regression(SIGMOID, lf=MSE)
