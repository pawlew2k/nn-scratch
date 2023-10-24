from dataset import Dataset
from nn.nn_functions import SIGMOID, TANH
from tests.classification_mass_tests import classification_mass_tests
from tests.classification_tests import three_gauss_classification, simple_classification
from tests.regression_mass_tests import regression_mass_tests
from tests.regression_tests import activation_regression, cube_regression

if __name__ == '__main__':
    # activation_regression()

    simple_classification(SIGMOID)
    simple_classification(TANH)
    three_gauss_classification(SIGMOID)
    three_gauss_classification(TANH)

    # classification_mass_tests()
    # regression_mass_tests()

    # cube_regression(SIGMOID)
    # cube_regression(TANH)
    # activation_regression(SIGMOID)
    # activation_regression(TANH)
