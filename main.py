from dataset import Dataset
from nn.nn_functions import SIGMOID, TANH, SOFTMAX, CROSS_ENTROPY, HINGE, MSE, MAE
from tests.classification_mass_tests import classification_mass_tests
from tests.classification_tests import three_gauss_classification, simple_classification
from tests.regression_mass_tests import regression_mass_tests
from tests.regression_tests import activation_regression, cube_regression

if __name__ == '__main__':
    # activation_regression()

    simple_classification(TANH, ll=SOFTMAX, lf=CROSS_ENTROPY)
    simple_classification(TANH, ll=TANH, lf=HINGE)
    three_gauss_classification(TANH, ll=SOFTMAX, lf=CROSS_ENTROPY)
    three_gauss_classification(TANH, ll=TANH, lf=HINGE)

    # classification_mass_tests()
    # regression_mass_tests()

    cube_regression(SIGMOID, lf=MSE)
    cube_regression(SIGMOID, lf=MAE)
    activation_regression(SIGMOID, lf=MSE)
    activation_regression(SIGMOID, lf=MAE)
