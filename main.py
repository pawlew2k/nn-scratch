from dataset import Dataset
from tests.classification_mass_tests import classification_mass_tests
from tests.classification_tests import three_gauss_classification, simple_classification
from tests.regression_mass_tests import regression_mass_tests
from tests.regression_tests import activation_regression

if __name__ == '__main__':
    # activation_regression()

    # simple_classification()
    # three_gauss_classification()

    classification_mass_tests()
    regression_mass_tests()
