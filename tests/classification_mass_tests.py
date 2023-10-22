from nn.classification import check_classification_dimensions, classification
from nn.neural_net import NeuralNet
from nn.nn_functions import TANH, SIGMOID, RELU, MSE, CROSS_ENTROPY, SOFTMAX

epochs = [10, 100, 300, 500, 1000, 2000]
learning_rates = [1, 0.5, 0.1, 0.05, 0.01, 0.005]

no_hidden_layers = [0, 1, 2, 3, 4]
no_node_in_layers = [1, 4, 8, 16, 32, 64, 128]
function_in_hidden_layers = [TANH, SIGMOID, RELU]
last_layer = [(SIGMOID, MSE), (SOFTMAX, CROSS_ENTROPY)]

no_of_train_data = [100, 500, 1000, 10000]

include_bias_arr = [False, True]

paths = {
    'simple': 'datasets/projekt1/classification/data.simple.',
    'three_gauss': 'datasets/projekt1/classification/data.three_gauss.',
    'circles': 'datasets/projekt1-oddanie/classification/data.circles.',
    'noisyXOR': 'datasets/projekt1-oddanie/classification/data.noisyXOR.',
    'XOR': 'datasets/projekt1-oddanie/classification/data.XOR.'
}

plots_default_path = '../plots/predict_classification'


def classification_mass_tests():
    for key in paths:
        for dataset in no_of_train_data:
            train_path = f'{paths[key]}train.{dataset}.csv'
            test_path = f'{paths[key]}train.{dataset}.csv'
            for hidden_layers in no_hidden_layers:
                for node_in_layer in no_node_in_layers:
                    for hidden_function in function_in_hidden_layers:
                        for ll in last_layer:
                            for no_epoch in epochs:
                                for learning_rate in learning_rates:
                                    for bias in include_bias_arr:
                                        plot_path = f'plots/predict_classification/{key}/dataset={dataset}_hl={hidden_layers}_nl={node_in_layer}_hf={hidden_function}_ll={ll}_epoch={no_epoch}_rate={learning_rate}_bias={bias}.jpg'
                                        x_dim, y_dim = check_classification_dimensions(train_path)
                                        neural_net_layers = [(x_dim, "")]
                                        for _ in range(hidden_layers):
                                            neural_net_layers.append((node_in_layer, hidden_function))
                                        neural_net_layers.append((y_dim, ll[0]))
                                        model = NeuralNet(neural_net_layers, ll[1], include_bias=bias)

                                        display_information = f"name={key}, dataset={dataset}, hl={hidden_layers}, nl={node_in_layer},\nhf={hidden_function}, ll={ll}, epoch={no_epoch}, rate={learning_rate}, bias={bias}"

                                        classification(train_path, test_path, model, hidden_function=hidden_function,
                                                       epochs=no_epoch, learning_rate=learning_rate, include_bias=bias,
                                                       plot_path=plot_path, savefig=True,
                                                       display_information=display_information)
