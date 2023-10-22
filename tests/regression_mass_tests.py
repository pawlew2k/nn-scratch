from nn.neural_net import NeuralNet
from nn.nn_functions import TANH, SIGMOID, MSE, LINEAR, MAE, MSLE
from nn.regression import regression

epochs = [10, 100, 300, 500, 1000, 2000]
learning_rates = [1, 0.5, 0.1, 0.05, 0.01, 0.005]

no_hidden_layers = [0, 1, 2, 3, 4]
no_node_in_layers = [1, 4, 8, 16, 32, 64, 128]
function_in_hidden_layers = [TANH, SIGMOID]
last_layer = [(LINEAR, MSE), (LINEAR, MAE), (LINEAR, MSLE)]

no_of_train_data = [100, 500, 1000, 10000]

include_bias_arr = [False, True]

paths = {
    'simple': 'datasets/projekt1/regression/data.activation.',
    'three_gauss': 'datasets/projekt1/regression/data.cube.',
    'circles': 'datasets/projekt1-oddanie/regression/data.linear.',
    'noisyXOR': 'datasets/projekt1-oddanie/regression/data.multimodal.',
    'XOR': 'datasets/projekt1-oddanie/regression/data.square.'
}

plots_default_path = '../plots/predict_regression'


def regression_mass_tests():
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
                                        plot_path = f'plots/predict_regression/{key}/dataset={dataset}_hl={hidden_layers}_nl={node_in_layer}_hf={hidden_function}_ll={ll}_epoch={no_epoch}_rate={learning_rate}_bias={bias}.jpg'
                                        neural_net_layers = [(1, "")]
                                        for _ in range(hidden_layers):
                                            neural_net_layers.append((node_in_layer, hidden_function))
                                        neural_net_layers.append((1, ll[0]))
                                        model = NeuralNet(neural_net_layers, ll[1], include_bias=bias)

                                        display_information = f"name={key}, dataset={dataset}, hl={hidden_layers}, nl={node_in_layer},\nhf={hidden_function}, ll={ll}, epoch={no_epoch}, rate={learning_rate}, bias={bias}"

                                        regression(train_path, test_path, model, hidden_function=hidden_function,
                                                   epochs=no_epoch, learning_rate=learning_rate, include_bias=bias,
                                                   plot_path=plot_path, savefig=True,
                                                   display_information=display_information)
