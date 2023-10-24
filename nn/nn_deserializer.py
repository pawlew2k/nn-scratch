import json
import os

import numpy as np

from nn.neural_net import NeuralNet, Layer
from nn.nn_functions import LOSS_FUNCTION_DICT, LOSS_FUNCTION_DERIVATIVE_DICT


class LayerDecoder:
    @staticmethod
    def decode(layer_dict):
        layer = Layer(in_size=1, out_size=1,
                      activ_func=layer_dict['activ_func_name'], is_last=layer_dict['is_last'])
        layer.weights = np.array(layer_dict['weights'])
        layer.outputs = np.array(layer_dict['outputs'])
        layer.delta = np.array(layer_dict['delta'])
        layer.gradient = np.array(layer_dict['gradient'])
        return layer


class NeuralNetDecoder:
    @staticmethod
    def decode(net_dict):
        net = NeuralNet(layers=[(0, "")], loss_func=net_dict['loss_name'],
                        seed=42, include_bias=True)
        net.net_structure = net_dict['net_structure']
        net.learning_rate = net_dict['learning_rate']
        net.layers = [LayerDecoder.decode(layer_dict) for layer_dict in net_dict['layers']]
        net.task_type = net_dict['task_type']
        return net


def deserialize_model(model_path: str):
    model_path = model_path.replace("/", "\\")
    path = os.path.abspath(os.curdir) + model_path
    with open(path, 'r') as f:
        # Load the JSON string from the file
        net_json = f.read()
        # Use the NeuralNetDecoder to convert the JSON string to a neural network object
        net = NeuralNetDecoder.decode(json.loads(net_json))
        return net
