import json

import numpy as np

from nn.neural_net import NeuralNet, Layer


class LayerDecoder:
    @staticmethod
    def decode(layer_dict):
        layer = Layer(in_size=layer_dict['in_size'], out_size=layer_dict['out_size'],
                      activ_func=layer_dict['activ_func'], is_last=layer_dict['is_last'],
                      include_bias=layer_dict['include_bias'])
        layer.weights = np.array(layer_dict['weights'])
        layer.outputs = np.array(layer_dict['outputs'])
        layer.delta = np.array(layer_dict['delta'])
        layer.gradient = np.array(layer_dict['gradient'])
        return layer


class NeuralNetDecoder:
    @staticmethod
    def decode(net_dict):
        net = NeuralNet(layers=net_dict['layers'], loss_func=net_dict['loss_name'],
                        seed=42, include_bias=True)
        net.net_structure = net_dict['net_structure']
        net.learning_rate = net_dict['learning_rate']
        net.layers = [LayerDecoder.decode(layer_dict) for layer_dict in net_dict['layers']]
        return net


def deserialize_model(model_path: str):
    with open(model_path, 'r') as f:
        # Load the JSON string from the file
        net_json = f.read()
        # Use the NeuralNetDecoder to convert the JSON string to a neural network object
        net = json.loads(net_json, object_hook=NeuralNetDecoder.decode)
        return net
