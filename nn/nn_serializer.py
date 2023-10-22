import os

from nn.neural_net import NeuralNet, Layer
import json


class NeuralNetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, NeuralNet):
            return {
                'net_structure': obj.net_structure,
                'learning_rate': obj.learning_rate,
                'loss_name': obj.loss_name,
                'layers': obj.layers,
            }
        elif isinstance(obj, Layer):
            return {
                'activ_func_name': obj.activ_func_name,
                'is_last': obj.is_last,
                'weights': obj.weights.tolist(),
                'outputs': obj.outputs.tolist(),
                'delta': obj.delta.tolist(),
                'gradient': obj.gradient.tolist()
            }
        return super().default(obj)


def serialize_model(model: NeuralNet, save_path: str):
    json_str = json.dumps(model, cls=NeuralNetEncoder)
    print(json_str)
    # Open a file for writing
    save_path = save_path.replace("/", "\\")
    path = os.path.abspath(os.curdir) + save_path
    with open(path, 'w') as f:
        # Use the NeuralNetEncoder to convert the neural network to a JSON string
        net_json = json.dumps(model, cls=NeuralNetEncoder)
        # Write the JSON string to the file
        f.write(net_json)
