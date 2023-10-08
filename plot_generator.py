import os

from dataset import Dataset
from visualizer import Visualizer


class PlotGenerator:
    @staticmethod
    def generate():
        for root, _, files in os.walk('datasets'):
            for file in files:
                path = os.path.join(root, file)
                data = Dataset(path=path)
                Visualizer.show_dataset(data, savefig=True)
                print(f'Generated: {os.path.join("plots", data.path)}')


# if __name__ == '__main__':
#     PlotGenerator.generate()
