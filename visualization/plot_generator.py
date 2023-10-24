import os

from dataset import Dataset
from visualization.visualizer import Visualizer


class PlotGenerator:
    @staticmethod
    def generate():
        for root, _, files in os.walk('../datasets'):
            for file in files:
                path = os.path.join(root, file)
                print(path)
                data = Dataset(path=path)
                Visualizer.show_dataset(data, savefig=True)
                print(f'Generated: {os.path.join(os.path.basename("plots"), data.name)}')


if __name__ == '__main__':
    PlotGenerator.generate()
