# Neural Network from Scratch

This project presents a Python implementation of a neural network created entirely from scratch. It
focuses on understanding the fundamental building blocks of neural networks, including perceptron
logic, activation functions, loss functions, backpropagation, and various experiments with datasets
to observe the impact of different network configurations on performance.

## Features

- Implementation of a multilayer perceptron network.
- Customizable layers, activation functions, and loss functions.
- Backpropagation for network training.
- Experiments with activation functions' impact on network efficacy.
- Visualization of weights and gradients throughout training iterations.
- Dataset handling for both regression and classification tasks.
- Experiments with real-world datasets including MNIST.

## Installation

To get started with this project, clone the repository to your local machine:

```bash
git clone https://github.com/pawlew2k/nn-scratch
cd nn-scratch
```

Ensure you have Python 3.6 or later installed on your system. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The project is structured into several modules, each responsible for different aspects of the neural
network's operation:

- neural_net.py: The core logic of the neural network, including training and prediction.
- nn_functions.py: Activation and loss functions along with their derivatives.
- nn_serializer.py: Saving and loading network configurations.
- dataset.py: Handling data loading, preprocessing, and splitting.
- visualizer.py: Visualization tools for weights, gradients, and dataset distributions.
  To run the neural network with default settings, use:

```bash
python main.py
```

## Experimentation

The project includes scripts for conducting various experiments, such as observing the effects of
different activation functions or network architectures on performance. To run an experiment:

```bash
python tests/classification_tests.py
```

```bash
python tests/classification_oddanie_tests.py
```

```bash
python tests/regression_tests.py
```

```bash
python tests/regression_oddanie_tests.py
```

```bash
python tests/mnist_tests.py
```

## Datasets

This project works with synthetic and real-world datasets. For custom datasets, ensure they are
formatted correctly as described in the dataset module. Example datasets for regression and
classification tasks are included in the _datasets_ directory.