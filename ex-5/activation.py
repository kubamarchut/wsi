import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_dev(x):
    return np.where(x >= 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_dev(x):
    return sigmoid(x) * (1 - sigmoid(x))
