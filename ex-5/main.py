from train import *
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


def main():
    layers = [10, 5]
    digits = load_digits()

    train_model([], [], layers)


if __name__ == "__main__":
    main()
