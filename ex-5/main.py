from train import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    digits = load_digits()
    input, output = digits.data, digits.target

    output = np.eye(10)[output]

    input_train, input_test, output_train, output_test = train_test_split(
        input, output, test_size=0.3, random_state=10
    )

    num_class = 10

    layers = [input.shape[1], 15, num_class]
    train_model(input_train, output_train, layers)


if __name__ == "__main__":
    main()
