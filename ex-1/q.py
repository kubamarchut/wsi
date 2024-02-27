import numpy as np
import matplotlib.pyplot as plt

x_range = np.linspace(-100, 100, 100)  # 100 points for each x_i
X = np.zeros((len(x_range), 10))  # Initialize X with zeros

for i, x_val in enumerate(x_range):
    X[i] = x_val * np.ones(10)  # Set each row of X to be a 1x10 matrix with x_val


def q(x, alpha=1):
    n = len(x)
    indices = np.arange(1, n + 1)
    powers = alpha ** ((indices - 1) / (n - 1))

    return np.sum(powers * np.power(x, 2))


def calc_q(X, alpha):
    return [q(x, alpha) for x in X]


if __name__ == "__main__":
    alphas = [1, 10, 100]

    for alpha in alphas:
        y = calc_q(X, alpha)

        plt.scatter(
            X[:, 0], y
        )  # Use scatter plot to visualize in 1D with color representing y
        plt.colorbar(label="q(x)")
        plt.xlabel("x")
        plt.ylabel("q(x)")
        plt.title(f"Visualization of q(x) for alpha={alpha}")
        plt.show()
