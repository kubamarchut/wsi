import numpy as np
import matplotlib.pyplot as plt


def generate_input(minX, maxX, n):
    return np.random.uniform(minX, maxX, n)


class QFunc:
    """
    A class to generate and visualize the function q(x).
    """

    def __init__(self, alpha, n=10):
        """
        Initialize QFunc object.

        Args:
            alpha (float): The alpha parameter for the function.
            n (int): The number of dimensions in the function (default is 10).
        """
        self.alpha = alpha
        self.n = n

    def gen_input(self, min_x, max_x):
        """
        Generate random input matrix for the function.

        Args:
            min_x (float): The minimum value for input x.
            max_x (float): The maximum value for input x.

        Returns:
            ndarray: An array of 1xn input matrix with values between min_x and max_x.
        """
        return generate_input(min_x, max_x, self.n)

    def set_alpha(self, new_alpha):
        """
        Set a new value for the alpha parameter.

        Args:
            new_alpha (float): The new value for alpha.
        """
        self.alpha = new_alpha

    def get_alpha(self):
        """
        Get the current value of the alpha parameter.

        Returns:
            float: The value of alpha.
        """
        return self.alpha

    def get_func(self, x):
        """
        Calculate the value of the function q(x) for a given input x.

        Args:
            x (ndarray): An array of input values.

        Returns:
            float: The value of q(x).
        """
        indices = np.arange(1, self.n + 1)
        powers = self.alpha ** ((indices - 1) / (self.n - 1))

        return np.sum(powers * np.power(x, 2))

    def draw(self, points=10000):
        """
        Visualize the function q(x) for a range of input values.

        Args:
            points (int): The number of points to plot (default is 10000).
        """
        x_range = np.linspace(-100, 100, points)
        X = np.zeros((len(x_range), self.n))
        for i, x_val in enumerate(x_range):
            X[i] = x_val * np.ones(self.n)

        y = self.calculate(X)

        plt.scatter(X[:, 0], y)
        plt.colorbar(label="q(x)")
        plt.xlabel("x")
        plt.ylabel("q(x)")
        plt.title(f"Visualization of q(x) for alpha={self.alpha}")
        plt.show()

    def calculate(self, X):
        """
        Calculate the values of the function q(x) for a range of input values.

        Args:
            X (ndarray): An array of input values.

        Returns:
            ndarray: An array of corresponding function values.
        """
        return [self.get_func(x) for x in X]


if __name__ == "__main__":
    alphas = [1, 10, 100]

    for alpha in alphas:
        objectiveFunc = QFunc(alpha)
        objectiveFunc.draw()
