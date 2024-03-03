import numpy as np
import matplotlib.pyplot as plt


def generate_input(minX, maxX, n):
    return np.random.uniform(minX, maxX, n)


class QFunc:
    """
    A class to generate and visualize the function q(x).
    """

    def __init__(self, alpha, min_x=-100, max_x=100, n=10):
        """
        Initialize QFunc object.

        Args:
            alpha (float): The alpha parameter for the function.
            min_x (float, optional): The minimum value for input x (default is -100).
            max_x (float, optional): The maximum value for input x (default is 100).
            n (int): The number of dimensions in the function (default is 10).
        """
        self.alpha = alpha
        self.n = n
        self.min_x = min_x
        self.max_x = max_x

    def gen_input(self):
        """
        Generate random input matrix for the function.

        Returns:
            ndarray: An array of 1xn input matrix with values between min_x and max_x.
        """
        return generate_input(self.min_x, self.max_x, self.n)

    def set_domain(self, min_x, max_x):
        """
        Set new domain values for the function.

        Args:
            min_x (float): The new minimum value for input x.
            max_x (float): The new maximum value for input x.
        """
        self.min_x = min_x
        self.max_x = max_x

    def get_domain(self):
        """
        Get the current domain values of the function.

        Returns:
            tuple: A tuple containing the minimum and maximum values of input x.
        """
        return self.min_x, self.max_x

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
        return np.array([self.get_func(x) for x in X])


if __name__ == "__main__":
    alphas = [1, 10, 100]

    for alpha in alphas:
        objectiveFunc = QFunc(alpha)
        objectiveFunc.draw()
