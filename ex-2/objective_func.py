import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from typing import Tuple
from utils import generate_input


class QFunc:
    """
    A class to generate and visualize the function q(x).
    """

    def __init__(self, min_x=-100, max_x=100, d=10, name=""):
        """
        Initialize QFunc object.

        Args:
            min_x (float, optional): The minimum value for input x (default is -100).
            max_x (float, optional): The maximum value for input x (default is 100).
            n (int): The number of dimensions in the function (default is 10).
        """
        self.d = d
        self.min_x = min_x
        self.max_x = max_x
        self.name = name

    def gen_input(self):
        """
        Generate random input matrix for the function.

        Returns:
            ndarray: An array of 1xn input matrix with values between min_x and max_x.
        """
        return generate_input(self.min_x, self.max_x, self.d)

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

    def draw(self, points=1000):
        """
        Visualize the function q(x) for a range of input values.

        Args:
            points (int): The number of points to plot (default is 1000).
        """
        if self.d == 2:
            X = np.linspace(self.min_x, self.max_x, points)
            Y = np.linspace(self.min_x, self.max_x, points)
            X, Y = np.meshgrid(X, Y)

            combined = np.vstack((X.ravel(), Y.ravel())).T

            y = self.calculate(combined)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            ax.plot_surface(X, Y, y.reshape(X.shape), cmap="viridis")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("q(x)")
            if self.name != "":
                plt.title(f"Wizualizacja funkcji {self.name}")
            else:
                plt.title(f"Visualization of q(X)")
            plt.show()
        else:
            print("I can draw only 3D functions")

    def calculate(self, X):
        """
        Calculate the values of the function q(x) for a range of input values.

        Args:
            X (ndarray): An array of input values.

        Returns:
            ndarray: An array of corresponding function values.
        """
        return np.array([self.get_func(x) for x in X])


class Rastrigin(QFunc):
    def __init__(self, min_x=-5.12, max_x=5.12, d=10):
        super().__init__(min_x, max_x, d, "Rastrigin")

    def get_func(self, x):
        """
        Calculate the value of the function q(x) for a given input x.

        Args:
            x (ndarray): An array of input values.

        Returns:
            float: The value of q(x).
        """
        value = 10 * self.d + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x))

        return value


class Gierwank(QFunc):
    def __init__(self, min_x=-50, max_x=50, d=10):
        super().__init__(min_x, max_x, d, "Griewank")

    def get_func(self, x):
        """
        Calculate the value of the function q(x) for a given input x.

        Args:
            x (ndarray): An array of input values.

        Returns:
            float: The value of q(x).
        """
        indices = np.arange(1, self.d + 1)
        value = (
            np.sum(np.power(x, 2) / 4000) - np.prod(np.cos(x / np.sqrt(indices))) + 1
        )

        return value


class DropWave(QFunc):
    def __init__(self, min_x=-5.12, max_x=5.12):
        super().__init__(min_x, max_x, 2, "Drop-Wave")

    def get_func(self, x):
        """
        Calculate the value of the function q(x) for a given input x.

        Args:
            x (ndarray): An array of input values.

        Returns:
            float: The value of q(x).
        """
        numerator = 1 + np.cos(12 * np.sqrt(np.power(x[0], 2) + np.power(x[1], 2)))
        denumerator = 0.5 * (np.power(x[0], 2) + np.power(x[1], 2)) + 2

        value = -1 * numerator / denumerator

        return value


if __name__ == "__main__":
    font = {"family": "normal", "weight": "bold", "size": 12}
    matplotlib.rc("font", **font)

    # Testing Rastrigin function implementation
    new_2d_rastrigin = Rastrigin(d=2)
    new_2d_rastrigin.draw(points=100)

    # Testing Gierwank function implementation
    new_2d_gierwank = Gierwank(d=2)
    new_2d_gierwank.draw(points=1500)

    # Testing Drop-Wave function implementation
    new_2d_dropwave = DropWave()
    new_2d_dropwave.draw(points=1500)
