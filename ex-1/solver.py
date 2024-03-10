from autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import math
import time


class SolverOutput:
    def __init__(self, subsequent_values, beta, starting_point, time):
        """
        Initialize SolverOutput object.

        Args:
            subsequent_values (list): List of subsequent values obtained during solver execution.
            beta (float): The beta parameter used in the solver.
            starting_point (ndarray): The initial starting point for the solver.
        """
        self.subsequent_values = subsequent_values
        self.beta = beta
        self.starting_point = starting_point
        self.time = time

    def print_steps(self):
        """
        Print each step's details during solver execution.
        """
        for index, value in enumerate(self.subsequent_values):
            index_length = int(math.log10(len(self.subsequent_values))) + 1
            print(f"step {(index+1):{index_length}} q(xt)={value}")

    def draw_graph(self):
        """
        Draw a graph to visualize the subsequent values obtained during solver execution.
        """
        plt.plot(self.subsequent_values)
        plt.xlabel("x")
        plt.ylabel("q(x)")
        plt.title(f"Visualization of q(x) for alpha={1}")
        plt.show()

    def get_final_value(self):
        """
        Get the final value obtained after solver execution.

        Returns:
            float: The final value.
        """
        return self.subsequent_values[-1]

    def get_run_steps(self):
        """
        Get the number of steps/iterations taken during solver execution.

        Returns:
            int: The number of steps/iterations.
        """
        return len(self.subsequent_values)


def solver(objective_func, x0, beta=0.008, max_steps=200, eps=1e-6, threshold=10e12):
    """
    Solve the optimization problem using gradient descent.

    Args:
        objective_func (callable): The objective function to minimize.
        x0 (ndarray): The initial starting point for optimization.
        beta (float, optional): The learning rate (default is 0.008).
        max_steps (int, optional): The maximum number of optimization steps (default is 200).
        eps (float, optional): The convergence threshold (default is 1e-6).
        threshold (float, optional): The threshold value for objective function (default is 10e12).

    Returns:
        SolverOutput: An object containing information about solver execution.
    """

    if not callable(objective_func):
        raise ValueError("Objective function must be callable.")

    current_value = np.copy(x0)
    steps_counter = 0
    func_values = []
    start_time = time.time()
    while steps_counter < max_steps:
        grad_f = grad(objective_func)
        diff = beta * grad_f(current_value)
        current_value -= diff
        func_values.append(objective_func(current_value))

        if len(func_values) > 2:
            if (
                abs(func_values[-1] - func_values[-2]) < eps
                or func_values[-1] - func_values[-2] > threshold
            ):
                break

        steps_counter += 1

    stop_time = time.time()
    delta = (stop_time - start_time) * 1000

    return SolverOutput(func_values, beta, x0, delta)
