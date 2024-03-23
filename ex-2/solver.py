import numpy as np
import matplotlib.pyplot as plt
import math
import time
import copy

from evolution import *


def print_steps(SolverOutput):
    """
    Print each step's details during solver execution.
    """
    for index, value in enumerate(SolverOutput.subsequent_values):
        index_length = int(math.log10(len(SolverOutput.subsequent_values))) + 1
        print(f"step {(index+1):{index_length}} q(xt)={value}")


def draw_graph(SolverOutput):
    """
    Draw a graph to visualize the subsequent values obtained during solver execution.
    """
    plt.plot(SolverOutput.subsequent_values)
    plt.xlabel("x")
    plt.ylabel("q(x)")
    plt.title(f"Visualization of q(x) for alpha={1}")
    plt.show()


def get_final_value(SolverOutput):
    """
    Get the final value obtained after solver execution.

    Returns:
        float: The final value.
    """
    return SolverOutput.subsequent_values[-1]


def get_run_steps(SolverOutput):
    """
    Get the number of steps/iterations taken during solver execution.

    Returns:
        int: The number of steps/iterations.
    """
    return len(SolverOutput.subsequent_values)


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


def solver(
    objective_func,
    population_size=100,
    max_steps=200,
    eps=1e-6,
    threshold=10e12,
):
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

    if not callable(objective_func.get_func):
        raise ValueError("Objective function must be callable.")

    # generate population
    population = Population(
        population_size,
        objective_func.d,
        min_x=objective_func.min_x,
        max_x=objective_func.max_x,
    )
    population.initialize()
    # fitness
    population.fitness(objective_func.get_func)

    initial_population = np.copy(population.individuals)
    best = sorted(initial_population, key=lambda x: x.fitness_score)[0]

    steps_counter = 0
    start_time = time.time()
    while steps_counter < max_steps:
        population.selection()
        population.perform_crossover(0.75)
        population.mutate(0.1)
        population.fitness(objective_func.get_func)
        current_best = sorted(population.individuals, key=lambda x: x.fitness_score)[0]
        # print(population)

        if current_best.fitness_score < best.fitness_score:
            best = copy.deepcopy(current_best)

        # print(best, "|", current_best)
        steps_counter += 1

    print("best solution for", objective_func.name, best)
    stop_time = time.time()
    delta = (stop_time - start_time) * 1000

    # return SolverOutput(func_values, beta, x0, delta)
