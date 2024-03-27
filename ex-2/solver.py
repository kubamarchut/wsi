import matplotlib.pyplot as plt
import numpy as np
import math
import time
import copy
from typing import List, Tuple

from evolution import *
from objective_func import QFunc


class SolverOutput:
    def __init__(
        self,
        best_in_each_generation: List[Individual],
        avg_fitness_per_pop: List[float],
        optimized_function: QFunc,
        time_delta: float,
    ) -> None:
        self.best_in_each_generation = best_in_each_generation
        self.avg_fitness_per_pop = avg_fitness_per_pop
        self.optimized_function = optimized_function
        self.time_delta = time_delta


class OptimizationParameters:
    def __init__(
        self,
        objective_func: QFunc,
        population_size: int = 100,
        max_steps: int = 100,
        sigma: float = 0.2,
        mut_prob: float = 0.6,
        cross_prob: float = 0.6,
        verbose_graphs: bool = False,
        random_start=True,
    ) -> None:
        self.objective_func = objective_func
        self.population_size = population_size
        self.max_steps = max_steps
        self.sigma = sigma
        self.mut_prob = mut_prob
        self.cross_prob = cross_prob
        self.verbose_graphs = verbose_graphs
        self.random_start = random_start


def print_steps(solver_output: SolverOutput) -> None:
    for index, value in enumerate(solver_output.best_in_each_generation):
        index_length = int(math.log10(len(solver_output.best_in_each_generation))) + 1
        print(f"step {(index+1):{index_length}} q(xt)={value:.3E}")


def get_final_value(solver_output: SolverOutput) -> Individual:
    return min(solver_output.best_in_each_generation, key=lambda ind: ind.fitness_score)


def solver(params: OptimizationParameters) -> SolverOutput:
    """
    Solve the optimization problem using genetic algorithm.

    Args:
        params (OptimizationParameters): The parameters for optimization.

    Returns:
        SolverOutput: An object containing information about solver execution.
    """
    objective_func = params.objective_func

    if not callable(objective_func.get_func):
        raise ValueError("Objective function must be callable.")

    # generate new population
    population = Population(
        params.population_size,
        objective_func.d,
        min_x=objective_func.min_x,
        max_x=objective_func.max_x,
    )
    # generate new individuals with random chromosomes
    population.initialize(params.random_start)

    # evaluating individuals based on fitness function
    population.fitness(objective_func.get_func)

    # finding best individual in initial population
    initial_population = np.copy(population.individuals)
    best = min(initial_population, key=lambda ind: ind.fitness_score)

    # creating list to store best individual from each generation
    best_in_each_generation = []
    best_in_each_generation.append(copy.deepcopy(best))

    avgs_per_pop = []

    initial_avg = (
        sum([ind.fitness_score for ind in population.individuals])
        / params.population_size
    )
    avgs_per_pop.append(initial_avg)

    steps_counter = 0
    start_time = time.time()
    while steps_counter < params.max_steps:
        population.selection()
        population.perform_crossover(params.cross_prob)
        population.mutate(
            params.sigma, params.mut_prob, steps_counter, params.max_steps
        )
        population.fitness(objective_func.get_func)
        current_best = min(population.individuals, key=lambda ind: ind.fitness_score)
        current_avg = (
            sum([ind.fitness_score for ind in population.individuals])
            / params.population_size
        )
        avgs_per_pop.append(current_avg)

        if params.verbose_graphs and steps_counter == params.max_steps - 1:
            draw_each_population(
                population, objective_func, steps_counter, params.max_steps
            )

        best_in_each_generation.append(copy.deepcopy(current_best))
        if current_best.fitness_score < best.fitness_score:
            best = copy.deepcopy(current_best)

        steps_counter += 1

    stop_time = time.time()
    time_delta = (stop_time - start_time) * 1000

    return SolverOutput(
        best_in_each_generation,
        avgs_per_pop,
        objective_func,
        time_delta,
    )


def draw_graph(solver_output: SolverOutput) -> None:
    plt.plot(solver_output.avg_fitness_per_pop)
    plt.xlabel("n of population")
    plt.ylabel("best q(x)")
    plt.title(f"Visualization of {solver_output.optimized_function.name} over time")
    plt.show()


def draw_points(
    points: List[Tuple[float, float]], function: QFunc, label: str, n_points: int = 1000
) -> None:
    X = np.linspace(function.min_x, function.max_x, n_points)
    Y = np.linspace(function.min_x, function.max_x, n_points)
    X, Y = np.meshgrid(X, Y)

    combined = np.vstack((X.ravel(), Y.ravel())).T

    Z = function.calculate(combined)

    Z = Z.reshape(X.shape)  # Reshape Z to match the shape of X and Y

    # Plot the heatmap
    plt.imshow(
        Z,
        extent=[function.min_x, function.max_x, function.min_x, function.max_x],
        origin="lower",
        cmap="viridis",
        alpha=1,
    )

    # Plot the points
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    plt.scatter(x_coords, y_coords, color="#D81E5B", alpha=0.8, label=label)
    plt.plot(x_coords, y_coords, color="#D81E5B", alpha=0.3)

    plt.xlabel("współrzędne X")
    plt.ylabel("współrzędne Y")
    plt.legend()
    plt.colorbar(label=f"Wartości funkcji {function.name}")
    plt.show()


def draw_each_population(
    population: Population,
    function: QFunc,
    population_n: int,
    max_population: int,
    n_points: int = 1000,
) -> None:
    points = [ind.x for ind in population.individuals]
    label = f"osobniki {population_n+1}/{max_population} polulacji"
    draw_points(points, function, label, n_points)


def draw_best_individuals(solver_output: SolverOutput, n_points: int = 1000) -> None:
    points = [ind.x for ind in solver_output.best_in_each_generation]
    label = f"najlepsze osobniki n-tych populacji"
    draw_points(points, solver_output.optimized_function, label, n_points)
