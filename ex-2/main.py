#!/usr/bin/env python

import numpy as np

from solver import *
from objective_func import Rastrigin, Gierwank, DropWave


def main():
    print("Testing algorithm for Rastrigin function")
    rastrigin = Rastrigin(d=2)
    params = OptimizationParameters(objective_func=rastrigin)
    result = solver(params)
    print("\tbest achieved value:", get_final_value(result))
    # draw_best_individuals(result, n_points=1000)
    draw_graph(result)

    print("\nTesting algorithm for Gierwank function")
    gierwank = Gierwank(d=2)
    params = OptimizationParameters(objective_func=gierwank, sigma=0.1)
    result = solver(params)
    print("\tbest achieved value:", get_final_value(result))
    # draw_best_individuals(result, n_points=1000)
    draw_graph(result)

    print("\nTesting algorithm for Drop-Wave function")
    dropwave = DropWave()
    params = OptimizationParameters(objective_func=dropwave, sigma=0.02)
    result = solver(params)
    print("\tbest achieved value:", get_final_value(result))
    # draw_best_individuals(result, n_points=1000)
    draw_graph(result)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
