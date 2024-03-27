#!/usr/bin/env python

import numpy as np
import matplotlib

from solver import *
from objective_func import Rastrigin, Gierwank, DropWave


def main():
    print("Testing algorithm for Rastrigin function")
    rastrigin = Rastrigin(d=2)
    params = OptimizationParameters(objective_func=rastrigin)
    # result = solver(params)
    # print("\tbest achieved value:", get_final_value(result))
    # draw_best_individuals(result, n_points=1000)
    # draw_graph(result)

    print("\nTesting algorithm for Griewank function")
    gierwank = Gierwank(d=2)
    params = OptimizationParameters(objective_func=gierwank)
    # result = solver(params)
    # print("\tbest achieved value:", get_final_value(result))
    # draw_best_individuals(result, n_points=1000)
    # draw_graph(result)

    print("\nTesting algorithm for Drop-Wave function")
    dropwave = DropWave()
    params = OptimizationParameters(objective_func=dropwave, sigma=0.5)
    result = solver(params)
    print("\tbest achieved value:", get_final_value(result))
    draw_best_individuals(result, n_points=1000)
    draw_graph(result)


if __name__ == "__main__":
    font = {"weight": "bold", "size": 12}
    matplotlib.rc("font", **font)
    np.set_printoptions(precision=3)
    main()
