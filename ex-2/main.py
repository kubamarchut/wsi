#!/usr/bin/env python

import numpy as np

from solver import solver
from objective_func import Rastrigin, Gierwank, DropWave


def main():
    rastrigin = Rastrigin(d=2)
    solver(rastrigin, population_size=100, max_steps=100)

    gierwank = Gierwank(d=2)
    solver(gierwank, population_size=100, max_steps=100)

    dropwave = DropWave()
    solver(dropwave, population_size=100, max_steps=100)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
