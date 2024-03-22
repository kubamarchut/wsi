#!/usr/bin/env python

from solver import solver
from objective_func import Rastrigin, Gierwank, DropWave


def main():
    rastrigin = Rastrigin(d=2)
    solver(rastrigin, population_size=4)


if __name__ == "__main__":
    main()
