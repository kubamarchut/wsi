import numpy as np
from solver import *
from objective_func import *


def main():
    alphas = [1, 10, 100]
    n = 10
    for alpha in alphas:
        print(f"\ntesting for alpha = {alpha}")
        objectiveFunc = QFunc(alpha=alpha, n=n)
        solver(objectiveFunc.getFunc, np.ones(n))


if __name__ == "__main__":
    main()
