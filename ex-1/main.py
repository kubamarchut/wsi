import numpy as np
from solver import *
from objective_func import *
from beta_tests import testBetaImpact

np.set_printoptions(precision=2)


def simple_test(QFunc, alphas, n):
    print("\tstarting simple test to check if solver works")
    for alpha in alphas:
        print(f"\t• testing for alpha = {alpha}")
        objectiveFunc = QFunc(alpha=alpha, n=n)
        startingPoint = objectiveFunc.gen_input(-100, 100)
        # solvedData = solver(objectiveFunc.get_func, np.ones(n))
        solvedData = solver(objectiveFunc.get_func, startingPoint)

        print(f"\t\tbeta={solvedData.beta}")
        print(f"\t\tstarting point={solvedData.starting_point}")
        print(f"\t\tsteps={solvedData.get_run_steps()}")
        print(f"\t\tq(xt)={solvedData.get_final_value()}\n")


def main():
    alphas = [1, 10, 100]
    n = 10

    print("\nChecking if solver works")
    simple_test(QFunc, alphas, n)

    print("\nTesting impact of beta parameter on gradient descant method performance")
    testBetaImpact(QFunc, alphas, n)


if __name__ == "__main__":
    main()
