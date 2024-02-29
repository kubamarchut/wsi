import numpy as np
from solver import *
from objective_func import *
from beta_tests import testBetaImpact


def simpleTest(QFunc, alphas, n):
    print("\tstarting simple test to check if solver works")
    for alpha in alphas:
        print(f"\t• testing for alpha = {alpha}")
        objectiveFunc = QFunc(alpha=alpha, n=n)
        startingPoint = QFunc.genInput(n, -100, 100)
        # solvedData = solver(objectiveFunc.getFunc, np.ones(n))
        solvedData = solver(objectiveFunc.getFunc, startingPoint)

        print(f"\t\tbeta={solvedData.beta}")
        print(f"\t\tstarting point={solvedData.startingPoint}")
        print(f"\t\tsteps={solvedData.getRunSteps()}")
        print(f"\t\tq(xt)={solvedData.getFinalValue()}\n")


def main():
    alphas = [1, 10, 100]
    n = 10

    print("\nChecking if solver works")
    simpleTest(QFunc, alphas, n)

    print("\nTesting impact of beta parameter on gradient descant method performance")
    testBetaImpact(QFunc, alphas, n)


if __name__ == "__main__":
    main()
