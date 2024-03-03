import numpy as np
from solver import solver
from objective_func import QFunc
from tests.beta_tests import test_beta_impact
from tests.simple_test import simple_test

np.set_printoptions(precision=2)


def main():
    """
    Main function to demonstrate the functionality of the solver and objective function.

    This function performs two tests:
    1. Simple test to check if the solver works.
    2. Test the impact of the beta parameter on gradient descent method performance.
    """
    alphas = [1, 10, 100]
    n = 10

    print("\nChecking if solver works")
    simple_test(QFunc, alphas, n)

    print("\nTesting impact of beta parameter on gradient descant method performance")
    test_beta_impact(QFunc, alphas, n)


if __name__ == "__main__":
    main()
