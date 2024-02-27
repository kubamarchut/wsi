#!/usr/bin/env python

import numpy as np
from autograd import grad
from q import *

max_steps = 6
beta = 0.2


def solver(object, x0, method="gradient-descent"):
    if method == "gradient-descent":
        current_value = x0
        steps_counter = 0

        while steps_counter <= max_steps:
            grad_f = grad(object)
            diff = beta * grad_f(current_value)
            current_value -= diff

            steps_counter += 1
            print(current_value)
    else:
        pass  # not yet implemented


def main():
    solver(q, np.ones(10))


if __name__ == "__main__":
    main()
