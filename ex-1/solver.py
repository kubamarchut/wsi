from autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import math

threshold = 10e12


class SoleverOuput:
    def __init__(self, subsequentValues, beta, startingPoint):
        self.subsequentValues = subsequentValues
        self.beta = beta
        self.startingPoint = startingPoint

    def printSteps(self):
        for index, value in enumerate(self.subsequentValues):
            indexLength = int(math.log10(len(self.subsequentValues))) + 1
            print(f"step {(index+1):{indexLength}} q(xt)={value}")

    def drawGraph(self):
        plt.plot(self.subsequentValues)
        plt.xlabel("x")
        plt.ylabel("q(x)")
        plt.title(f"Visualization of q(x) for alpha={1}")
        plt.show()

    def getFinalValue(self):
        return self.subsequentValues[-1]

    def getRunSteps(self):
        return len(self.subsequentValues)


def solver(object, x0, method="gradient-descent", beta=0.008, max_steps=200, eps=1e-6):
    if method == "gradient-descent":
        current_value = np.copy(x0)
        steps_counter = 0
        func_values = []

        while steps_counter < max_steps:
            grad_f = grad(object)
            diff = beta * grad_f(current_value)
            current_value -= diff
            func_values.append(object(current_value))

            if object(current_value) > threshold:
                break
            if len(func_values) > 2:
                if (
                    abs(func_values[-1] - func_values[-2]) < eps
                    or abs(func_values[-1] - func_values[-2]) > threshold
                ):
                    break

            steps_counter += 1

        return SoleverOuput(func_values, beta, x0)

    else:
        pass  # not yet implemented
