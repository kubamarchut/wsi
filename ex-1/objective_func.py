import numpy as np
import matplotlib.pyplot as plt


class QFunc:
    def __init__(self, alpha, n=10):
        self.alpha = alpha
        self.n = n

    def genInput(n, minX, maxX):
        return np.random.uniform(minX, maxX, n)

    def setAlpha(self, new_alpha):
        self.alpha = new_alpha

    def getAlpha(self):
        return self.alpha

    def getFunc(self, x):
        indices = np.arange(1, self.n + 1)
        powers = self.alpha ** ((indices - 1) / (self.n - 1))

        return np.sum(powers * np.power(x, 2))

    def drawFunc(self, points=10000):
        x_range = np.linspace(-100, 100, points)
        X = np.zeros((len(x_range), self.n))
        for i, x_val in enumerate(x_range):
            X[i] = x_val * np.ones(self.n)

        y = self.calculateFunc(X)

        plt.scatter(X[:, 0], y)
        plt.colorbar(label="q(x)")
        plt.xlabel("x")
        plt.ylabel("q(x)")
        plt.title(f"Visualization of q(x) for alpha={self.alpha}")
        plt.show()

    def calculateFunc(self, X):
        return [self.getFunc(x) for x in X]


if __name__ == "__main__":
    alphas = [1, 10, 100]

    for alpha in alphas:
        objectiveFunc = QFunc(alpha)
        objectiveFunc.drawFunc()
