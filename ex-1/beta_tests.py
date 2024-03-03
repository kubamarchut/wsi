import matplotlib.pyplot as plt
import numpy as np
import sys

from solver import solver


def generate_input(minX, maxX, n):
    return np.random.uniform(minX, maxX, n)


def testBetaImpact(QFunc, alphas, n):
    print("\tstarting beta impact analysis function")
    fig, axes = plt.subplots(1, len(alphas), figsize=(15, 5))
    plt.style.use("ggplot")
    startingX = generate_input(-100, 100, n)
    print(f"\tstarting point={startingX}")

    for i, alpha in enumerate(alphas):
        print(f"\t• testing for alpha = {alpha}")
        objectiveFunc = QFunc(alpha=alpha, n=n)

        breakpoint = 1 / alpha
        betaMax = 1.001 * breakpoint
        betaMin = 0.996 * breakpoint
        betaStep = 0.0005 * breakpoint
        betas = np.arange(betaMin, betaMax, betaStep)

        for j, beta in enumerate(betas):
            print(
                f"\r\t\tcalculating for {j+1}/{len(betas)} of betas", end="", flush=True
            )
            sys.stdout.flush()
            roundFactor = len(np.format_float_positional(betaStep).split(".")[1])
            displayedBeta = (
                str(round((beta * 100), roundFactor - 2)) + "%"
                if beta < 0.01
                else str(round(beta, roundFactor))
            )
            solvedDatab1 = solver(objectiveFunc.get_func, startingX, beta=beta)
            axes[i].plot(
                solvedDatab1.subsequent_values,
                label=r"$\beta$=" + displayedBeta,
            )
        sys.stdout.write("\r\033[K")
        print(f"\r\t\tcalculated for {len(betas)} betas\n")
        axes[i].set_xlabel("step's number")
        axes[i].set_ylabel("q(xt)")
        axes[i].set_title(f"Visualization of q(x) for alpha={alpha}")
        axes[i].legend()
        axes[i].grid()

    plt.tight_layout()
    plt.show()
