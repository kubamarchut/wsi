import matplotlib.pyplot as plt
import numpy as np
import sys
import multiprocessing as mp

from solver import solver


def generate_input(min_x, max_x, n):
    """
    Generate input matrix of values within a specified range.

    Args:
        min_x (float): The minimum value for input x.
        max_x (float): The maximum value for input x.
        n (int): The number of dimensions in the input matrix to generate.

    Returns:
        ndarray: An array of n input values.
    """
    return np.random.uniform(min_x, max_x, n)


def test_beta_impact(QFunc, alphas, n):
    """
    Perform beta impact analysis for a given QFunc and alpha values.

    Args:
        QFunc (class): The QFunc class representing the objective function.
        alphas (list): A list of alpha values to test.
        n (int): The number of dimensions in the function (default is 10).

    """
    print("\tstarting beta impact analysis function")
    _, axes = plt.subplots(1, len(alphas), figsize=(15, 5))
    plt.style.use("ggplot")
    #startingX = generate_input(-100, 100, n)
    startingX = np.ones(n)
    print(f"\tstarting point={startingX}")

    for i, alpha in enumerate(alphas):
        print(f"\t• testing for alpha = {alpha}")
        objectiveFunc = QFunc(alpha=alpha, n=n)

        breakpoint = 1 / alpha
        betaMax = 1.00 * breakpoint
        betaMin = 0.11 * breakpoint
        betaStep = 0.1 * breakpoint
        betas = np.arange(betaMin, betaMax, betaStep)

        for j, beta in enumerate(betas):
            print(
                f"\r\t\tcalculating for {j+1}/{len(betas)} of betas", end="", flush=True
            )
            sys.stdout.flush()
            roundFactor = len(np.format_float_positional(betaStep).split(".")[1])
            if (roundFactor > 4): roundFactor = 4
            displayedBeta = (
                str(round((beta * 100), roundFactor - 2)) + "%"
                if beta < 0.01
                else str(round(beta, roundFactor))
            )
            solvedDatab1 = solver(objectiveFunc.get_func, startingX, beta=beta, eps=0, threshold = 1e3, max_steps=25)
            axes[i].plot(
                solvedDatab1.subsequent_values,
                ":o",
                label=r"$\beta$=" + displayedBeta
            )
        sys.stdout.write("\r\033[K")
        print(f"\r\t\tcalculated for {len(betas)} betas\n")
        axes[i].set_xlabel("step's number")
        axes[i].set_ylabel(r"$q(x_t)$")
        axes[i].set_yscale("log")
        axes[i].set_title(f"Visualization of {r"$q(x_t)$"} for {r"$\alpha$"}={alpha}")
        axes[i].legend()
        axes[i].grid()

    plt.tight_layout()
    plt.show()


def worker_func(alpha, beta_range, n_test, objective_function, solver_func):
    results = []
    for beta in beta_range:
        print("Processing for beta", beta)
        SolvedDataForBeta = []
        for i in range(int(n_test)):
            starting_point = objective_function.gen_input()
            SolvedData = solver_func(
                objective_function.get_func, starting_point, beta, max_steps=10
            )
            SolvedDataForBeta.append(SolvedData)

        result = np.mean([obj.get_final_value() for obj in SolvedDataForBeta])
        results.append((beta, result))
        
    return results

def beta_test_impact_multiple(QFunc, alphas, n, n_test=5e2):
    print("\tstarting beta impact analysis function")
    for alpha in alphas:
        objective_function = QFunc(alpha=alpha, min_x=-100, max_x=100, n=n)

        breakpoint = 1 / alpha
        betaMax = 1.10 * breakpoint
        betaMin = 0.001 * breakpoint
        betaStep = 0.025 * breakpoint
        betas = np.arange(betaMin, betaMax, betaStep)
        print(betas)

        # Create a pool of workers
        pool = mp.Pool(mp.cpu_count())

        results = []
        chunk_size = len(betas) // mp.cpu_count()
        for i in range(0, len(betas), chunk_size):
            beta_range = betas[i:i+chunk_size]
            results.append(pool.apply_async(worker_func, args=(alpha, beta_range, n_test, objective_function, solver)))

        pool.close()
        pool.join()

        res = []
        for r in results:
            res.extend(r.get())

        betas, values = zip(*res)

        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        ax.plot(betas, values, ":o")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$q(x_t)$")
        ax.set_title(f"Impact of {r'$\beta$'} on final {r"$q(x_t)$"} value for {r'$\alpha$'}={alpha}")
        plt.yscale("log")
        plt.show()


if __name__ == "__main__":
    beta_test_impact_multiple()
