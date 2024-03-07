import numpy as np
import multiprocessing as mp

from solver import solver


def worker_func(alpha, beta_range, n_test, objective_function, solver_func):
    results = []
    for beta in beta_range:
        print("Processing for beta", beta)
        SolvedDataForBeta = []
        for i in range(int(n_test)):
            starting_point = objective_function.gen_input()
            SolvedData = solver_func(
                objective_function.get_func, starting_point, beta, max_steps=100
            )
            SolvedDataForBeta.append(SolvedData)

        result = np.mean([obj.get_final_value() for obj in SolvedDataForBeta])
        result_steps = np.mean([obj.get_run_steps() for obj in SolvedDataForBeta])
        results.append((beta, result, result_steps))
    return results


def beta_test_impact_stat(QFunc, alphas, n, n_test=1e2):
    print("\tstarting beta impact stat analysis function")
    for alpha in alphas:
        print("testing for alpha =", alpha)
        objective_function = QFunc(alpha=alpha, min_x=-100, max_x=100, n=n)

        breakpoint = 1 / alpha
        betaMax = 1.00 * breakpoint
        betaMin = 0.11 * breakpoint
        betaStep = 0.1 * breakpoint
        betas = np.arange(betaMin, betaMax, betaStep)
        print(betas)

        # Create a pool of workers
        pool = mp.Pool(mp.cpu_count())

        results = []
        chunk_size = len(betas) // mp.cpu_count()
        for i in range(0, len(betas), chunk_size):
            beta_range = betas[i : i + chunk_size]
            results.append(
                pool.apply_async(
                    worker_func,
                    args=(alpha, beta_range, n_test, objective_function, solver),
                )
            )

        pool.close()
        pool.join()

        res = []
        for r in results:
            res.extend(r.get())

        # betas, values = zip(*res)
        for beta, value, steps in res:
            roundFactor = len(np.format_float_positional(betaStep).split(".")[1])
            if roundFactor > 4:
                roundFactor = 4
            displayedBeta = (
                str(round((beta * 100), roundFactor - 2)) + "%"
                if beta < 0.01
                else str(round(beta, roundFactor))
            )
            print(f"{displayedBeta:1.3} | {value:8.2} | {steps:>3}")


if __name__ == "__main__":
    alphas = [1, 10, 100]
    n = 10
    beta_test_impact_stat(QFunc, alphas, n)
