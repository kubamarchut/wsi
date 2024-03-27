import concurrent.futures
import numpy as np
from objective_func import *
from solver import *
from main import OptimizationParameters, solver
from typing import List
from tqdm import tqdm


def run_solver(params: OptimizationParameters) -> float:
    result = solver(params)
    return get_final_value(result).fitness_score


def param_test(param_name: str, param_range: List, function: QFunc, n=100) -> None:
    results = []

    for param_variation in param_range:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_results = []
            progress_bar = tqdm(
                total=n,
                desc=f"Testing {param_name}={param_variation:.2f}",
                position=0,
                leave=True,
            )
            for _ in range(n):
                params = OptimizationParameters(function)
                params.random_start = True
                setattr(params, param_name, param_variation)
                params.population_size = 20
                params.max_steps = 10
                params.sigma = 1
                future_results.append(executor.submit(run_solver, params))

            for future in concurrent.futures.as_completed(future_results):
                future.result()
                progress_bar.update(1)
            progress_bar.close()

            result_for_var = [future.result() for future in future_results]
            results.append((param_variation, sum(result_for_var) / n))

    for param_variation, avg_result in results:
        print(f"{param_variation:.2f}", avg_result)


if __name__ == "__main__":
    test_function = Rastrigin(d=2)

    param_test("mul_prob", np.arange(0, 1, 0.1), test_function)

    test_function = Gierwank(d=2)

    param_test("max_steps", np.arange(1, 50, 1), test_function)

    test_function = DropWave()

    param_test("max_steps", np.arange(1, 50, 1), test_function)
    # param_test("cross_prob", np.arange(0, 1.01, 0.1), test_function)
