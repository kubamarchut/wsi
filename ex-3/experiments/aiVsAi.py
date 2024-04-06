import sys
import time
import threading
import statistics
from tqdm import tqdm
import numpy as np
from math import inf
import matplotlib.pyplot as plt

sys.path.append("..")
from MiniMax import minmax, alpha_pruning, possible_moves


def plot_time_per_step(data):
    fig, ax = plt.subplots()

    for method in data:
        averages = {key: sum(values) / len(values) for key, values in method.items()}

        # Extract keys and values
        steps = list(averages.keys())
        avg_values = list(averages.values())

        # Creating a line plot
        ax.plot(
            steps, avg_values, marker="o", linestyle="--"
        )  # 'o' adds a marker for each data point

    # Adding title and labels
    ax.set_title("Duration per step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Avg duration")

    # Display the plot
    plt.show()


def game_simulation(opt_function, time_per_step, print_stages=False):
    if not callable(opt_function):
        raise ValueError("Optimalization function must be callable.")
    starting_board = np.zeros(shape=(3, 3))
    starting_board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    board = starting_board
    for i in range(9):
        if i % 2 == 0:
            best = -inf
        else:
            best = inf
        chosen = np.zeros(shape=(3, 3))
        for move in possible_moves(board):
            if i % 2 == 0:
                start_time = time.time()
                move_score = opt_function(board, move, False, 9)
                end_time = time.time()
                elapsed_time = end_time - start_time
                if i in time_per_step:
                    time_per_step[i].append(elapsed_time)
                else:
                    time_per_step[i] = [elapsed_time]
                if best < move_score:
                    best = move_score
                    chosen = move

            else:
                start_time = time.time()
                move_score = opt_function(board, move, True, 9)
                end_time = time.time()
                elapsed_time = end_time - start_time
                if i in time_per_step:
                    time_per_step[i].append(elapsed_time)
                else:
                    time_per_step[i] = [elapsed_time]
                if best > move_score:
                    best = move_score
                    chosen = move

        board[chosen] = 1 if i % 2 == 0 else -1
        if print_stages:
            print(10 * "-")
            print(board, best)
            print(10 * "-")


def print_statistics(method, statistics_dict):
    print(f"AI vs AI Game Duration Statistics - {method}:")
    print("-------------------------------------" + "-" * len(method))
    print(f'Average Duration: {statistics_dict["average"]:.3f}')
    print(f'Minimum Duration: {statistics_dict["minimum"]:.3f}')
    print(f'Maximum Duration: {statistics_dict["maximum"]:.3f}')
    print(f'Standard Deviation: {statistics_dict["stddev"]:.3f}')
    # print("All Durations:", statistics_dict["results"])
    print("\n")


def test_solution_single(opt_function, time_per_step):
    start_time = time.time()
    game_simulation(opt_function, time_per_step)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time


def test_solution_multiple(opt_function, n_tests=10):
    results = []
    time_per_step = {}
    progress_bar = tqdm(
        total=n_tests,
        desc=f"Testing {opt_function.__name__}",
        position=0,
        leave=True,
    )

    def run_test():
        results.append(test_solution_single(opt_function, time_per_step))

    # threads = []
    for _ in range(n_tests):
        progress_bar.update(1)
        run_test()
        # thread = threading.Thread(target=run_test)
        # thread.start()
        # threads.append(thread)

    progress_bar.close()
    # for thread in threads:
    #    thread.join()

    avg = statistics.mean(results)
    minimum = min(results)
    maximum = max(results)
    stddev = statistics.stdev(results)

    return {
        "average": avg,
        "minimum": minimum,
        "maximum": maximum,
        "stddev": stddev,
        "results": results,
        "per_step": time_per_step,
    }


if __name__ == "__main__":
    print("Testing alpha pruning algorithm:")
    stats_data = test_solution_multiple(alpha_pruning)
    print_statistics("AlphaBeta", stats_data)

    print("Testing minmax algorithm:")
    stats_data_2 = test_solution_multiple(minmax)
    print_statistics("MiniMax", stats_data_2)

    plot_time_per_step([stats_data["per_step"], stats_data_2["per_step"]])
