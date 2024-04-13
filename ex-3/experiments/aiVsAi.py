import sys
import time
import threading
import statistics
from tqdm import tqdm
import numpy as np
from math import inf
import matplotlib
import matplotlib.pyplot as plt

sys.path.append("..")
from MiniMax import minmax, alpha_pruning, possible_moves


import csv


import csv


def plot_time_per_step(data, csv_filename=None):
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6)
    )  # Create 1 row and 2 columns of subplots

    csv_data = []

    for ax_idx, ax in enumerate(axes):  # Iterate over the axes
        ax.grid(which="minor", linewidth=0.5, color="lightgrey")
        ax.grid()
        ax.minorticks_on()

        for index, method in enumerate(data):
            method_data = list(method.values())[0]
            method_name = list(method.keys())[0]
            averages = {
                key: sum(values) / len(values) for key, values in method_data.items()
            }
            maxes = {key: max(values) for key, values in method_data.items()}
            mins = {key: min(values) for key, values in method_data.items()}
            stddev = {
                key: statistics.stdev(values) for key, values in method_data.items()
            }

            # Extract keys and values
            steps = np.array(list(averages.keys())) + 1
            avg_values = np.array(list(averages.values()))
            maxes = np.array(list(maxes.values()))
            mins = np.array(list(mins.values()))
            stddev = np.array(list(stddev.values()))

            # Append data to CSV
            if (
                csv_filename and ax_idx == 0
            ):  # Save data only once (for the linear scale plot)
                for step, avg, max_val, min_val, std_dev in zip(
                    steps, avg_values, maxes, mins, stddev
                ):
                    csv_data.append([method_name, step, avg, max_val, min_val, std_dev])

            else:
                ax.set_yscale("log")
            # Creating a line plot
            ax.errorbar(
                steps,
                avg_values,
                [avg_values - mins, maxes - avg_values],
                fmt="None",
                ecolor="black",
                capsize=5,
                capthick=2,
                elinewidth=2,
                lw=1,
                label="min - max",
            )
            ax.errorbar(
                steps,
                avg_values,
                stddev,
                fmt="None",
                ecolor="grey",
                capsize=7,
                capthick=2,
                elinewidth=5,
                lw=3,
                label="odchylenie standardowe",
            )
            ax.plot(
                steps,
                np.clip(avg_values, 1e-10, None),
                marker="o",
                linestyle="--",
                label=f"{method_name} - wartości średnie",
            )

        # Adding title and labels
        ax.set_title("Czas wyboru kolejnego ruchu w symulacji gry")
        ax.set_xlabel("Numer ruchu")
        ax.set_ylabel("Średni czas wyboru kolejnego ruchu")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    # Show the plot
    plt.tight_layout()
    plt.savefig("graph.png")

    # Save data to CSV if filename provided
    if csv_filename:
        with open(csv_filename, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Method", "Steps", "Average", "Max", "Min", "StdDev"])
            for row in csv_data:
                csv_writer.writerow(row)


def game_simulation(
    opt_function,
    time_per_step,
    starting_board=np.zeros(shape=(3, 3)),
    print_stages=False,
):
    if not callable(opt_function):
        raise ValueError("Optimalization function must be callable.")

    board = starting_board
    for i in range(9):
        if i % 2 == 0:
            best = -inf
        else:
            best = inf
        chosen = np.zeros(shape=(3, 3))
        for move in possible_moves(board):
            if i % 2 == 0:
                start_time = time.time_ns()
                move_score = opt_function(board, move, False, 9)
                end_time = time.time_ns()
                elapsed_time = (end_time - start_time) / 10**9
                if i in time_per_step:
                    time_per_step[i].append(elapsed_time)
                else:
                    time_per_step[i] = [elapsed_time]
                if best < move_score:
                    best = move_score
                    chosen = move

            else:
                start_time = time.time_ns()
                move_score = opt_function(board, move, True, 9)
                end_time = time.time_ns()
                elapsed_time = (end_time - start_time) / 10**9
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


def test_solution_single(opt_function, time_per_step, starting_board):
    start_time = time.time()
    game_simulation(opt_function, time_per_step, starting_board)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time


def test_solution_multiple(opt_function, n_tests=2):
    results = []
    time_per_step = {}
    progress_bar = tqdm(
        total=n_tests,
        desc=f"Testing {opt_function.__name__}",
        position=0,
        leave=True,
    )

    def run_test():
        starting_board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        results.append(
            test_solution_single(opt_function, time_per_step, starting_board)
        )

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
    font = {"weight": "bold", "size": 12}

    matplotlib.rc("font", **font)
    print("Testing alpha pruning algorithm:")
    stats_data = test_solution_multiple(alpha_pruning)
    # print_statistics("AlphaBeta", stats_data)

    print("Testing minmax algorithm:")
    stats_data_2 = test_solution_multiple(minmax)
    # print_statistics("MiniMax", stats_data_2)

    plot_time_per_step(
        [{"alpha": stats_data["per_step"]}, {"minmax": stats_data_2["per_step"]}],
        "simulation-data.csv",
    )
