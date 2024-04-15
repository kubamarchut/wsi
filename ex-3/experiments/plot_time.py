import csv
import matplotlib.pyplot as plt
import statistics
import numpy as np


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
    plt.show()

    # Save data to CSV if filename provided
    if csv_filename:
        with open(csv_filename, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Method", "Steps", "Average", "Max", "Min", "StdDev"])
            for row in csv_data:
                csv_writer.writerow(row)
