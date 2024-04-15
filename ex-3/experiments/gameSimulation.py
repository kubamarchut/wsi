import sys
import time
import numpy as np
from math import inf
from typing import Callable

sys.path.append("..")
from MiniMax import minmax, alpha_pruning, possible_moves


def game_simulation(
    opt_function: Callable,
    time_per_step,
    starting_board=np.zeros(shape=(3, 3)),
    print_stages=False,
):
    if not callable(opt_function):
        raise ValueError("Optimalization function must be callable.")

    board = starting_board
    for i in range(np.count_nonzero(board), 9):
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
