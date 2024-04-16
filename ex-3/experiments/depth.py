import sys
import numpy as np
from typing import Tuple, List
from math import inf

sys.path.append("..")
from TicTacToeHeuristic import *
from MiniMaxDepth import minmax, alpha_pruning, possible_moves

max_depth = 9


def analyze_game_states():
    state = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    max_depth = np.size(state) - np.count_nonzero(state)
    initial_states = possible_moves(state)
    initial_states = [(2, 2)]

    for action in initial_states:
        # print("Analyzing state:", state)
        depth_counter_minimax = [0]
        node_counter_minimax = [0]
        depth_counter_alpha_pruning = [0]
        node_counter_alpha_pruning = [0]

        # Analiza Minimax
        res_m = minmax(
            state,
            action,
            maximizing=True,
            depth=max_depth,
            depth_counter=depth_counter_minimax,
            node_counter=node_counter_minimax,
        )

        # Analiza Alpha-Beta Pruning
        res_a = alpha_pruning(
            state,
            action,
            maximizing=True,
            depth=max_depth,
            depth_counter=depth_counter_alpha_pruning,
            node_counter=node_counter_alpha_pruning,
        )
        print(action)
        print(
            "Minimax - Depth:",
            f"{max_depth - sum(depth_counter_minimax) / len(depth_counter_minimax):.3f}",
            "Nodes visited:",
            node_counter_minimax[0],
        )
        print(
            "Alpha-Beta - Depth:",
            f"{max_depth - sum(depth_counter_alpha_pruning) / len(depth_counter_alpha_pruning):.3f}",
            "Nodes visited:",
            node_counter_alpha_pruning[0],
        )


analyze_game_states()
