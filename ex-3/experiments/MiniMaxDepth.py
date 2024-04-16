import sys
import numpy as np
from typing import Tuple, List
from math import inf

from TicTacToeHeuristic import *

sys.path.append("..")


def possible_moves(state: np.ndarray) -> List[Tuple[int, int]]:
    empty_positions = list(zip(*np.where(state == 0)))
    return empty_positions


def minmax(
    state: np.ndarray,
    action: Tuple[int, int],
    maximizing: bool,
    depth: int,
    depth_counter: List[int],
    node_counter: List[int],
) -> int:
    node_counter[0] += 1
    player_marker = 1 if not maximizing else -1
    state[action] = player_marker
    terminal_flag = is_terminal(state)

    # base cases
    if terminal_flag is not None:
        state[action] = 0
        depth_counter.append(depth)
        return terminal_flag * 10
    if depth == 0:
        res = evaluate_game(state)
        state[action] = 0
        depth_counter.append(depth)
        return res

    # recursively calculate scores for possible moves
    if maximizing:
        value = -inf
        for child in possible_moves(state):
            child_value = minmax(
                state, child, False, depth - 1, depth_counter, node_counter
            )
            value = max(value, child_value)

        state[action] = 0
        return value
    else:
        value = inf
        for child in possible_moves(state):
            child_value = minmax(
                state, child, True, depth - 1, depth_counter, node_counter
            )
            value = min(value, child_value)

        state[action] = 0
        return value


def alpha_pruning(
    state: np.ndarray,
    action: Tuple[int, int],
    maximizing: bool,
    depth: int,
    depth_counter: List[int],
    node_counter: List[int],
    alpha: int = -inf,
    beta: int = inf,
) -> int:
    node_counter[0] += 1
    player_marker = 1 if not maximizing else -1
    state[action] = player_marker
    terminal_flag = is_terminal(state)

    # base cases
    if terminal_flag is not None:
        state[action] = 0
        depth_counter.append(depth)
        return terminal_flag * 10
    if depth == 0:
        res = evaluate_game(state)
        state[action] = 0
        depth_counter.append(depth)
        return res

    # recursively calculate scores for possible moves
    if maximizing:
        for child in possible_moves(state):
            child_value = alpha_pruning(
                state, child, False, depth - 1, depth_counter, node_counter, alpha, beta
            )
            alpha = max(alpha, child_value)
            if alpha >= beta:
                state[action] = 0
                return alpha

        state[action] = 0
        return alpha
    else:
        for child in possible_moves(state):
            child_value = alpha_pruning(
                state, child, True, depth - 1, depth_counter, node_counter, alpha, beta
            )
            beta = min(beta, child_value)
            if alpha >= beta:
                state[action] = 0
                return beta

        state[action] = 0
        return beta
