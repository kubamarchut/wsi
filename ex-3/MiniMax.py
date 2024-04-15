import numpy as np
from math import inf
from typing import Tuple, List

from TicTacToeHeuristic import evaluate_game, is_terminal


def possible_moves(state: np.ndarray) -> List[Tuple[int, int]]:
    empty_positions = list(zip(*np.where(state == 0)))
    return empty_positions


def minmax(
    state: np.ndarray, action: Tuple[int, int], maximizing: bool, depth: int
) -> int:
    """Minimax algorithm with depth-limiting."""
    player_marker = 1 if not maximizing else -1
    state[action] = player_marker
    terminal_flag = is_terminal(state)

    # base cases
    if terminal_flag is not None:
        state[action] = 0
        return terminal_flag * 10
    if depth == 0:
        res = evaluate_game(state)
        state[action] = 0
        return res

    # recursively calculate scores for possible moves
    if maximizing:
        value = -inf
        for child in possible_moves(state):
            value = max(value, minmax(state, child, False, depth - 1))

        state[action] = 0
        return value
    else:
        value = inf
        for child in possible_moves(state):
            value = min(value, minmax(state, child, True, depth - 1))

        state[action] = 0
        return value


def alpha_pruning(
    state: np.ndarray,
    action: Tuple[int, int],
    maximizing: bool,
    depth: int,
    alpha: int = -inf,
    beta: int = inf,
) -> int:
    """Alpha-beta pruning algorithm with depth-limiting."""
    player_marker = 1 if not maximizing else -1
    state[action] = player_marker
    terminal_flag = is_terminal(state)

    # base cases
    if terminal_flag is not None:
        state[action] = 0
        return terminal_flag * 10
    if depth == 0:
        res = evaluate_game(state)
        state[action] = 0
        return res

    # recursively calculate scores for possible moves
    if maximizing:
        for child in possible_moves(state):
            alpha = max(
                alpha, alpha_pruning(state, child, False, depth - 1, alpha, beta)
            )
            if alpha >= beta:
                state[action] = 0
                return alpha

        state[action] = 0
        return alpha
    else:
        for child in possible_moves(state):
            beta = min(beta, alpha_pruning(state, child, True, depth - 1, alpha, beta))
            if alpha >= beta:
                state[action] = 0
                return beta

        state[action] = 0
        return beta
