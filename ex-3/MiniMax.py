import numpy as np
from typing import Tuple
from math import inf

from TicTacToeHeuristic import TicTacToe_heuristic


def evaluate_game(state: np.ndarray) -> int:
    raise NotImplementedError


def minmax(
    state: np.ndarray,
    depth: int,
    maximizing: bool,
    action: Tuple[int, int],
) -> int:
    if depth == 0:
        return TicTacToe_heuristic(3, state)
    if maximizing:
        value = -inf
        for child in state:
            value = max(value, minmax(child, depth - 1, False, (0, 0)))
        return value
    else:
        value = inf
        for child in state:
            value = min(value, minmax(child, depth - 1, True, (0, 0)))
        return value


def alpha_pruning(
    state: np.ndarray,
    action: Tuple[int, int],
    maximizing: bool,
    alpha: int,
    beta: int,
) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    print(minmax([(1, 1)], 1, False, (0, 0)))
