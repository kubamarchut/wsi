import numpy as np
from typing import Tuple, List
from math import inf

from TicTacToeHeuristic import evaluate_game, is_terminal


def possible_moves(state: np.ndarray, player: bool) -> List[np.ndarray]:
    player_marker = 1 if player else -1
    empty_positions = list(zip(*np.where(state == 0)))
    return [apply_move(state, pos, player_marker) for pos in empty_positions]


def apply_move(
    state: np.ndarray, position: Tuple[int, int], player_marker: int
) -> np.ndarray:
    next_state = state.copy()
    next_state[position] = player_marker
    return next_state


def minmax(
    state: np.ndarray, action: Tuple[int, int], maximizing: bool, depth: int
) -> int:
    if depth == 0 or is_terminal(state) != None:
        return evaluate_game(state)

    if maximizing:
        value = -inf
        for child in possible_moves(np.copy(state), maximizing):
            value = max(value, minmax(child, (0, 0), False, depth - 1))
        return value
    else:
        value = inf
        for child in possible_moves(np.copy(state), maximizing):
            value = min(value, minmax(child, (0, 0), True, depth - 1))
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
    starting_board = np.zeros(shape=(3, 3))

    starting_board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    # print(minmax(starting_board, (1, 1), False, 10))
    # print(minmax(starting_board, (1, 1), True, 10))

    board = starting_board
    for i in range(9):
        if i % 2 == 0:
            best = -inf
        else:
            best = inf
        chosen = np.zeros(shape=(3, 3))
        for move in possible_moves(board, i % 2 == 0):
            if i % 2 == 0:
                move_score = minmax(move, (1, 1), False, 9)
                # print(move, move_score)
                if best < move_score:
                    best = move_score
                    chosen = move

            else:
                move_score = minmax(move, (1, 1), True, 9)
                # print(move, move_score)
                if best > move_score:
                    best = move_score
                    chosen = move

        print(10 * "-")
        print(chosen, best)
        print(10 * "-")
        board = chosen
