import numpy as np
from typing import Tuple, List
from math import inf

from TicTacToeHeuristic import evaluate_game, is_terminal


def possible_moves(state: np.ndarray) -> List[Tuple[int, int]]:
    empty_positions = list(zip(*np.where(state == 0)))
    return empty_positions


def minmax(
    state: np.ndarray, action: Tuple[int, int], maximizing: bool, depth: int
) -> int:
    player_marker = 1 if not maximizing else -1
    state[action] = player_marker
    terminal_flag = is_terminal(state)
    if terminal_flag is not None:
        state[action] = 0
        return terminal_flag * 10
    if depth == 0:
        res = evaluate_game(state)
        state[action] = 0
        return res
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
    alpha: int,
    beta: int,
) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    starting_board = np.zeros(shape=(3, 3))

    starting_board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    # print(minmax(starting_board, (1, 1), False, 10))
    """
    for move in possible_moves(starting_board):
        print(minmax(starting_board, move, False, 9))
        break
    """
    board = starting_board
    for i in range(9):
        if i % 2 == 0:
            best = -inf
        else:
            best = inf
        chosen = np.zeros(shape=(3, 3))
        for move in possible_moves(board):
            if i % 2 == 0:
                move_score = minmax(board, move, False, 9)
                # print("move:", board, move_score)
                # print(move, move_score)
                if best < move_score:
                    best = move_score
                    chosen = move

            else:
                move_score = minmax(board, move, True, 9)
                # print("move:", board, move, move_score)
                # print(move, move_score)
                if best > move_score:
                    best = move_score
                    chosen = move

        board[chosen] = 1 if i % 2 == 0 else -1
        print(10 * "-")
        print(board, best)
        print(10 * "-")
