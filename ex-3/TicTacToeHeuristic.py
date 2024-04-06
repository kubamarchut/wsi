import numpy as np
from typing import Optional

cache = {}


def heuristic_matrix(size: int = 3) -> np.ndarray:
    matrix = np.full((size, size), size)
    center = size // 2
    matrix[center][center] += 1
    matrix[center][0] -= 1
    matrix[center][size - 1] -= 1
    matrix[0][center] -= 1
    matrix[size - 1][center] -= 1

    return matrix


def is_terminal(state: np.ndarray, use_cache: bool = False) -> Optional[int]:
    if use_cache and tuple(state.flatten()) in cache:
        return cache[tuple(state.flatten())]

    winner = check_for_winner(state)

    if winner is not None:
        if use_cache:
            cache[tuple(state.flatten())] = winner
        return winner
    elif np.count_nonzero(state) == state.size:
        if use_cache:
            cache[tuple(state.flatten())] = 0
        return 0
    else:
        if use_cache:
            cache[tuple(state.flatten())] = None
        return None


def evaluate_game(state: np.ndarray) -> int:
    if is_terminal(state) != None:
        return is_terminal(state) * 10
    else:
        return np.sum(np.multiply(heuristic_matrix(), state))


def check_for_winner(state: np.ndarray) -> Optional[int]:
    diag = np.diag(state)
    flipped_diag = np.diag(np.fliplr(state))

    for player in [-1, 1]:
        if np.any(np.all(state == player, axis=0) | np.all(state == player, axis=1)):
            return player

        if np.all(diag == player) or np.all(flipped_diag == player):
            return player

    return None


if __name__ == "__main__":
    """print(heuristic_matrix((0, 0)))
    print(heuristic_matrix((0, 1)))
    print(heuristic_matrix((0, 2)))

    print(heuristic_matrix((1, 0)))
    print(heuristic_matrix((1, 1)))
    print(heuristic_matrix((1, 2)))

    print(heuristic_matrix((2, 0)))
    print(heuristic_matrix((2, 1)))
    print(heuristic_matrix((2, 2)))
    """
    test_board = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    test_board = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    test_board = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])

    print(is_terminal(test_board))
    print(evaluate_game(test_board))
