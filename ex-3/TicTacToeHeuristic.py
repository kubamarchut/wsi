import numpy as np
from typing import Tuple


def TicTacToe_heuristic(size: int, move: Tuple[int, int]):
    matrix = np.full((size, size), size)
    center = size // 2
    matrix[center][center] += 1
    matrix[center][0] -= 1
    matrix[center][size - 1] -= 1
    matrix[0][center] -= 1
    matrix[size - 1][center] -= 1

    return matrix[move[0]][move[1]]


if __name__ == "__main__":
    print(TicTacToe_heuristic(3, (0, 0)))
    print(TicTacToe_heuristic(3, (0, 1)))
    print(TicTacToe_heuristic(3, (0, 2)))

    print(TicTacToe_heuristic(3, (1, 0)))
    print(TicTacToe_heuristic(3, (1, 1)))
    print(TicTacToe_heuristic(3, (1, 2)))

    print(TicTacToe_heuristic(3, (2, 0)))
    print(TicTacToe_heuristic(3, (2, 1)))
    print(TicTacToe_heuristic(3, (2, 2)))
