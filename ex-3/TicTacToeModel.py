import numpy as np
from typing import Tuple


class GameParams:
    def __init__(
        self,
        size: int = 3,
        empty: str = " ",
        first_player: str = "O",
        second_player: str = "X",
    ) -> None:
        self.size = size
        self.empty = empty
        self.first_player = first_player
        self.second_player = second_player


class TicTacToeModel:
    def __init__(self, game_params: GameParams) -> None:
        self.move_counter = 0
        self.game_params = game_params
        self.board = np.full(
            (self.game_params.size, self.game_params.size), self.game_params.empty
        )

    def set_game_players(self, player_types: Tuple[str, str]) -> None:
        self.player_types = player_types

    def get_current_player_type(self) -> str:
        return self.player_types[self.move_counter % 2]

    def make_move(self, row: int, column: int) -> bool:
        if self.board[row][column] == self.game_params.empty:
            if self.move_counter % 2 == 0:
                self.board[row][column] = self.game_params.first_player
            else:
                self.board[row][column] = self.game_params.second_player

            self.move_counter += 1
            return True
        else:
            return False

    def continue_game(self) -> None:
        return (
            self.move_counter < self.game_params.size**2
            and self.check_for_winner() == False
        )

    def check_for_winner(self) -> str:
        for player in [self.game_params.first_player, self.game_params.second_player]:
            if np.any(np.all(self.board == player, axis=0)) or np.any(
                np.all(self.board == player, axis=1)
            ):
                return player

            if np.all(np.diag(self.board) == player) or np.all(
                np.diag(np.fliplr(self.board)) == player
            ):
                return player

        return False
