from TicTacToeModel import TicTacToeModel, GameParams
from TicTacToeView import TicTacToeView

from typing import Tuple


class TicTacToeController:
    def __init__(self, model: TicTacToeModel, view: TicTacToeView) -> None:
        self.model = model
        self.view = view

    def get_competitors(self) -> None:
        self.model.set_game_players(self.view.get_player_types())

    def get_human_player_move(self) -> Tuple[int, int]:
        while True:
            try:
                row = (
                    int(
                        input(
                            f"Enter row number from 1 to {self.model.game_params.size}: "
                        )
                    )
                    - 1
                )
                column = (
                    int(
                        input(
                            f"Enter column number from 1 to {self.model.game_params.size}: "
                        )
                    )
                    - 1
                )

                if (
                    0 <= row < self.model.game_params.size
                    and 0 <= column < self.model.game_params.size
                ):
                    return row, column
                else:
                    print("Invalid input! Please enter values within the board range.")
            except ValueError:
                print("Invalid input! Please enter integer values.")

    def conduct_game(self) -> None:
        self.get_competitors()

        while self.model.continue_game():
            self.view.present_board(self.model.board)

            if self.model.get_current_player_type() != "h":
                self.model.ai_player_move()
                continue
            else:
                move = self.get_human_player_move()
                if self.model.make_move(*move):
                    continue

            print("Invalid move! Try again")

        print("\nThe game has ended!")
        self.view.present_board(self.model.board)

        winner = self.model.check_for_winner()
        if winner != False:
            print(f"Player {winner} won!")
        else:
            print("It is a tie!")


if __name__ == "__main__":
    game_params = GameParams()
    model = TicTacToeModel(game_params)
    view = TicTacToeView()
    controller = TicTacToeController(model, view)

    controller.conduct_game()
