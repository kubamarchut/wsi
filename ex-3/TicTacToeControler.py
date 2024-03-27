from TicTacToeModel import TicTacToeModel, GameParams
from TicTacToeView import TicTacToeView


class TicTacToeController:
    def __init__(self, model: TicTacToeModel, view: TicTacToeView) -> None:
        self.model = model
        self.view = view

    def conduct_game(self):
        while self.model.continue_game():
            self.view.present_board(self.model.board)
            row = int(
                input(f"Enter row number from 1 to {self.model.game_params.size}: ")
            )
            column = int(
                input(f"Enter column number from 1 to {self.model.game_params.size}: ")
            )
            row -= 1
            column -= 1
            if (row >= 0 and row < self.model.game_params.size) and (
                column >= 0 and column < self.model.game_params.size
            ):
                if self.model.make_move(row, column):
                    continue

            print("Invalid move! Try again")

        print("\nThe game has ended!")
        self.view.present_board(self.model.board)

        winner = self.model.check_for_winner()
        if winner != False:
            print(f"Player {winner} won!")


if __name__ == "__main__":
    game_params = GameParams()
    model = TicTacToeModel(game_params)
    view = TicTacToeView()
    controller = TicTacToeController(model, view)

    controller.conduct_game()
