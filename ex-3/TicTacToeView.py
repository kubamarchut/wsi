from typing import Tuple


class TicTacToeView:
    def __init__(self) -> None:
        pass

    def get_player_types(self) -> Tuple[str, str]:
        print("Define player (human or computer h/c)")
        player_1 = input("player 1: ").lower()
        player_2 = input("player 2: ").lower()

        while player_1 not in ["h", "c"] or player_2 not in ["h", "c"]:
            print("invalid input try again")
            player_1 = input("player 1: ").lower()
            player_2 = input("player 2: ").lower()

        return player_1, player_2

    def present_board(self, board) -> None:
        print()
        for row_number, row in enumerate(board):
            print(f" {" | ".join(row)} ")
            if row_number < len(board) - 1:
                row_sep = "---+" * len(row)
                print(row_sep[:-1])
        print()


if __name__ == "__main__":
    board = [
        [" ", "O", " ", "O"],
        ["X", " ", "O", "X"],
        [" ", " ", "X", " "],
        ["X", " ", "O", "X"],
    ]
    view = TicTacToeView()
    view.present_board(board)
