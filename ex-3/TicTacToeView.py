class TicTacToeView:
    def __init__(self) -> None:
        pass

    def present_board(self, board) -> None:
        for row_number, row in enumerate(board):
            print(f" {" | ".join(row)} ")
            if row_number < len(board) - 1:
                row_sep = "---+" * len(row)
                print(row_sep[:-1])


if __name__ == "__main__":
    board = [
        [" ", "O", " ", "O"],
        ["X", " ", "O", "X"],
        [" ", " ", "X", " "],
        ["X", " ", "O", "X"],
    ]
    view = TicTacToeView()
    view.present_board(board)
