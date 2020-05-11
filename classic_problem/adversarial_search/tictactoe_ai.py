#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from minimax import find_best_move
from tictactoe import TicTacToeBoard
from board import Move, Board

board: Board = TicTacToeBoard()


def get_player_move() -> Move:
    player_move: Move = Move(-1)
    while player_move not in board.legal_moves:
        play: int = int(input("Move to (0-8): "))
        player_move = Move(play)
    return player_move


if __name__ == "__main__":
    while True:
        human_move = get_player_move()
        board = board.move(human_move)
        if board.is_done:
            print("You win!")
            break
        elif board.is_draw:
            print("Draw!")
            break
        computer_move = find_best_move(board)
        print("Computer moved to %d." % computer_move)
        board = board.move(computer_move)
        print(board)
        if board.is_done:
            print("You lose!")
            break
        elif board.is_draw:
            print("Draw!")
            break
