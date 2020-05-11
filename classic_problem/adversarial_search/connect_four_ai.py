#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from minimax import find_best_move
from connect_four import C4Board
from board import Board, Move

board: Board = C4Board()


def get_player_move() -> Move:
    player_move = Move(-1)
    while player_move not in board.legal_moves:
        move = int(input('Move to (0-6): '))
        player_move = Move(move)
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
        computer_move = find_best_move(board, 3)
        print("Computer moved to %d." % computer_move)
        board = board.move(computer_move)
        print(board)
        if board.is_done:
            print("You lose!")
            break
        elif board.is_draw:
            print("Draw!")
            break
