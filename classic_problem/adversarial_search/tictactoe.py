#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from typing import List
from enum import Enum

from board import Piece, Board, Move


class TicTacToePiece(Piece, Enum):
    O = "O"
    X = "X"
    E = " "

    @property
    def opposite(self) -> "TicTacToePiece":
        if self == TicTacToePiece.X:
            return TicTacToePiece.O
        elif self == TicTacToePiece.O:
            return TicTacToePiece.X
        else:
            return TicTacToePiece.E

    def __str__(self) -> str:
        return self.value


class TicTacToeBoard(Board):
    def __init__(self,
                 position: List[TicTacToePiece] = [TicTacToePiece.E] * 9,
                 turn: TicTacToePiece = TicTacToePiece.X) -> None:
        self.position = position
        self._turn = turn

    @property
    def turn(self) -> Piece:
        return self._turn

    def move(self, location: Move) -> Board:
        position = self.position.copy()
        position[location] = self._turn
        return TicTacToeBoard(position, self._turn.opposite)

    @property
    def legal_moves(self) -> List[Move]:
        return [Move(l) for l in range(len(self.position)) if self.position[l] == TicTacToePiece.E]

    @property
    def is_done(self) -> bool:
        return self.position[0] == self.position[1] and self.position[0] == self.position[2] and self.position[0] != TicTacToePiece.E or \
                self.position[3] == self.position[4] and self.position[3] == self.position[5] and self.position[3] != TicTacToePiece.E or \
                self.position[6] == self.position[7] and self.position[6] == self.position[8] and self.position[6] != TicTacToePiece.E or \
                self.position[0] == self.position[3] and self.position[0] == self.position[6] and self.position[0] != TicTacToePiece.E or \
                self.position[1] == self.position[4] and self.position[1] == self.position[7] and self.position[1] != TicTacToePiece.E or \
                self.position[2] == self.position[5] and self.position[2] == self.position[8] and self.position[2] != TicTacToePiece.E or \
                self.position[0] == self.position[4] and self.position[0] == self.position[8] and self.position[0] != TicTacToePiece.E or \
                self.position[2] == self.position[4] and self.position[2] == self.position[6] and self.position[2] != TicTacToePiece.E

    def evaluate(self, player: Piece) -> float:
        if self.is_done and self.turn == player:
            return -1
        elif self.is_done and self.turn != player:
            return 1
        else:
            return 0

    def __repr__(self) -> str:
        return "%s|%s|%s\n---\n%s|%s|%s\n---\n%s|%s|%s" % tuple(self.position)
