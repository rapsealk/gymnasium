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
                 position: List[TicTacToePiece]=[TicTacToePiece.E] * 9,
                 turn: TicTacToePiece=TicTacToePiece.X) -> None:
        self.position = position
        self._turn = turn

    @property
    def turn(self) -> Piece:
        return self._turn
