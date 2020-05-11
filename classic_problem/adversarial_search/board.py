#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from typing import NewType, List
from abc import ABC, abstractmethod

Move = NewType("Move", int)


class Piece:
    @property
    def opposite(self) -> "Piece":
        raise NotImplementedError()


class Board(ABC):
    @property
    @abstractmethod
    def turn(self) -> Piece:
        ...

    @abstractmethod
    def move(self, location: Move) -> "Board":
        ...

    @property
    @abstractmethod
    def legal_moves(self) -> List[Move]:
        ...

    @property
    @abstractmethod
    def is_done(self) -> bool:
        ...

    @property
    def is_draw(self) -> bool:
        return (not self.is_done) and (len(self.legal_moves) == 0)

    @abstractmethod
    def evaluate(self, player: Piece) -> float:
        ...
