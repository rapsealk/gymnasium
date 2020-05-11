#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple
from enum import Enum

from board import Board, Piece, Move


class C4Piece(Piece, Enum):
    B = "B"
    R = "R"
    E = " "

    @property
    def opposite(self) -> "C4Piece":
        if self == C4Piece.B:
            return C4Piece.R
        elif self == C4Piece.R:
            return C4Piece.B
        return C4Piece.E

    def __str__(self) -> str:
        return self.value


def generate_segments(num_columns: int, num_rows: int, segment_length: int) -> List[List[Tuple[int, int]]]:
    segments = []
    for col in range(num_columns):
        for row in range(num_rows - segment_length + 1):
            segment: List[Tuple[int, int]] = []
            for t in range(segment_length):
                segment.append((col, row+t))
            segments.append(segment)
    for col in range(num_columns - segment_length + 1):
        for row in range(num_rows):
            segment = []
            for t in range(segment_length):
                segment.append((col+t, row))
            segments.append(segment)
    for col in range(num_columns - segment_length + 1):
        for row in range(num_rows - segment_length + 1):
            segment = []
            for t in range(segment_length):
                segment.append((col+t, row+t))
            segments.append(segment)
    for col in range(num_columns - segment_length + 1):
        for row in range(segment_length - 1, num_rows):
            segment = []
            for t in range(segment_length):
                segment.append((col+t, row-t))
            segments.append(segment)
    return segments


class C4Board(Board):
    NUM_ROWS: int = 6
    NUM_COLUMNS: int = 7
    SEGMENT_LENGTH: int = 4
    SEGMENTS: List[List[Tuple[int, int]]] = generate_segments(NUM_COLUMNS, NUM_ROWS, SEGMENT_LENGTH)

    class Column:
        def __init__(self):
            self._container: List[C4Piece] = []

        @property
        def full(self) -> bool:
            return len(self._container) == C4Board.NUM_ROWS

        def push(self, item: C4Piece):
            if self.full:
                raise OverflowError("Cannot get out of range.")
            self._container.append(item)

        def __getitem__(self, index: int) -> C4Piece:
            if index > len(self._container) - 1:
                return C4Piece.E
            return self._container[index]

        def __repr__(self) -> str:
            return repr(self._container)

        def copy(self) -> "C4Board.Column":
            column = C4Board.Column()
            column._container = self._container.copy()
            return column

    def __init__(self, position: Optional[List["C4Board.Column"]] = None, turn: C4Piece = C4Piece.B):
        if position is None:
            self.position = [C4Board.Column() for _ in range(C4Board.NUM_COLUMNS)]
        else:
            self.position = position
        self._turn = turn

    @property
    def turn(self) -> Piece:
        return self._turn

    def move(self, location: Move) -> Board:
        position = self.position.copy()
        for col in range(C4Board.NUM_COLUMNS):
            position[col] = self.position[col].copy()
        position[location].push(self._turn)
        return C4Board(position, self._turn.opposite)

    @property
    def legal_moves(self) -> List[Move]:
        return [Move(col) for col in range(C4Board.NUM_COLUMNS) if not self.position[col].full]

    def _count_segment(self, segment: List[Tuple[int, int]]) -> Tuple[int, int]:
        black_count: int = 0
        red_count: int = 0
        for col, row in segment:
            if self.position[col][row] == C4Piece.B:
                black_count += 1
            elif self.position[col][row] == C4Piece.R:
                red_count += 1
        return black_count, red_count

    @property
    def is_done(self) -> bool:
        for segment in C4Board.SEGMENTS:
            black_count, red_count = self._count_segment(segment)
            if black_count == 4 or red_count == 4:
                return True
        return False

    def _evaluate_segment(self, segment: List[Tuple[int, int]], player: Piece) -> float:
        black_count, red_count = self._count_segment(segment)
        if red_count > 0 and black_count > 0:
            return 0
        count: int = max(red_count, black_count)
        score: float = 0.0
        if count == 2:
            score = 1
        elif count == 3:
            score = 100
        elif count == 4:
            score = 1000000
        color = C4Piece.B
        if red_count > black_count:
            color = C4Piece.R
        if color != player:
            return -score
        return score

    def evaluate(self, player: Piece) -> float:
        # total = 0.0
        # for segment in C4Piece.SEGMENTS:
        #     total += self._evaluate_segment(segment, player)
        # return total
        return sum(list(map(lambda x: self._evaluate_segment(x, player), C4Board.SEGMENTS)))

    def __repr__(self) -> str:
        display = ''
        for row in reversed(range(C4Board.NUM_ROWS)):
            display += '|'
            for col in range(C4Board.NUM_COLUMNS):
                display += '%s|' % self.position[col][row]
            display += '\n'
        return display
