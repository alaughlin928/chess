"""Game status tracking: checkmate, stalemate, draws."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from chess_engine.board import Board
from chess_engine.legal_moves import generate_legal_moves
from chess_engine.move import Move


STATUS_ONGOING = "ongoing"
STATUS_CHECK = "check"
STATUS_CHECKMATE = "checkmate"
STATUS_STALEMATE = "stalemate"
STATUS_DRAW_50 = "draw_50_move"
STATUS_DRAW_REPETITION = "draw_repetition"
STATUS_DRAW_INSUFFICIENT = "draw_insufficient_material"


@dataclass
class GameState:
    """Full game: board, move history, repetition tracking, status."""

    board: Board = field(default_factory=Board)
    move_history: List[Move] = field(default_factory=list)
    position_counts: Dict[str, int] = field(default_factory=dict)
    status: str = STATUS_ONGOING
    winner: Optional[str] = None  # 'w', 'b', or None for draw

    def __post_init__(self) -> None:
        if not self.position_counts:
            self._record_position()

    def _record_position(self) -> None:
        key = self.board.position_key()
        self.position_counts[key] = self.position_counts.get(key, 0) + 1

    def reset(self) -> None:
        """Reset to starting position."""
        self.board = Board()
        self.move_history = []
        self.position_counts = {}
        self.status = STATUS_ONGOING
        self.winner = None
        self._record_position()

    def apply_move(self, move: Move) -> None:
        """Apply a legal move and update game status."""
        self.board.apply_move(move)
        self.move_history.append(move)
        self._record_position()
        self._update_status()

    def _update_status(self) -> None:
        """Recompute status after the last move."""
        color = self.board.active_color
        legal = generate_legal_moves(self.board)
        in_check = self.board.is_in_check(color)

        if self.board.halfmove_clock >= 100:
            self.status = STATUS_DRAW_50
            self.winner = None
            return

        key = self.board.position_key()
        if self.position_counts.get(key, 0) >= 3:
            self.status = STATUS_DRAW_REPETITION
            self.winner = None
            return

        if _insufficient_material(self.board):
            self.status = STATUS_DRAW_INSUFFICIENT
            self.winner = None
            return

        if not legal:
            if in_check:
                self.status = STATUS_CHECKMATE
                self.winner = "b" if color == "w" else "w"
            else:
                self.status = STATUS_STALEMATE
                self.winner = None
            return

        self.status = STATUS_CHECK if in_check else STATUS_ONGOING
        self.winner = None

    def is_game_over(self) -> bool:
        """Return True if the game has ended."""
        return self.status not in (STATUS_ONGOING, STATUS_CHECK)

    def to_dict(self) -> dict:
        """Serialize full game state for API."""
        return {
            **self.board.to_dict(),
            "status": self.status,
            "winner": self.winner,
            "in_check": self.board.is_in_check(self.board.active_color),
            "legal_moves": [_move_to_api(m) for m in generate_legal_moves(self.board)],
            "move_history": [m.uci() for m in self.move_history],
        }


def _move_to_api(move: Move) -> dict:
    from chess_engine.board import piece_type

    promo = piece_type(move.promotion) if move.promotion else None
    return {
        "from": move.from_sq,
        "to": move.to_sq,
        "promotion": promo,
    }


def _insufficient_material(board: Board) -> bool:
    """Detect K vs K and other insufficient mating material cases."""
    from chess_engine.board import (
        EMPTY,
        IS_BLACK,
        IS_WHITE,
        MAILBOX64,
        bB,
        bK,
        bN,
        wB,
        wK,
        wN,
        piece_type,
    )

    pieces: List[int] = []
    bishops: List[tuple] = []
    for sq64 in range(64):
        p = board.squares[MAILBOX64[sq64]]
        if p == EMPTY or p in (wK, bK):
            continue
        pieces.append(p)
        pt = piece_type(p)
        if pt == "B":
            color = "w" if IS_WHITE(p) else "b"
            bishops.append((color, (sq64 // 8 + sq64 % 8) % 2))
        elif pt == "N":
            pass

    if not pieces:
        return True
    if len(pieces) == 1 and piece_type(pieces[0]) in ("B", "N"):
        return True
    if len(pieces) == 2 and piece_type(pieces[0]) == "B" and piece_type(pieces[1]) == "B":
        if bishops[0][0] != bishops[1][0]:
            return False
        return bishops[0][1] == bishops[1][1]
    return False
