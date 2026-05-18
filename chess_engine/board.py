"""Chess board state — 120-square mailbox, integer piece encoding."""

from __future__ import annotations

import functools
from typing import Dict, List, Optional, Tuple

# Piece encoding
EMPTY = 0
wP, wN, wB, wR, wQ, wK = 1, 2, 3, 4, 5, 6
bP, bN, bB, bR, bQ, bK = 9, 10, 11, 12, 13, 14
OFF = 255

FILES = "abcdefgh"

PIECE_TO_STR: Dict[int, Optional[str]] = {
    EMPTY: None,
    wP: "wP",
    wN: "wN",
    wB: "wB",
    wR: "wR",
    wQ: "wQ",
    wK: "wK",
    bP: "bP",
    bN: "bN",
    bB: "bB",
    bR: "bR",
    bQ: "bQ",
    bK: "bK",
}

STR_TO_PIECE: Dict[str, int] = {v: k for k, v in PIECE_TO_STR.items() if v}

FEN_CHAR_TO_PIECE: Dict[str, int] = {
    "P": wP,
    "N": wN,
    "B": wB,
    "R": wR,
    "Q": wQ,
    "K": wK,
    "p": bP,
    "n": bN,
    "b": bB,
    "r": bR,
    "q": bQ,
    "k": bK,
}

PIECE_TO_FEN: Dict[int, str] = {
    wP: "P",
    wN: "N",
    wB: "B",
    wR: "R",
    wQ: "Q",
    wK: "K",
    bP: "p",
    bN: "n",
    bB: "b",
    bR: "r",
    bQ: "q",
    bK: "k",
}

# Rank-major 0..63 (a1=0 … h8=63) → mailbox index
MAILBOX64: List[int] = [91 - 10 * (i // 8) + (i % 8) for i in range(64)]

# Mailbox index → rank-major 0..63, or -1 if off-board
MAILBOX120: List[int] = [-1] * 120
for _sq64, _idx in enumerate(MAILBOX64):
    MAILBOX120[_idx] = _sq64

# Castling / king mailbox indices (e1=95, g1=97, c1=93, a1=91, h1=98, f1=96, d1=94)
IDX_E1, IDX_G1, IDX_C1, IDX_A1, IDX_H1, IDX_F1, IDX_D1 = 95, 97, 93, 91, 98, 96, 94
IDX_E8, IDX_G8, IDX_C8, IDX_A8, IDX_H8, IDX_F8, IDX_D8 = 25, 27, 23, 21, 28, 26, 24
IDX_B1, IDX_B8 = MAILBOX64[1], MAILBOX64[57]

WHITE_KING_START = IDX_E1
BLACK_KING_START = IDX_E8


def IS_WHITE(p: int) -> bool:
    return 1 <= p <= 6


def IS_BLACK(p: int) -> bool:
    return 9 <= p <= 14


def COLOR(p: int) -> Optional[str]:
    if IS_WHITE(p):
        return "w"
    if IS_BLACK(p):
        return "b"
    return None


def piece_type(p: int) -> Optional[str]:
    """Return piece type letter P,N,B,R,Q,K."""
    if p == EMPTY:
        return None
    if IS_WHITE(p):
        return "PNBRQK"[p - 1]
    return "PNBRQK"[p - 9]


def _empty_mailbox() -> List[int]:
    """Return a 120-square mailbox filled with OFF borders."""
    mb = [OFF] * 120
    for sq64 in range(64):
        mb[MAILBOX64[sq64]] = EMPTY
    return mb


def _set_startpos(mb: List[int]) -> None:
    """Place standard starting position on mailbox."""
    back_w = [wR, wN, wB, wQ, wK, wB, wN, wR]
    back_b = [bR, bN, bB, bQ, bK, bB, bN, bR]
    for f in range(8):
        mb[MAILBOX64[f]] = back_w[f]
        mb[MAILBOX64[8 + f]] = wP
        mb[MAILBOX64[48 + f]] = bP
        mb[MAILBOX64[56 + f]] = back_b[f]


@functools.lru_cache(maxsize=64)
def sq_to_coords(square: str) -> Tuple[int, int]:
    """Convert algebraic square (e.g. 'e4') to (file, rank) indices."""
    file_idx = FILES.index(square[0])
    rank_idx = int(square[1]) - 1
    return file_idx, rank_idx


@functools.lru_cache(maxsize=64)
def coords_to_sq(file_idx: int, rank_idx: int) -> str:
    """Convert (file, rank) indices to algebraic square."""
    return f"{FILES[file_idx]}{rank_idx + 1}"


def sq_to_idx(square: str) -> int:
    """Convert algebraic square to mailbox index."""
    f, r = sq_to_coords(square)
    return MAILBOX64[r * 8 + f]


def idx_to_sq(idx: int) -> str:
    """Convert mailbox index to algebraic square."""
    sq64 = MAILBOX120[idx]
    if sq64 < 0:
        raise ValueError(f"Off-board mailbox index: {idx}")
    return coords_to_sq(sq64 % 8, sq64 // 8)


def in_bounds(file_idx: int, rank_idx: int) -> bool:
    """Return True if coordinates are on the board (API / FEN only)."""
    return 0 <= file_idx < 8 and 0 <= rank_idx < 8


class Board:
    """8x8 chess position using a 120-square mailbox."""

    def __init__(
        self,
        mailbox: Optional[List[int]] = None,
        active_color: str = "w",
        castling_rights: str = "KQkq",
        en_passant: Optional[str] = None,
        halfmove_clock: int = 0,
        fullmove_number: int = 1,
    ) -> None:
        if mailbox is not None:
            self.squares = mailbox[:]
        else:
            self.squares = _empty_mailbox()
            _set_startpos(self.squares)
        self.active_color = active_color
        self.castling_rights = castling_rights
        self.en_passant = en_passant
        self.en_passant_idx: Optional[int] = (
            sq_to_idx(en_passant) if en_passant else None
        )
        self.halfmove_clock = halfmove_clock
        self.fullmove_number = fullmove_number
        self.white_king_sq = WHITE_KING_START
        self.black_king_sq = BLACK_KING_START
        self._init_king_squares()

    @classmethod
    def from_rank_file_grid(
        cls,
        grid: List[List[Optional[str]]],
        active_color: str = "w",
        castling_rights: str = "KQkq",
        en_passant: Optional[str] = None,
        halfmove_clock: int = 0,
        fullmove_number: int = 1,
    ) -> "Board":
        """Build board from 8x8 string grid (rank 0 = white back rank)."""
        mb = _empty_mailbox()
        for r in range(8):
            for f in range(8):
                pstr = grid[r][f]
                mb[MAILBOX64[r * 8 + f]] = STR_TO_PIECE[pstr] if pstr else EMPTY
        return cls(mb, active_color, castling_rights, en_passant, halfmove_clock, fullmove_number)

    def _init_king_squares(self) -> None:
        """Cache king mailbox indices."""
        self.white_king_sq = WHITE_KING_START
        self.black_king_sq = BLACK_KING_START
        for sq64 in range(64):
            idx = MAILBOX64[sq64]
            p = self.squares[idx]
            if p == wK:
                self.white_king_sq = idx
            elif p == bK:
                self.black_king_sq = idx

    def copy(self) -> "Board":
        """Return a copy of this board with independent state."""
        b = object.__new__(Board)
        b.squares = self.squares[:]
        b.active_color = self.active_color
        b.castling_rights = self.castling_rights
        b.en_passant = self.en_passant
        b.en_passant_idx = self.en_passant_idx
        b.halfmove_clock = self.halfmove_clock
        b.fullmove_number = self.fullmove_number
        b.white_king_sq = self.white_king_sq
        b.black_king_sq = self.black_king_sq
        return b

    def piece_at_idx(self, idx: int) -> int:
        """Return piece code at mailbox index."""
        return self.squares[idx]

    def piece_at(self, square: str) -> Optional[str]:
        """Return piece string at square (e.g. 'wK') or None."""
        return PIECE_TO_STR.get(self.squares[sq_to_idx(square)], None)

    def set_piece(self, square: str, piece: Optional[str]) -> None:
        """Place piece on square (None clears the square)."""
        code = STR_TO_PIECE[piece] if piece else EMPTY
        self.squares[sq_to_idx(square)] = code

    def find_king(self, color: str) -> str:
        """Return algebraic square of the king for the given color."""
        idx = self.white_king_sq if color == "w" else self.black_king_sq
        return idx_to_sq(idx)

    def find_king_idx(self, color: str) -> int:
        """Return mailbox index of the king."""
        return self.white_king_sq if color == "w" else self.black_king_sq

    def color_of_idx(self, idx: int) -> Optional[str]:
        return COLOR(self.squares[idx])

    def color_of(self, square: str) -> Optional[str]:
        return COLOR(self.squares[sq_to_idx(square)])

    def opponent(self, color: str) -> str:
        return "b" if color == "w" else "w"

    def is_square_attacked(self, square: str, by_color: str) -> bool:
        from chess_engine.pieces import attacks_square

        return attacks_square(self, sq_to_idx(square), by_color)

    def is_square_attacked_idx(self, idx: int, by_color: str) -> bool:
        from chess_engine.pieces import attacks_square

        return attacks_square(self, idx, by_color)

    def is_in_check(self, color: str) -> bool:
        king_idx = self.find_king_idx(color)
        return self.is_square_attacked_idx(king_idx, self.opponent(color))

    def position_key(self) -> str:
        """Hashable key for threefold repetition."""
        rows: List[str] = []
        for rank in range(7, -1, -1):
            empty = 0
            row = ""
            for file_idx in range(8):
                p = self.squares[MAILBOX64[rank * 8 + file_idx]]
                if p == EMPTY:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += PIECE_TO_FEN[p]
            if empty:
                row += str(empty)
            rows.append(row)
        ep = self.en_passant or "-"
        return "|".join(
            [
                "/".join(rows),
                self.active_color,
                self.castling_rights or "-",
                ep,
            ]
        )

    def to_dict(self) -> Dict:
        """Serialize board for JSON API."""
        board_dict: Dict[str, Optional[str]] = {}
        for sq64 in range(64):
            sq = coords_to_sq(sq64 % 8, sq64 // 8)
            board_dict[sq] = PIECE_TO_STR.get(self.squares[MAILBOX64[sq64]], None)
        return {
            "board": board_dict,
            "active_color": self.active_color,
            "castling_rights": self.castling_rights,
            "en_passant": self.en_passant,
            "halfmove_clock": self.halfmove_clock,
            "fullmove_number": self.fullmove_number,
        }

    def apply_move(self, move: "Move") -> None:
        """Apply move to board (does not validate legality)."""
        piece = move.piece
        color = COLOR(piece) or self.active_color
        from_idx = move.from_idx
        to_idx = move.to_idx
        moving = self.squares[from_idx]
        if moving == EMPTY:
            moving = piece

        self.squares[from_idx] = EMPTY

        if move.is_castling:
            if to_idx == IDX_G1:
                self.squares[IDX_H1] = EMPTY
                self.squares[IDX_F1] = wR
            elif to_idx == IDX_C1:
                self.squares[IDX_A1] = EMPTY
                self.squares[IDX_D1] = wR
            elif to_idx == IDX_G8:
                self.squares[IDX_H8] = EMPTY
                self.squares[IDX_F8] = bR
            elif to_idx == IDX_C8:
                self.squares[IDX_A8] = EMPTY
                self.squares[IDX_D8] = bR
            self.squares[to_idx] = moving
        elif move.is_en_passant:
            cap_idx = from_idx + (10 if color == "w" else -10)
            self.squares[cap_idx] = EMPTY
            self.squares[to_idx] = moving
        else:
            self.squares[to_idx] = move.promotion if move.promotion else moving

        if moving == wK:
            self.white_king_sq = to_idx
        elif moving == bK:
            self.black_king_sq = to_idx

        rights = set(self.castling_rights)
        for idx in (from_idx, to_idx):
            if idx == IDX_E1:
                rights -= {"K", "Q"}
            elif idx in (IDX_A1, IDX_H1):
                rights -= {"Q"} if idx == IDX_A1 else {"K"}
            elif idx == IDX_E8:
                rights -= {"k", "q"}
            elif idx in (IDX_A8, IDX_H8):
                rights -= {"q"} if idx == IDX_A8 else {"k"}
        self.castling_rights = "".join(r for r in "KQkq" if r in rights)

        self.en_passant = None
        self.en_passant_idx = None
        if moving == wP or moving == bP:
            delta = to_idx - from_idx
            if delta == -20 or delta == 20:
                self.en_passant_idx = (from_idx + to_idx) // 2
                self.en_passant = idx_to_sq(self.en_passant_idx)

        captured = move.captured
        reset_halfmove = (
            moving == wP
            or moving == bP
            or captured is not None
            or move.is_en_passant
        )
        self.halfmove_clock = 0 if reset_halfmove else self.halfmove_clock + 1

        if self.active_color == "b":
            self.fullmove_number += 1
        self.active_color = self.opponent(color)
