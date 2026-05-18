"""Chess move representation — mailbox indices internally."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Move:
    """A single chess move with metadata for special cases."""

    from_idx: int
    to_idx: int
    piece: int
    captured: Optional[int] = None
    promotion: Optional[int] = None
    is_castling: bool = False
    is_en_passant: bool = False

    @property
    def from_sq(self) -> str:
        from chess_engine.board import idx_to_sq

        return idx_to_sq(self.from_idx)

    @property
    def to_sq(self) -> str:
        from chess_engine.board import idx_to_sq

        return idx_to_sq(self.to_idx)

    @property
    def piece_str(self) -> str:
        from chess_engine.board import PIECE_TO_STR

        return PIECE_TO_STR[self.piece] or ""

    @property
    def captured_str(self) -> Optional[str]:
        from chess_engine.board import PIECE_TO_STR

        if self.captured is None:
            return None
        return PIECE_TO_STR.get(self.captured)

    @property
    def promotion_str(self) -> Optional[str]:
        from chess_engine.board import PIECE_TO_STR

        if self.promotion is None:
            return None
        return PIECE_TO_STR.get(self.promotion)

    def uci(self) -> str:
        """Return move in UCI notation (e.g. e2e4, e7e8q)."""
        from chess_engine.board import piece_type as pt

        promo = ""
        if self.promotion is not None:
            promo = (pt(self.promotion) or "").lower()
        return f"{self.from_sq}{self.to_sq}{promo}"

    def san_key(self) -> str:
        """Stable key for move history / repetition tracking."""
        return self.uci()
