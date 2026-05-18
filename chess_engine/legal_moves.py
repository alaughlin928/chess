"""Filter pseudo-legal moves to legal moves (king not left in check)."""

from __future__ import annotations

from typing import List

from chess_engine.board import (
    COLOR,
    IDX_C1,
    IDX_C8,
    IDX_D1,
    IDX_D8,
    IDX_E1,
    IDX_E8,
    IDX_F1,
    IDX_F8,
    IDX_G1,
    IDX_G8,
    STR_TO_PIECE,
    Board,
    wQ,
    bQ,
)
from chess_engine.move import Move
from chess_engine.pieces import generate_pseudo_legal_moves


def _castling_passes_through_check(board: Board, move: Move) -> bool:
    """Return True if castling move passes through or into check."""
    color = COLOR(move.piece)
    if color == "w":
        if move.to_idx == IDX_G1:
            return board.is_square_attacked_idx(IDX_E1, "b") or board.is_square_attacked_idx(
                IDX_F1, "b"
            )
        if move.to_idx == IDX_C1:
            return board.is_square_attacked_idx(IDX_E1, "b") or board.is_square_attacked_idx(
                IDX_D1, "b"
            )
    else:
        if move.to_idx == IDX_G8:
            return board.is_square_attacked_idx(IDX_E8, "w") or board.is_square_attacked_idx(
                IDX_F8, "w"
            )
        if move.to_idx == IDX_C8:
            return board.is_square_attacked_idx(IDX_E8, "w") or board.is_square_attacked_idx(
                IDX_D8, "w"
            )
    return False


def is_legal_move(board: Board, move: Move) -> bool:
    """Return True if `move` is legal in the current position."""
    color = board.active_color
    from_piece = board.piece_at(move.from_sq)
    if from_piece is None or from_piece[0] != color:
        return False

    pseudo = generate_pseudo_legal_moves(board)
    matching = [
        m for m in pseudo if m.from_idx == move.from_idx and m.to_idx == move.to_idx
    ]
    if not matching:
        return False

    if move.promotion:
        matching = [m for m in matching if m.promotion == move.promotion]
    else:
        queen_promo = wQ if color == "w" else bQ
        for m in matching:
            if m.promotion == queen_promo:
                matching = [m]
                break

    if not matching:
        return False
    candidate = matching[0]
    if move.promotion:
        promo_matches = [m for m in matching if m.promotion == move.promotion]
        if promo_matches:
            candidate = promo_matches[0]

    if candidate.is_castling:
        if board.is_in_check(color):
            return False
        if _castling_passes_through_check(board, candidate):
            return False

    test_board = board.copy()
    test_board.apply_move(candidate)
    return not test_board.is_in_check(color)


def generate_legal_moves(board: Board) -> List[Move]:
    """Generate all legal moves for the side to move."""
    color = board.active_color
    legal: List[Move] = []
    for move in generate_pseudo_legal_moves(board):
        if move.is_castling:
            if board.is_in_check(color):
                continue
            if _castling_passes_through_check(board, move):
                continue
        test_board = board.copy()
        test_board.apply_move(move)
        if not test_board.is_in_check(color):
            legal.append(move)
    return legal


def find_legal_move(
    board: Board, from_sq: str, to_sq: str, promotion: str | None = None
) -> Move | None:
    """Find a legal move matching from/to (and optional promotion piece like 'Q')."""
    matches = [
        m
        for m in generate_legal_moves(board)
        if m.from_sq == from_sq and m.to_sq == to_sq
    ]
    if not matches:
        return None
    if promotion:
        promo_piece = STR_TO_PIECE[f"{board.active_color}{promotion.upper()}"]
        for move in matches:
            if move.promotion == promo_piece:
                return move
        return None
    queen_promo = wQ if board.active_color == "w" else bQ
    for move in matches:
        if move.promotion == queen_promo:
            return move
    return matches[0]
