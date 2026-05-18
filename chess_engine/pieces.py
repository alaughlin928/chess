"""Pseudo-legal move generation — mailbox integer indices."""

from __future__ import annotations

from typing import List, Optional

from chess_engine.board import (
    EMPTY,
    IDX_A1,
    IDX_A8,
    IDX_B1,
    IDX_B8,
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
    IDX_H1,
    IDX_H8,
    IS_BLACK,
    IS_WHITE,
    MAILBOX64,
    OFF,
    bB,
    bK,
    bN,
    bP,
    bQ,
    bR,
    wB,
    wK,
    wN,
    wP,
    wQ,
    wR,
    Board,
)
from chess_engine.move import Move

# Mailbox direction deltas (north = toward rank 8 = -10 on board)
N, S, E, W = -10, 10, 1, -1
KNIGHT_OFFSETS = (-19, -21, -8, -12, 8, 12, 19, 21)
BISHOP_OFFSETS = (-11, -9, 9, 11)
ROOK_OFFSETS = (-10, 10, -1, 1)
KING_OFFSETS = BISHOP_OFFSETS + ROOK_OFFSETS

PROMO_WHITE = (wQ, wR, wB, wN)
PROMO_BLACK = (bQ, bR, bB, bN)

ACTIVE_TO_COLOR = {"w": True, "b": False}


def _make_move(
    from_idx: int,
    to_idx: int,
    piece: int,
    captured: Optional[int] = None,
    promotion: Optional[int] = None,
    is_castling: bool = False,
    is_en_passant: bool = False,
) -> Move:
    return Move(
        from_idx=from_idx,
        to_idx=to_idx,
        piece=piece,
        captured=captured,
        promotion=promotion,
        is_castling=is_castling,
        is_en_passant=is_en_passant,
    )


def _is_empty(board: Board, idx: int) -> bool:
    return board.squares[idx] == EMPTY


def _can_capture(board: Board, idx: int, white: bool) -> bool:
    p = board.squares[idx]
    if p == EMPTY or p == OFF:
        return False
    return IS_WHITE(p) if not white else IS_BLACK(p)


def generate_pawn_moves(board: Board, from_idx: int, piece: int) -> List[Move]:
    """Generate pseudo-legal pawn moves including double push, EP, promotion."""
    moves: List[Move] = []
    white = piece == wP
    push = -10 if white else 10
    start_rank_mailbox = 81 if white else 31  # rank 2 / rank 7 start squares
    promo_rank_mailbox = {21, 22, 23, 24, 25, 26, 27, 28} if white else {91, 92, 93, 94, 95, 96, 97, 98}
    cap_dirs = (-11, -9) if white else (11, 9)

    def try_promotions(to_idx: int, cap: Optional[int], ep: bool = False) -> None:
        if to_idx in promo_rank_mailbox:
            for promo in PROMO_WHITE if white else PROMO_BLACK:
                moves.append(_make_move(from_idx, to_idx, piece, cap, promo, is_en_passant=ep))
        else:
            moves.append(_make_move(from_idx, to_idx, piece, cap, is_en_passant=ep))

    one = from_idx + push
    if board.squares[one] == EMPTY:
        try_promotions(one, None)
        if from_idx in range(start_rank_mailbox, start_rank_mailbox + 8):
            two = from_idx + 2 * push
            if board.squares[two] == EMPTY:
                moves.append(_make_move(from_idx, two, piece))

    for d in cap_dirs:
        tgt = from_idx + d
        if board.squares[tgt] != OFF and _can_capture(board, tgt, white):
            try_promotions(tgt, board.squares[tgt])

    ep = board.en_passant_idx
    if ep is not None:
        for d in cap_dirs:
            if from_idx + d == ep:
                cap_idx = ep + (10 if white else -10)
                cap_piece = board.squares[cap_idx]
                try_promotions(ep, cap_piece, ep=True)
                break

    return moves


def generate_sliding_moves(
    board: Board, from_idx: int, piece: int, offsets: tuple
) -> List[Move]:
    """Generate pseudo-legal moves for bishop/rook/queen."""
    moves: List[Move] = []
    white = IS_WHITE(piece)
    for d in offsets:
        tgt = from_idx + d
        while board.squares[tgt] != OFF:
            target = board.squares[tgt]
            if target == EMPTY:
                moves.append(_make_move(from_idx, tgt, piece))
            else:
                if (IS_WHITE(target) and not white) or (IS_BLACK(target) and white):
                    moves.append(_make_move(from_idx, tgt, piece, target))
                break
            tgt += d
    return moves


def generate_knight_moves(board: Board, from_idx: int, piece: int) -> List[Move]:
    """Generate pseudo-legal knight moves."""
    moves: List[Move] = []
    white = IS_WHITE(piece)
    for d in KNIGHT_OFFSETS:
        tgt = from_idx + d
        if board.squares[tgt] == OFF:
            continue
        target = board.squares[tgt]
        if target == EMPTY:
            moves.append(_make_move(from_idx, tgt, piece))
        elif (IS_WHITE(target) and not white) or (IS_BLACK(target) and white):
            moves.append(_make_move(from_idx, tgt, piece, target))
    return moves


def generate_king_moves(board: Board, from_idx: int, piece: int) -> List[Move]:
    """Generate pseudo-legal king moves including castling."""
    moves: List[Move] = []
    white = piece == wK
    for d in KING_OFFSETS:
        tgt = from_idx + d
        if board.squares[tgt] == OFF:
            continue
        target = board.squares[tgt]
        if target == EMPTY:
            moves.append(_make_move(from_idx, tgt, piece))
        elif (IS_WHITE(target) and not white) or (IS_BLACK(target) and white):
            moves.append(_make_move(from_idx, tgt, piece, target))

    if white and from_idx == IDX_E1:
        if "K" in board.castling_rights and board.squares[IDX_F1] == EMPTY and board.squares[IDX_G1] == EMPTY:
            if board.squares[IDX_H1] == wR:
                moves.append(_make_move(IDX_E1, IDX_G1, wK, is_castling=True))
        if "Q" in board.castling_rights and board.squares[IDX_D1] == EMPTY and board.squares[IDX_C1] == EMPTY and board.squares[IDX_B1] == EMPTY:
            if board.squares[IDX_A1] == wR:
                moves.append(_make_move(IDX_E1, IDX_C1, wK, is_castling=True))
    elif not white and from_idx == IDX_E8:
        if "k" in board.castling_rights and board.squares[IDX_F8] == EMPTY and board.squares[IDX_G8] == EMPTY:
            if board.squares[IDX_H8] == bR:
                moves.append(_make_move(IDX_E8, IDX_G8, bK, is_castling=True))
        if "q" in board.castling_rights and board.squares[IDX_D8] == EMPTY and board.squares[IDX_C8] == EMPTY and board.squares[IDX_B8] == EMPTY:
            if board.squares[IDX_A8] == bR:
                moves.append(_make_move(IDX_E8, IDX_C8, bK, is_castling=True))

    return moves


def generate_piece_moves(board: Board, from_idx: int) -> List[Move]:
    """Generate all pseudo-legal moves for the piece at mailbox index."""
    piece = board.squares[from_idx]
    if piece == EMPTY:
        return []
    white_piece = IS_WHITE(piece)
    if ACTIVE_TO_COLOR[board.active_color] != white_piece:
        return []

    if piece in (wP, bP):
        return generate_pawn_moves(board, from_idx, piece)
    if piece in (wN, bN):
        return generate_knight_moves(board, from_idx, piece)
    if piece in (wB, bB):
        return generate_sliding_moves(board, from_idx, piece, BISHOP_OFFSETS)
    if piece in (wR, bR):
        return generate_sliding_moves(board, from_idx, piece, ROOK_OFFSETS)
    if piece in (wQ, bQ):
        return generate_sliding_moves(board, from_idx, piece, BISHOP_OFFSETS + ROOK_OFFSETS)
    if piece in (wK, bK):
        return generate_king_moves(board, from_idx, piece)
    return []


def generate_pseudo_legal_moves(board: Board) -> List[Move]:
    """Generate all pseudo-legal moves for the side to move."""
    moves: List[Move] = []
    for sq64 in range(64):
        idx = MAILBOX64[sq64]
        piece = board.squares[idx]
        if piece == EMPTY:
            continue
        if ACTIVE_TO_COLOR[board.active_color] == IS_WHITE(piece):
            moves.extend(generate_piece_moves(board, idx))
    return moves


def attacks_square(board: Board, target_idx: int, by_color: str) -> bool:
    """Return True if `by_color` attacks `target_idx`."""
    mb = board.squares
    if by_color == "w":
        for d in (11, 9):
            p = mb[target_idx + d]
            if p == wP:
                return True
    else:
        for d in (-11, -9):
            p = mb[target_idx + d]
            if p == bP:
                return True

    for d in KNIGHT_OFFSETS:
        p = mb[target_idx + d]
        if by_color == "w" and p == wN:
            return True
        if by_color == "b" and p == bN:
            return True

    for d in KING_OFFSETS:
        p = mb[target_idx + d]
        if by_color == "w" and p == wK:
            return True
        if by_color == "b" and p == bK:
            return True

    for d in BISHOP_OFFSETS:
        tgt = target_idx + d
        while mb[tgt] != OFF:
            p = mb[tgt]
            if p != EMPTY:
                if by_color == "w" and p in (wB, wQ):
                    return True
                if by_color == "b" and p in (bB, bQ):
                    return True
                break
            tgt += d

    for d in ROOK_OFFSETS:
        tgt = target_idx + d
        while mb[tgt] != OFF:
            p = mb[tgt]
            if p != EMPTY:
                if by_color == "w" and p in (wR, wQ):
                    return True
                if by_color == "b" and p in (bR, bQ):
                    return True
                break
            tgt += d

    return False
