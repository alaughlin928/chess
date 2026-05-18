"""FEN/PGN parsing and SAN helpers (outside chess_engine)."""

from __future__ import annotations

import re
from typing import List, Optional

from chess_engine.board import (
    EMPTY,
    FEN_CHAR_TO_PIECE,
    MAILBOX64,
    PIECE_TO_FEN,
    Board,
    piece_type,
)
from chess_engine.legal_moves import find_legal_move, generate_legal_moves
from chess_engine.move import Move


def board_to_fen(board: Board) -> str:
    """Serialize board to FEN string."""
    rows: List[str] = []
    for rank in range(7, -1, -1):
        empty = 0
        row = ""
        for file_idx in range(8):
            p = board.squares[MAILBOX64[rank * 8 + file_idx]]
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
    castling = board.castling_rights if board.castling_rights else "-"
    ep = board.en_passant if board.en_passant else "-"
    return (
        f"{'/'.join(rows)} {board.active_color} {castling} {ep} "
        f"{board.halfmove_clock} {board.fullmove_number}"
    )


def board_from_fen(fen: str) -> Board:
    """Parse FEN into a Board."""
    parts = fen.strip().split()
    if not parts:
        raise ValueError("Empty FEN")
    rank_strs = parts[0].split("/")
    if len(rank_strs) != 8:
        raise ValueError("FEN must have 8 ranks")
    squares: List[List[Optional[str]]] = []
    for rank_str in reversed(rank_strs):
        row: List[Optional[str]] = []
        for ch in rank_str:
            if ch.isdigit():
                row.extend([None] * int(ch))
            else:
                color = "w" if ch.isupper() else "b"
                row.append(f"{color}{ch.upper()}")
        if len(row) != 8:
            raise ValueError("Invalid rank in FEN")
        squares.append(row)
    active = parts[1] if len(parts) > 1 else "w"
    castling = parts[2] if len(parts) > 2 else "-"
    if castling == "-":
        castling = ""
    ep = parts[3] if len(parts) > 3 else "-"
    en_passant = None if ep in ("-", None) else ep
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1
    return Board.from_rank_file_grid(squares, active, castling, en_passant, halfmove, fullmove)


def extract_pgn_movetext(pgn: str) -> str:
    """Strip headers and comments; return movetext."""
    text = pgn.strip()
    if text.startswith("["):
        lines = text.split("\n")
        body: List[str] = []
        in_headers = True
        for line in lines:
            if in_headers and line.strip().startswith("["):
                continue
            in_headers = False
            body.append(line)
        text = "\n".join(body)
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\$\d+", " ", text)
    return text


def tokenize_movetext(movetext: str) -> List[str]:
    """Tokenize PGN movetext into SAN tokens."""
    movetext = movetext.replace("...", " ")
    movetext = re.sub(r"\d+\.+", " ", movetext)
    tokens = re.split(r"\s+", movetext.strip())
    result: List[str] = []
    for tok in tokens:
        if not tok:
            continue
        if tok in ("1-0", "0-1", "1/2-1/2", "*"):
            break
        if re.match(r"^\d+\.?$", tok):
            continue
        result.append(tok.rstrip("+").rstrip("#"))
    return result


def parse_pgn_moves(pgn: str) -> List[Move]:
    """Parse PGN movetext into a list of legal Move objects."""
    tokens = tokenize_movetext(extract_pgn_movetext(pgn))
    board = Board()
    moves: List[Move] = []
    for san in tokens:
        move = _find_move_by_san(board, san)
        if move is None:
            raise ValueError(f"Illegal or unknown move: {san}")
        board.apply_move(move)
        moves.append(move)
    return moves


def _find_move_by_san(board: Board, san: str) -> Optional[Move]:
    """Match SAN against legal moves."""
    san = san.strip()
    if san in ("O-O", "0-0"):
        for m in generate_legal_moves(board):
            if m.is_castling and m.to_sq in ("g1", "g8"):
                return m
    if san in ("O-O-O", "0-0-0"):
        for m in generate_legal_moves(board):
            if m.is_castling and m.to_sq in ("c1", "c8"):
                return m
    for m in generate_legal_moves(board):
        if move_to_san(board, m) == san or move_to_san(board, m).rstrip("+") == san:
            return m
    return None


def move_to_san(board: Board, move: Move) -> str:
    """Convert a move to short algebraic notation."""
    if move.is_castling:
        return "O-O" if move.to_sq in ("g1", "g8") else "O-O-O"
    ptype = piece_type(move.piece)
    capture = bool(move.captured or move.is_en_passant)
    to_sq = move.to_sq
    if ptype == "P":
        if capture:
            return f"{move.from_sq[0]}x{to_sq}" + _promo_suffix(move)
        return f"{to_sq}" + _promo_suffix(move)
    san = {"N": "N", "B": "B", "R": "R", "Q": "Q", "K": "K"}[ptype or "P"]
    san += _disambiguation(board, move, ptype or "P")
    if capture:
        san += "x"
    san += to_sq
    test = board.copy()
    test.apply_move(move)
    if test.is_in_check(test.active_color):
        legal = generate_legal_moves(test)
        if not legal:
            san += "#"
        else:
            san += "+"
    return san


def _promo_suffix(move: Move) -> str:
    if move.promotion:
        return f"={piece_type(move.promotion)}"
    return ""


def _disambiguation(board: Board, move: Move, ptype: str) -> str:
    """File/rank disambiguation when multiple pieces can reach the square."""
    others = [
        m
        for m in generate_legal_moves(board)
        if piece_type(m.piece) == ptype
        and m.to_idx == move.to_idx
        and m.from_idx != move.from_idx
    ]
    if not others:
        return ""
    need_file = any(m.from_sq[0] == move.from_sq[0] for m in others)
    need_rank = any(m.from_sq[1] == move.from_sq[1] for m in others)
    if need_file:
        return move.from_sq[0]
    if need_rank:
        return move.from_sq[1]
    return move.from_sq


def uci_to_move(board: Board, uci: str) -> Optional[Move]:
    """Parse UCI string to a legal move."""
    uci = uci.strip().lower()
    if len(uci) < 4:
        return None
    from_sq, to_sq = uci[:2], uci[2:4]
    promo = uci[4].upper() if len(uci) > 4 else None
    return find_legal_move(board, from_sq, to_sq, promo)


def moves_to_san_list(board: Board, moves: List[Move]) -> List[str]:
    """Convert moves to SAN from a starting board."""
    b = board.copy()
    sans: List[str] = []
    for m in moves:
        sans.append(move_to_san(b, m))
        b.apply_move(m)
    return sans
