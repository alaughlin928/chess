"""Chess AI: minimax, alpha-beta, iterative deepening, quiescence, transposition table."""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, TypedDict

from chess_engine.board import (
    EMPTY,
    FILES,
    IDX_A1,
    IDX_A8,
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
    IS_WHITE,
    MAILBOX64,
    MAILBOX120,
    bP,
    bR,
    idx_to_sq,
    wP,
    wR,
    Board,
    piece_type,
)
from chess_engine.game_state import GameState
from chess_engine.legal_moves import generate_legal_moves
from chess_engine.move import Move

# Zobrist hashing (64-bit keys for transposition table)
random.seed(0x5EED)
ZOBRIST_PIECES: List[List[List[int]]] = [
    [[random.getrandbits(64) for _ in range(15)] for _ in range(8)] for _ in range(8)
]
ZOBRIST_SIDE = random.getrandbits(64)
ZOBRIST_CASTLING: List[int] = [random.getrandbits(64) for _ in range(16)]
ZOBRIST_EP: List[int] = [random.getrandbits(64) for _ in range(9)]

TT_MAX_SIZE = 1_000_000

# Material values (centipawns)
PIECE_VALUES = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 20000}

# Piece-square tables (white perspective; rank 0 = white's 1st rank)
PAWN_PST = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0],
]

KNIGHT_PST = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
]

BISHOP_PST = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
]

ROOK_PST = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [0, 0, 0, 5, 5, 0, 0, 0],
]

QUEEN_PST = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20],
]

KING_MG_PST = [
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [20, 30, 10, 0, 0, 10, 30, 20],
]

KING_EG_PST = [
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10, 0, 0, -10, -20, -30],
    [-30, -10, 20, 20, 20, 20, -10, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -10, 20, 20, 20, 20, -10, -30],
    [-30, -30, 0, 0, 0, 0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50],
]

PST_MAP = {
    "P": PAWN_PST,
    "N": KNIGHT_PST,
    "B": BISHOP_PST,
    "R": ROOK_PST,
    "Q": QUEEN_PST,
}


class TTEntry(TypedDict):
    depth: int
    score: int
    flag: str
    best_move: Optional[Move]


def _castling_zobrist_index(rights: str) -> int:
    """Map castling rights string to index 0..15."""
    mask = 0
    if "K" in rights:
        mask |= 1
    if "Q" in rights:
        mask |= 2
    if "k" in rights:
        mask |= 4
    if "q" in rights:
        mask |= 8
    return mask


def compute_zobrist(board: Board) -> int:
    """Compute 64-bit Zobrist hash for the current position."""
    h = 0
    for sq64 in range(64):
        piece = board.squares[MAILBOX64[sq64]]
        if piece != EMPTY:
            rank = sq64 // 8
            file_idx = sq64 % 8
            h ^= ZOBRIST_PIECES[rank][file_idx][piece]
    if board.active_color == "b":
        h ^= ZOBRIST_SIDE
    h ^= ZOBRIST_CASTLING[_castling_zobrist_index(board.castling_rights)]
    ep_file = 8
    if board.en_passant:
        ep_file = FILES.index(board.en_passant[0])
    h ^= ZOBRIST_EP[ep_file]
    return h


def _zobrist_xor_piece(h: int, sq64: int, piece: int) -> int:
    if piece == EMPTY:
        return h
    rank = sq64 // 8
    file_idx = sq64 % 8
    return h ^ ZOBRIST_PIECES[rank][file_idx][piece]


def zobrist_after_move(board: Board, move: Move, zobrist: int) -> int:
    """Incrementally update Zobrist hash for `move` applied to `board` (before apply_move)."""
    from_idx = move.from_idx
    to_idx = move.to_idx
    moving = board.squares[from_idx]
    if moving == EMPTY:
        moving = move.piece

    h = zobrist
    sq64_from = MAILBOX120[from_idx]
    sq64_to = MAILBOX120[to_idx]

    h = _zobrist_xor_piece(h, sq64_from, moving)

    if move.is_en_passant:
        cap_idx = from_idx + (10 if board.active_color == "w" else -10)
        cap_sq64 = MAILBOX120[cap_idx]
        cap_piece = board.squares[cap_idx]
        h = _zobrist_xor_piece(h, cap_sq64, cap_piece)
        placed = move.promotion if move.promotion else moving
        h = _zobrist_xor_piece(h, sq64_to, placed)
    elif move.is_castling:
        placed = moving
        h = _zobrist_xor_piece(h, sq64_to, placed)
        if to_idx == IDX_G1:
            h = _zobrist_xor_piece(h, MAILBOX120[IDX_H1], wR)
            h = _zobrist_xor_piece(h, MAILBOX120[IDX_F1], wR)
        elif to_idx == IDX_C1:
            h = _zobrist_xor_piece(h, MAILBOX120[IDX_A1], wR)
            h = _zobrist_xor_piece(h, MAILBOX120[IDX_D1], wR)
        elif to_idx == IDX_G8:
            h = _zobrist_xor_piece(h, MAILBOX120[IDX_H8], bR)
            h = _zobrist_xor_piece(h, MAILBOX120[IDX_F8], bR)
        elif to_idx == IDX_C8:
            h = _zobrist_xor_piece(h, MAILBOX120[IDX_A8], bR)
            h = _zobrist_xor_piece(h, MAILBOX120[IDX_D8], bR)
    else:
        target = board.squares[to_idx]
        if target != EMPTY:
            h = _zobrist_xor_piece(h, sq64_to, target)
        placed = move.promotion if move.promotion else moving
        h = _zobrist_xor_piece(h, sq64_to, placed)

    old_castle = _castling_zobrist_index(board.castling_rights)
    rights = set(board.castling_rights)
    for idx in (from_idx, to_idx):
        if idx == IDX_E1:
            rights -= {"K", "Q"}
        elif idx in (IDX_A1, IDX_H1):
            rights -= {"Q"} if idx == IDX_A1 else {"K"}
        elif idx == IDX_E8:
            rights -= {"k", "q"}
        elif idx in (IDX_A8, IDX_H8):
            rights -= {"q"} if idx == IDX_A8 else {"k"}
    new_rights = "".join(r for r in "KQkq" if r in rights)
    h ^= ZOBRIST_CASTLING[old_castle] ^ ZOBRIST_CASTLING[_castling_zobrist_index(new_rights)]

    old_ep = 8
    if board.en_passant:
        old_ep = FILES.index(board.en_passant[0])
    h ^= ZOBRIST_EP[old_ep]

    new_ep = 8
    if moving in (wP, bP):
        delta = to_idx - from_idx
        if delta == -20 or delta == 20:
            mid_idx = (from_idx + to_idx) // 2
            new_ep = FILES.index(idx_to_sq(mid_idx)[0])
    h ^= ZOBRIST_EP[new_ep]

    h ^= ZOBRIST_SIDE  # side to move always flips after a legal move

    return h


class ChessAI:
    """Minimax engine with alpha-beta, iterative deepening, quiescence, and TT."""

    def __init__(
        self,
        time_limit: float = 2.0,
        max_depth: int = 6,
        contempt_cp: int = -10,
    ) -> None:
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.contempt_cp = contempt_cp
        self.nodes = 0
        self.deadline = 0.0
        self.abort_search = False
        self.tt: Dict[int, TTEntry] = {}

    def _store_tt(
        self,
        key: int,
        depth: int,
        score: float,
        flag: str,
        best_move: Optional[Move],
    ) -> None:
        """Store entry in transposition table, evicting if over size limit."""
        if len(self.tt) > TT_MAX_SIZE:
            self.tt.clear()
        self.tt[key] = {
            "depth": depth,
            "score": int(score),
            "flag": flag,
            "best_move": best_move,
        }

    def choose_move(self, game: GameState) -> Optional[Move]:
        """Return best move for the side to move within the time limit."""
        legal = generate_legal_moves(game.board)
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        self.tt = {}
        self.nodes = 0
        self.deadline = time.time() + self.time_limit
        self.abort_search = False

        root_hash = compute_zobrist(game.board)
        game.board.zobrist_hash = root_hash

        best_move = legal[0]
        ordered = order_moves(game.board, legal)

        depth = 1
        while depth <= self.max_depth:
            if time.time() >= self.deadline:
                break
            alpha = -float("inf")
            beta = float("inf")
            depth_best = ordered[0]
            depth_score = -float("inf")

            for move in order_moves(game.board, ordered):
                if time.time() >= self.deadline:
                    self.abort_search = True
                    break
                child = game.board.copy()
                child_hash = zobrist_after_move(game.board, move, root_hash)
                child.apply_move(move)
                child.zobrist_hash = child_hash
                score = -self._search(
                    child,
                    depth - 1,
                    -beta,
                    -alpha,
                    game,
                    not maximizing_for_white(child),
                    child_hash,
                )
                if self.abort_search:
                    break
                if score > depth_score:
                    depth_score = score
                    depth_best = move
                alpha = max(alpha, score)

            if not self.abort_search:
                best_move = depth_best
                ordered = [depth_best] + [m for m in ordered if m != depth_best]
            depth += 1

        return best_move

    def _search(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        game: GameState,
        maximizing_white: bool,
        zobrist: int,
    ) -> float:
        """Negamax search with alpha-beta pruning and transposition table."""
        if time.time() >= self.deadline:
            self.abort_search = True
            return evaluate(board, game, self.contempt_cp)

        self.nodes += 1
        original_alpha = alpha

        tt_move: Optional[Move] = None
        entry = self.tt.get(zobrist)
        if entry is not None and entry["depth"] >= depth:
            tt_score = entry["score"]
            flag = entry["flag"]
            if flag == "exact":
                return tt_score
            if flag == "lower":
                alpha = max(alpha, tt_score)
            elif flag == "upper":
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score
            tt_move = entry.get("best_move")

        legal = generate_legal_moves(board)
        if not legal:
            if board.is_in_check(board.active_color):
                return -20000 + (self.max_depth - depth)
            return self.contempt_cp

        if board.halfmove_clock >= 100:
            return self.contempt_cp
        rep_key = board.position_key()
        if game.position_counts.get(rep_key, 0) >= 3:
            return self.contempt_cp

        if depth <= 0:
            return self._quiescence(board, alpha, beta, game, maximizing_white)

        best_move: Optional[Move] = None
        best_score = -float("inf")
        ordered = order_moves(board, legal, tt_move)

        for move in ordered:
            child = board.copy()
            child_hash = zobrist_after_move(board, move, zobrist)
            child.apply_move(move)
            child.zobrist_hash = child_hash
            score = -self._search(
                child,
                depth - 1,
                -beta,
                -alpha,
                game,
                not maximizing_white,
                child_hash,
            )
            if self.abort_search:
                return score
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        flag = "exact"
        if best_score <= original_alpha:
            flag = "upper"
        elif best_score >= beta:
            flag = "lower"
        self._store_tt(zobrist, depth, best_score, flag, best_move)
        return best_score

    def _quiescence(
        self,
        board: Board,
        alpha: float,
        beta: float,
        game: GameState,
        maximizing_white: bool,
    ) -> float:
        """Capture-only extension at leaf nodes."""
        stand_pat = evaluate(board, game, self.contempt_cp)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        captures = [
            m for m in generate_legal_moves(board) if m.captured or m.is_en_passant
        ]

        for move in order_moves(board, captures):
            child = board.copy()
            child.apply_move(move)
            score = -self._quiescence(
                child,
                -beta,
                -alpha,
                game,
                not maximizing_white,
            )
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha


def maximizing_for_white(board: Board) -> bool:
    """Return True if white is to move (maximizing side in eval)."""
    return board.active_color == "w"


def evaluate(board: Board, game: GameState, contempt_cp: int) -> int:
    """Evaluate position in centipawns from white's perspective."""
    if board.halfmove_clock >= 100:
        return contempt_cp
    rep_key = board.position_key()
    if game.position_counts.get(rep_key, 0) >= 3:
        return contempt_cp

    total_pieces = 0
    for sq64 in range(64):
        if board.squares[MAILBOX64[sq64]] != EMPTY:
            total_pieces += 1

    score = 0
    for sq64 in range(64):
        piece = board.squares[MAILBOX64[sq64]]
        if piece == EMPTY:
            continue
        ptype = piece_type(piece)
        if not ptype:
            continue
        sign = 1 if IS_WHITE(piece) else -1
        rank = sq64 // 8
        file_idx = sq64 % 8
        material = PIECE_VALUES[ptype]
        pst_rank = rank if IS_WHITE(piece) else 7 - rank
        if ptype == "K":
            pst = KING_EG_PST if total_pieces <= 10 else KING_MG_PST
            pst_val = pst[pst_rank][file_idx]
        else:
            pst_val = PST_MAP[ptype][pst_rank][file_idx]
        score += sign * (material + pst_val)

    return score


def _mvv_lva(move: Move) -> int:
    """Most valuable victim, least valuable attacker ordering score."""
    victim = PIECE_VALUES.get(piece_type(move.captured) or "", 0) if move.captured else 0
    if move.is_en_passant:
        victim = PIECE_VALUES["P"]
    attacker = PIECE_VALUES.get(piece_type(move.piece) or "", 0)
    return victim * 1000 - attacker


def order_moves(
    board: Board, moves: List[Move], tt_move: Optional[Move] = None
) -> List[Move]:
    """Order moves: TT move, captures (MVV-LVA), then castling."""

    def score_move(m: Move) -> int:
        if tt_move is not None and m.from_idx == tt_move.from_idx and m.to_idx == tt_move.to_idx:
            return 50000
        if m.captured or m.is_en_passant:
            return 10000 + _mvv_lva(m)
        if m.is_castling:
            return 100
        return 0

    return sorted(moves, key=score_move, reverse=True)
