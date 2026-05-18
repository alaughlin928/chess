"""Multipv analysis using chess_engine AI (no modifications to chess_engine)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from chess_engine.ai import ChessAI, evaluate, maximizing_for_white, order_moves
from chess_engine.board import Board
from chess_engine.game_state import STATUS_CHECKMATE, STATUS_STALEMATE, GameState
from chess_engine.legal_moves import generate_legal_moves
from chess_engine.move import Move

from fen_tools import board_from_fen, move_to_san

MATE_SCORE = 20000


def game_from_board(board: Board) -> GameState:
    """Create a GameState snapshot for evaluation."""
    g = GameState()
    g.board = board.copy()
    g.move_history = []
    g.position_counts = {}
    g._record_position()
    g._update_status()
    return g


def score_to_white_pov(score: float, side_before_move: str) -> int:
    """Convert search score to centipawns from white's perspective."""
    if side_before_move == "w":
        return int(round(score))
    return int(round(-score))


def format_eval_display(cp: int) -> Dict[str, Any]:
    """Format eval for API (centipawns + display string)."""
    if cp > MATE_SCORE - 500:
        mate_in = max(1, (MATE_SCORE - cp + 99) // 100)
        return {"cp": cp, "display": f"M{mate_in}", "mate": mate_in, "is_mate": True}
    if cp < -MATE_SCORE + 500:
        mate_in = max(1, (MATE_SCORE + cp + 99) // 100)
        return {"cp": cp, "display": f"M-{mate_in}", "mate": -mate_in, "is_mate": True}
    pawns = cp / 100.0
    sign = "+" if pawns > 0 else ""
    return {"cp": cp, "display": f"{sign}{pawns:.1f}", "mate": None, "is_mate": False}


def analyze_fen(fen: str, depth: int, num_lines: int) -> Dict[str, Any]:
    """Run multipv analysis at fixed depth; return top lines."""
    board = board_from_fen(fen)
    game = game_from_board(board)
    depth = max(1, min(20, depth))
    num_lines = max(1, min(10, num_lines))

    legal = generate_legal_moves(board)
    if not legal:
        in_check = board.is_in_check(board.active_color)
        status = STATUS_CHECKMATE if in_check else STATUS_STALEMATE
        if in_check:
            cp = MATE_SCORE if board.active_color == "b" else -MATE_SCORE
        else:
            cp = 0
        return {
            "status": status,
            "lines": [],
            "eval": format_eval_display(cp),
            "fen": fen,
        }

    ai = ChessAI(time_limit=3600.0, max_depth=depth)
    ai.deadline = time.time() + 3600.0
    ai.abort_search = False

    side = board.active_color
    root_scores: List[Tuple[Move, float]] = []
    for move in order_moves(board, legal):
        child = board.copy()
        child.apply_move(move)
        score = -ai._search(
            child,
            depth - 1,
            -float("inf"),
            float("inf"),
            game,
            maximizing_for_white(child),
        )
        root_scores.append((move, score))

    root_scores.sort(key=lambda x: x[1], reverse=True)

    lines_out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for move, raw_score in root_scores:
        if len(lines_out) >= num_lines:
            break
        key = move.uci()
        if key in seen:
            continue
        seen.add(key)
        white_cp = score_to_white_pov(raw_score, side)
        pv_moves = _build_pv(board, move, depth, game, ai)
        lines_out.append(
            {
                "rank": len(lines_out) + 1,
                "move": move_to_san(board, move),
                "move_uci": move.uci(),
                "eval": format_eval_display(white_cp),
                "eval_cp": white_cp,
                "pv": _pv_to_san(board, pv_moves),
                "pv_uci": [m.uci() for m in pv_moves],
            }
        )

    best_cp = lines_out[0]["eval_cp"] if lines_out else evaluate(board, game, ai.contempt_cp)
    return {
        "status": game.status,
        "lines": lines_out,
        "eval": format_eval_display(best_cp),
        "fen": fen,
    }


def _build_pv(
    board: Board,
    first_move: Move,
    depth: int,
    game: GameState,
    ai: ChessAI,
) -> List[Move]:
    """Build principal variation starting with first_move."""
    pv: List[Move] = [first_move]
    b = board.copy()
    b.apply_move(first_move)
    for _ in range(max(0, depth - 1)):
        legal = generate_legal_moves(b)
        if not legal:
            break
        best_move = legal[0]
        best_score = -float("inf")
        for mv in order_moves(b, legal):
            child = b.copy()
            child.apply_move(mv)
            score = -ai._search(
                child,
                max(0, depth - 2),
                -float("inf"),
                float("inf"),
                game,
                maximizing_for_white(child),
            )
            if score > best_score:
                best_score = score
                best_move = mv
        pv.append(best_move)
        b.apply_move(best_move)
    return pv


def _pv_to_san(start_board: Board, moves: List[Move]) -> List[str]:
    """Convert PV moves to SAN list."""
    b = start_board.copy()
    sans: List[str] = []
    for m in moves:
        sans.append(move_to_san(b, m))
        b.apply_move(m)
    return sans
