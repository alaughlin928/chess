"""Flask REST API — analysis board + play mode."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory

from analysis_service import analyze_fen, game_from_board
from chess_engine.ai import ChessAI
from chess_engine.game_state import GameState
from chess_engine.legal_moves import find_legal_move
from chess_engine.move import Move
from fen_tools import (
    board_from_fen,
    board_to_fen,
    move_to_san,
    moves_to_san_list,
    parse_pgn_moves,
)

app = Flask(__name__, static_folder="static")
game = GameState()
ai_engine = ChessAI(time_limit=2.0, contempt_cp=-10)
vs_ai = False
pgn_fens: List[str] = []


def _enrich_state(payload: dict) -> dict:
    """Add FEN and SAN move history to state payload."""
    payload["fen"] = board_to_fen(game.board)
    payload["moves_san"] = [
        move_to_san(
            _board_at_ply(i),
            game.move_history[i],
        )
        for i in range(len(game.move_history))
    ]
    payload["pgn_fens"] = pgn_fens
    payload["pgn_index"] = len(game.move_history)
    return payload


def _board_at_ply(ply: int):
    """Reconstruct board before move at ply (for SAN)."""
    from chess_engine.board import Board

    b = Board()
    for i in range(ply):
        b.apply_move(game.move_history[i])
    return b


def _state_response(extra: dict | None = None):
    payload = _enrich_state(game.to_dict())
    payload["vs_ai"] = vs_ai
    if extra:
        payload.update(extra)
    return jsonify(payload)


def _load_board_from_fen(fen: str) -> None:
    """Replace global game from FEN."""
    global game, pgn_fens
    game = game_from_board(board_from_fen(fen))
    game.move_history = []
    pgn_fens = [fen]


def _rebuild_pgn_fens() -> None:
    """Rebuild FEN list after each move for PGN stepping."""
    global pgn_fens
    from chess_engine.board import Board

    start = Board()
    fens = [board_to_fen(start)]
    b = Board()
    for m in game.move_history:
        b.apply_move(m)
        fens.append(board_to_fen(b))
    pgn_fens = fens


@app.route("/")
def index():
    """Serve the chess UI."""
    return send_from_directory("static", "index.html")


@app.route("/state", methods=["GET"])
def get_state():
    """Return current board state as JSON."""
    return _state_response()


@app.route("/move", methods=["POST"])
def post_move():
    """Apply a move; body: {from, to, promotion?}."""
    data = request.get_json(force=True) or {}
    from_sq = data.get("from", "").lower()
    to_sq = data.get("to", "").lower()
    promotion = data.get("promotion")

    if game.is_game_over():
        return jsonify({"error": "Game is over", **_enrich_state(game.to_dict())}), 400

    move = find_legal_move(game.board, from_sq, to_sq, promotion)
    if move is None:
        return jsonify({"error": "Illegal move", **_enrich_state(game.to_dict())}), 400

    game.apply_move(move)
    _rebuild_pgn_fens()
    return _state_response(
        {
            "last_move": move.uci(),
            "last_move_san": move_to_san(_board_at_ply(len(game.move_history) - 1), move),
        }
    )


@app.route("/reset", methods=["POST"])
def post_reset():
    """Reset to starting position."""
    global vs_ai, game, pgn_fens
    data = request.get_json(silent=True) or {}
    vs_ai = bool(data.get("vs_ai", False))
    game = GameState()
    pgn_fens = [board_to_fen(game.board)]
    return _state_response()


@app.route("/ai_move", methods=["GET"])
def get_ai_move():
    """Let the AI play for the side to move."""
    if game.is_game_over():
        return jsonify({"error": "Game is over", **_enrich_state(game.to_dict())}), 400

    ply = len(game.move_history)
    move = ai_engine.choose_move(game)
    if move is None:
        return jsonify({"error": "No legal moves", **_enrich_state(game.to_dict())}), 400

    san = move_to_san(_board_at_ply(ply), move)
    game.apply_move(move)
    _rebuild_pgn_fens()
    return _state_response({"ai_move": move.uci(), "ai_move_san": san, "last_move": move.uci()})


@app.route("/mode", methods=["POST"])
def set_mode():
    """Set human vs human or human vs AI."""
    global vs_ai
    data = request.get_json(force=True) or {}
    vs_ai = bool(data.get("vs_ai", False))
    return _state_response()


@app.route("/analyze", methods=["POST"])
def post_analyze():
    """Analyze a FEN at given depth; return multipv lines."""
    data = request.get_json(force=True) or {}
    fen = data.get("fen", board_to_fen(game.board))
    depth = int(data.get("depth", 12))
    num_lines = int(data.get("num_lines", 3))
    try:
        result = analyze_fen(fen, depth, num_lines)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/set_position", methods=["POST"])
def post_set_position():
    """Load position from FEN or PGN."""
    global game, pgn_fens
    data = request.get_json(force=True) or {}
    fen = data.get("fen")
    pgn = data.get("pgn")
    moves_out: List[Dict[str, str]] = []

    try:
        if pgn:
            parsed = parse_pgn_moves(pgn)
            game = GameState()
            game.move_history = []
            game.position_counts = {}
            for m in parsed:
                game.apply_move(m)
            _rebuild_pgn_fens()
            start_board = __import__("chess_engine.board", fromlist=["Board"]).Board()
            sans = moves_to_san_list(start_board, parsed)
            for i, m in enumerate(parsed):
                moves_out.append({"uci": m.uci(), "san": sans[i]})
            fen = board_to_fen(game.board)
            # pgn_fens rebuilt with start + after each move
        elif fen:
            _load_board_from_fen(fen)
            pgn_fens = [fen]
        else:
            game.reset()
            pgn_fens = [board_to_fen(game.board)]
            fen = pgn_fens[0]
            moves_out = []
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    payload = _enrich_state(game.to_dict())
    payload["fen"] = fen
    payload["moves"] = moves_out
    return jsonify(payload)


if __name__ == "__main__":
    pgn_fens = [board_to_fen(game.board)]
    app.run(debug=True, port=5000)
