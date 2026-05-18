# Architecture

This project splits chess into three layers that communicate through a thin Flask API.

## 1. Rules Engine (`chess_engine/`)

Pure Python with no external chess libraries.

| Module | Responsibility |
|--------|----------------|
| `board.py` | 8×8 board, castling rights (`KQkq`), en passant square, halfmove/fullmove clocks, apply moves, attack detection |
| `pieces.py` | Pseudo-legal move generation for all piece types |
| `move.py` | `Move` dataclass (from/to, piece, capture, promotion, castling, en passant) |
| `legal_moves.py` | Filters pseudo-legal moves so the king is never left in check; castling through check |
| `game_state.py` | `GameState`: move history, threefold repetition counts, checkmate/stalemate/draw detection |
| `ai.py` | Search and evaluation (see below) |

**Data flow:** `GameState` owns a `Board`. Moves are validated with `find_legal_move` / `generate_legal_moves`, then applied via `Board.apply_move`. Status is recomputed after each ply.

## 2. Web Interface

**Backend (`app.py`):** Single global `GameState` and `ChessAI` instance.

| Endpoint | Purpose |
|----------|---------|
| `GET /state` | JSON board + status + legal moves |
| `POST /move` | Human move `{from, to, promotion?}` |
| `POST /reset` | New game; optional `{vs_ai: true}` |
| `GET /ai_move` | AI plays for side to move |
| `POST /analyze` | Multipv analysis for a FEN (depth, num_lines) |
| `POST /set_position` | Load FEN or PGN |

**Frontend (`static/index.html`):** [chessboard.js](https://chessboardjs.com) for drag-and-drop rendering; [chess.js](https://github.com/jhlywa/chess.js) only to build FEN for display helpers. **All rules run on the server** — illegal drops snap back after the API rejects them.

Human vs AI: white is human; after each white move the client calls `GET /ai_move` for black.

## 3. AI Engine (`chess_engine/ai.py`)

`ChessAI.choose_move(game)`:

1. **Iterative deepening** — depth 1…N until time limit (default 2s)
2. **Negamax + alpha-beta** on top of legal moves
3. **Move ordering** — MVV-LVA captures, then checks
4. **Quiescence** — at depth 0, extend with capture-only search
5. **Evaluation** — material + piece-square tables (centipawns, white-positive)
6. **Contempt** — stalemate/repetition/50-move draws scored at -10 cp so the engine prefers fighting for a win

The AI copies `Board` positions during search; it does not mutate the live `GameState` until a move is chosen and applied in `app.py`.

## Layer interaction

```
Browser (chessboard.js UI)
    │  fetch /state, /move, /ai_move
    ▼
Flask (app.py)
    │  GameState + legal_moves + ai.choose_move
    ▼
chess_engine (rules + search)
```

Run locally:

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000
