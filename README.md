# Python Chess

Rules engine, Flask REST API, and minimax AI — no `python-chess` dependency.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the layers fit together.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

- **Human vs Human** — both sides drag pieces on the board
- **Human vs AI** — you play white; black is powered by the engine (~2s per move)
