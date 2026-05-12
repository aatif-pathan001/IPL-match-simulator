# Cricket Coaching Simulator — MVP V1

Real-time tactical decision platform for IPL fans. Every over break you
have 20 seconds to pick the next bowler and set the field. Your choices
are scored against historical IPL tactical benchmarks and compared to
the captain's actual decision.

## Quickstart

```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
# Open http://localhost:8000
```

## Project structure

```
cricket_simulator/
├── engine.py        Core: players, match sim, captain logic, scorer
├── server.py        FastAPI: WebSocket, match loop, REST endpoints
├── static/
│   └── index.html   Complete frontend (field, bowler cards, timer, leaderboard)
├── requirements.txt
└── README.md
```

## How a match works

1. Mumbai Mavricks bat, Chennai Cheetahs bowl (20 overs)
2. Each over break → 20-second decision window opens
3. Pick a bowler + set 10 fielders across 8 zones
4. Window closes → captain's actual decision is revealed
5. Your score = bowler merit (0–50) + field merit (0–50)
6. Repeat for 20 overs; national leaderboard updates live

## Scoring model

| Component      | Max | Basis                                             |
|----------------|-----|---------------------------------------------------|
| Bowler type    | 35  | How well type fits situation (phase + batsman)    |
| Captain match  | 10  | Exact bowler match bonus                          |
| Wicket bonus   | 5   | You picked the bowler who took a wicket           |
| Field merit    | 30  | L1 similarity to benchmark ideal field            |
| Field captain  | 20  | Similarity to captain's actual field              |

Grades: S ≥ 85 · A ≥ 70 · B ≥ 50 · C < 50

## Timing (configurable in server.py)

| Constant          | Default | Description                      |
|-------------------|---------|----------------------------------|
| WINDOW_SECONDS    | 22      | Fan decision window              |
| BALL_INTERVAL     | 0.9 s   | Time between ball-by-ball reveal |
| REVEAL_SECONDS    | 7       | Score reveal display time        |
| GAP_SECONDS       | 3       | Pause before next window         |

Full 20-over match ≈ 13 minutes at default settings.

## Adding real data (V2 path)

Replace `init_match()` in `engine.py` with a CricAPI call.
The rest of the engine is data-agnostic — same scoring model,
same WebSocket protocol.
