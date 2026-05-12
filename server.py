"""
server.py — Cricket Coaching Simulator Server

FastAPI app that:
- Serves the frontend static file
- Manages WebSocket connections (one per fan)
- Runs an async match loop (over-by-over simulation)
- Broadcasts events to all connected fans
- Scores fan decisions and pushes individual results
"""

from __future__ import annotations
import asyncio
import json
import uuid
from collections import defaultdict
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from engine import (
    BallResult, FieldPlacement, MatchState, Player, WindowState,
    captain_choose_bowler, captain_set_field, compute_full_score,
    get_benchmark_key, init_match, score_bowler_choice,
    score_field_placement, simulate_over, TEAM_META, BENCHMARKS,
)

# ─────────────────────────────────────────────────────────────
# Timing constants  (seconds — tune for demo speed)
# ─────────────────────────────────────────────────────────────
WINDOW_SECONDS   = 22    # fan decision window
BALL_INTERVAL    = 0.9   # simulated time between each ball
REVEAL_SECONDS   = 7     # how long the reveal screen stays
GAP_SECONDS      = 3     # brief pause before next window opens
TOTAL_OVERS      = 20


# ─────────────────────────────────────────────────────────────
# Connection Manager
# ─────────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.connections: dict[str, WebSocket] = {}    # fan_id → WebSocket
        self.fan_names:   dict[str, str]       = {}    # fan_id → display name
        self.scores:      dict[str, int]       = defaultdict(int)  # fan_id → cumulative score
        self.over_history: dict[str, list]     = defaultdict(list)

    async def connect(self, ws: WebSocket) -> str:
        await ws.accept()
        fan_id  = str(uuid.uuid4())[:8]
        fan_num = len(self.connections) + 1
        name    = f"Coach #{fan_num}"
        self.connections[fan_id] = ws
        self.fan_names[fan_id]   = name
        return fan_id

    def disconnect(self, fan_id: str):
        self.connections.pop(fan_id, None)

    async def send(self, fan_id: str, data: dict):
        ws = self.connections.get(fan_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                pass

    async def broadcast(self, data: dict):
        dead = []
        for fid, ws in self.connections.items():
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(fid)
        for fid in dead:
            self.disconnect(fid)

    def leaderboard(self) -> list[dict]:
        board = [
            {"fan_id": fid, "name": self.fan_names[fid], "score": self.scores[fid]}
            for fid in self.connections
        ]
        board.sort(key=lambda x: x["score"], reverse=True)
        for i, entry in enumerate(board, 1):
            entry["rank"] = i
        return board

    def add_score(self, fan_id: str, points: int, over_num: int):
        self.scores[fan_id] += points
        self.over_history[fan_id].append({"over": over_num + 1, "points": points})


# ─────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="Cricket Coaching Simulator")
app.mount("/static", StaticFiles(directory="static"), name="static")

mgr = ConnectionManager()

# Submissions for current over: fan_id → {bowler_id, field}
current_submissions: dict[str, dict] = {}

# Match components (set during startup)
match_state:   MatchState | None = None
all_players:   dict[str, Player]  = {}
batting_side:  dict[str, Player]  = {}
bowling_side:  dict[str, Player]  = {}
over_results:  list               = []    # list of OverResult


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def build_over_context(state: MatchState, available_bowlers: list[Player],
                        striker: Player, nonstriker: Player) -> dict:
    """Assemble the full context packet sent to fans at window-open."""
    return {
        "over_number":  state.current_over + 1,   # 1-indexed for display
        "phase":        state.phase.value,
        "score":        state.runs,
        "wickets":      state.wickets,
        "run_rate":     state.run_rate,
        "last_over_runs":    state.last_over_runs,
        "last_over_wickets": state.last_over_wickets,
        "striker":   striker.to_dict(),
        "nonstriker": nonstriker.to_dict(),
        "bowlers":   [b.to_dict() for b in available_bowlers],
        "team_batting": {
            "id":   state.team_batting_id,
            **TEAM_META[state.team_batting_id],
        },
        "team_bowling": {
            "id":   state.team_bowling_id,
            **TEAM_META[state.team_bowling_id],
        },
    }


def ball_to_dict(b: BallResult) -> dict:
    return {
        "ball":       b.ball_num,
        "runs":       b.runs,
        "is_wicket":  b.is_wicket,
        "commentary": b.commentary,
    }


# ─────────────────────────────────────────────────────────────
# Match Loop
# ─────────────────────────────────────────────────────────────

async def run_match():
    """Main async match loop — runs for TOTAL_OVERS overs."""
    global match_state, all_players, batting_side, bowling_side

    await asyncio.sleep(3)   # brief pause before match starts

    await mgr.broadcast({"type": "match_start", "message": "Match is live! Mumbai Mavricks vs Chennai Cheetahs"})
    await asyncio.sleep(1)

    for over_index in range(TOTAL_OVERS):
        if match_state.wickets >= 10:
            break

        match_state.current_over = over_index
        match_state.window_state = WindowState.OPEN
        current_submissions.clear()

        # Pick captain's decision (fans don't see this yet)
        striker    = all_players[match_state.current_striker_id]
        nonstriker = all_players[match_state.current_nonstriker_id]
        bowlers    = [p for p in bowling_side.values() if p.is_bowler]

        cap_bowler = captain_choose_bowler(bowlers, match_state, striker)
        cap_field  = captain_set_field(cap_bowler, match_state, striker)

        # Available bowlers list (what fans see)
        available  = [b for b in bowlers if b.can_bowl() and b.id != match_state.last_bowler_id]
        if not available:
            available = [b for b in bowlers if b.can_bowl()]

        context = build_over_context(match_state, available, striker, nonstriker)

        # ── Broadcast: window open ────────────────────────────
        await mgr.broadcast({
            "type":    "window_open",
            "context": context,
            "seconds": WINDOW_SECONDS,
        })

        # Countdown ticks
        for remaining in range(WINDOW_SECONDS - 1, 0, -1):
            await asyncio.sleep(1)
            if remaining in (15, 10, 5, 3, 2, 1):
                await mgr.broadcast({"type": "tick", "remaining": remaining})

        await asyncio.sleep(1)
        match_state.window_state = WindowState.LOCKED

        await mgr.broadcast({"type": "window_closed"})
        await asyncio.sleep(0.5)

        # ── Simulate the over ─────────────────────────────────
        over_result = simulate_over(cap_bowler, batting_side, match_state, cap_field)
        over_results.append(over_result)

        # Broadcast balls one by one
        for ball in over_result.balls:
            await asyncio.sleep(BALL_INTERVAL)
            await mgr.broadcast({
                "type":  "ball_result",
                "ball":  ball_to_dict(ball),
                "score": match_state.runs + over_result.balls[:over_result.balls.index(ball)+1]
                    .count(ball) and sum(
                        b.runs for b in over_result.balls[:over_result.balls.index(ball)+1]
                    ) + match_state.runs,
            })

        # Update global match state
        match_state.runs    += over_result.total_runs
        match_state.wickets += over_result.wickets
        match_state.last_over_runs    = over_result.total_runs
        match_state.last_over_wickets = over_result.wickets
        match_state.last_bowler_id    = cap_bowler.id

        await asyncio.sleep(0.5)

        # ── Broadcast: reveal ─────────────────────────────────
        await mgr.broadcast({
            "type": "over_reveal",
            "over": over_index + 1,
            "captain_bowler": cap_bowler.to_dict(),
            "captain_field":  cap_field.to_dict(),
            "over_result": {
                "total_runs": over_result.total_runs,
                "wickets":    over_result.wickets,
                "balls":      [ball_to_dict(b) for b in over_result.balls],
            },
            "match_state": match_state.to_dict(),
        })

        # ── Score each fan ────────────────────────────────────
        bench_key = get_benchmark_key(cap_bowler, match_state, striker)
        scores_for_leaderboard: list[dict] = []

        for fan_id in list(mgr.connections.keys()):
            sub = current_submissions.get(fan_id)

            if sub:
                fan_bowler_id = sub.get("bowler_id")
                fan_bowler    = all_players.get(fan_bowler_id, cap_bowler)
                fan_field     = FieldPlacement.from_dict(sub.get("field", {}))

                b_scores = score_bowler_choice(fan_bowler, cap_bowler, bench_key)
                f_scores = score_field_placement(fan_field, cap_field, bench_key)
                full     = compute_full_score(b_scores, f_scores, over_result, fan_bowler_id)
                points   = full["total"]
                submitted = True
            else:
                # No submission — 0 points, still get a score packet
                full = {
                    "bowler": {"type_score": 0, "captain_bonus": 0, "wicket_bonus": 0, "total": 0},
                    "field":  {"merit_score": 0, "captain_score": 0, "total": 0},
                    "total":  0,
                    "grade":  "–",
                }
                points = 0
                submitted = False

            mgr.add_score(fan_id, points, over_index)
            scores_for_leaderboard.append({
                "fan_id": fan_id,
                "name":   mgr.fan_names[fan_id],
                "score":  mgr.scores[fan_id],
            })

            await mgr.send(fan_id, {
                "type":       "score_update",
                "over":       over_index + 1,
                "submitted":  submitted,
                "breakdown":  full,
                "cumulative": mgr.scores[fan_id],
                "leaderboard": mgr.leaderboard(),
                "benchmark_desc": (
                    BENCHMARKS[bench_key]["desc"] if bench_key in BENCHMARKS else ""
                ),
            })

        await asyncio.sleep(REVEAL_SECONDS)

        match_state.window_state = WindowState.IDLE
        await mgr.broadcast({
            "type":        "over_complete",
            "over":        over_index + 1,
            "match_state": match_state.to_dict(),
            "leaderboard": mgr.leaderboard(),
        })

        if match_state.wickets >= 10:
            break

        await asyncio.sleep(GAP_SECONDS)

    # ── Match over ────────────────────────────────────────────
    match_state.innings_complete = True
    await mgr.broadcast({
        "type":        "match_complete",
        "final_score": f"{match_state.runs}/{match_state.wickets}",
        "overs":       match_state.current_over + 1,
        "leaderboard": mgr.leaderboard(),
        "mvp":         mgr.leaderboard()[0] if mgr.leaderboard() else None,
    })


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global match_state, all_players, batting_side, bowling_side
    match_state, all_players, batting_side, bowling_side = init_match()
    asyncio.create_task(run_match())


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/api/state")
async def get_state():
    if not match_state:
        return JSONResponse({"error": "match not initialised"}, 404)
    return {
        "match_state":   match_state.to_dict(),
        "team_batting":  TEAM_META[match_state.team_batting_id],
        "team_bowling":  TEAM_META[match_state.team_bowling_id],
        "leaderboard":   mgr.leaderboard(),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    fan_id = await mgr.connect(ws)

    # Send current match state immediately on connect
    await mgr.send(fan_id, {
        "type":    "connected",
        "fan_id":  fan_id,
        "name":    mgr.fan_names[fan_id],
        "state":   match_state.to_dict() if match_state else {},
        "leaderboard": mgr.leaderboard(),
    })

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "decision":
                over_num   = msg.get("over")
                bowler_id  = msg.get("bowler_id")
                field_data = msg.get("field", {})

                # Validate the submission window is still open
                if (match_state and
                        match_state.window_state == WindowState.OPEN and
                        over_num == match_state.current_over + 1 and
                        fan_id not in current_submissions):

                    current_submissions[fan_id] = {
                        "bowler_id": bowler_id,
                        "field":     field_data,
                    }
                    await mgr.send(fan_id, {
                        "type":      "submission_ack",
                        "message":   "Decision locked in!",
                        "bowler_id": bowler_id,
                    })

            elif msg_type == "set_name":
                new_name = str(msg.get("name", ""))[:20].strip()
                if new_name:
                    mgr.fan_names[fan_id] = new_name
                    await mgr.send(fan_id, {"type": "name_updated", "name": new_name})

    except WebSocketDisconnect:
        mgr.disconnect(fan_id)
