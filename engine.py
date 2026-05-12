"""
engine.py — Cricket Coaching Simulator Core Engine

Handles:
- Player and team data (two IPL-style squads)
- Match state machine and ball-by-ball simulation
- Captain decision heuristic (bowler selection + field logic)
- Tactical merit scoring model (synthetic benchmarks)
"""

from __future__ import annotations
import random
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────

class BowlerType(str, Enum):
    PACE_FAST   = "pace_fast"
    PACE_MEDIUM = "pace_medium"
    OFF_SPIN    = "off_spin"
    LEG_SPIN    = "leg_spin"

class Phase(str, Enum):
    POWERPLAY = "powerplay"   # overs 1–6
    MIDDLE    = "middle"      # overs 7–15
    DEATH     = "death"       # overs 16–20

class BattingStyle(str, Enum):
    RHB = "RHB"
    LHB = "LHB"

class BattingType(str, Enum):
    AGGRESSOR = "aggressor"
    ANCHOR    = "anchor"

class WindowState(str, Enum):
    IDLE      = "idle"
    OPEN      = "open"
    LOCKED    = "locked"
    REVEALING = "revealing"
    SCORED    = "scored"


# ─────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────

@dataclass
class Player:
    id: str
    name: str
    team_id: str
    is_batsman: bool
    is_bowler: bool
    bowler_type: Optional[BowlerType] = None
    max_overs: int = 4
    batting_style: BattingStyle = BattingStyle.RHB
    batting_type: BattingType = BattingType.AGGRESSOR
    # Bowling profile
    economy: float = 8.5
    wicket_prob: float = 0.12   # probability of taking wicket per ball
    # In-match bowling stats (reset each match)
    overs_bowled: int = 0
    runs_given: int = 0
    wickets_taken: int = 0
    # In-match batting stats
    balls_faced: int = 0
    runs_scored: int = 0
    is_out: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "team_id": self.team_id,
            "is_bowler": self.is_bowler,
            "bowler_type": self.bowler_type.value if self.bowler_type else None,
            "max_overs": self.max_overs,
            "batting_style": self.batting_style.value,
            "batting_type": self.batting_type.value,
            "economy": self.economy,
            "overs_bowled": self.overs_bowled,
            "runs_given": self.runs_given,
            "wickets_taken": self.wickets_taken,
            "balls_faced": self.balls_faced,
            "runs_scored": self.runs_scored,
            "is_out": self.is_out,
        }

    def match_economy(self) -> float:
        if self.overs_bowled == 0:
            return self.economy
        return round(self.runs_given / self.overs_bowled, 2)

    def can_bowl(self) -> bool:
        return self.is_bowler and self.overs_bowled < self.max_overs


@dataclass
class FieldPlacement:
    """Fan or captain field placement across 8 tactical zones. Total must = 10."""
    slip_cordon:  int = 0   # 0–3  | close catchers behind wicket, off-side
    point_cover:  int = 0   # 0–3  | off-side ring
    mid_off:      int = 1   # 0–1  | straight off-side (up=1, back=0)
    mid_on:       int = 1   # 0–1  | straight on-side
    square_leg:   int = 0   # 0–2  | on-side ring
    fine_leg:     int = 1   # 0–2  | behind wicket on-side / third man
    deep_offside: int = 2   # 0–3  | off-side boundary
    deep_onside:  int = 2   # 0–3  | on-side boundary

    def total(self) -> int:
        return (self.slip_cordon + self.point_cover + self.mid_off +
                self.mid_on + self.square_leg + self.fine_leg +
                self.deep_offside + self.deep_onside)

    def to_dict(self) -> dict:
        return {
            "slip_cordon":  self.slip_cordon,
            "point_cover":  self.point_cover,
            "mid_off":      self.mid_off,
            "mid_on":       self.mid_on,
            "square_leg":   self.square_leg,
            "fine_leg":     self.fine_leg,
            "deep_offside": self.deep_offside,
            "deep_onside":  self.deep_onside,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FieldPlacement":
        return cls(
            slip_cordon  = d.get("slip_cordon",  0),
            point_cover  = d.get("point_cover",  0),
            mid_off      = d.get("mid_off",      1),
            mid_on       = d.get("mid_on",       1),
            square_leg   = d.get("square_leg",   0),
            fine_leg     = d.get("fine_leg",     1),
            deep_offside = d.get("deep_offside", 2),
            deep_onside  = d.get("deep_onside",  2),
        )


@dataclass
class BallResult:
    ball_num: int            # 1–6
    runs: int
    is_wicket: bool
    commentary: str


@dataclass
class OverResult:
    over_num: int            # 0-indexed
    bowler_id: str
    total_runs: int
    wickets: int
    balls: list[BallResult]
    captain_field: FieldPlacement


@dataclass
class MatchState:
    team_batting_id: str
    team_bowling_id: str
    runs: int = 0
    wickets: int = 0
    current_over: int = 0       # 0-indexed; 0 = first over
    last_over_runs: int = 0
    last_over_wickets: int = 0
    batting_order: list = field(default_factory=list)   # ordered player IDs
    current_striker_id: Optional[str] = None
    current_nonstriker_id: Optional[str] = None
    last_bowler_id: Optional[str] = None
    window_state: WindowState = WindowState.IDLE
    innings_complete: bool = False

    @property
    def phase(self) -> Phase:
        if self.current_over < 6:
            return Phase.POWERPLAY
        if self.current_over < 15:
            return Phase.MIDDLE
        return Phase.DEATH

    @property
    def run_rate(self) -> float:
        if self.current_over == 0:
            return 0.0
        return round(self.runs / self.current_over, 2)

    def to_dict(self) -> dict:
        return {
            "runs": self.runs,
            "wickets": self.wickets,
            "current_over": self.current_over,
            "last_over_runs": self.last_over_runs,
            "last_over_wickets": self.last_over_wickets,
            "phase": self.phase.value,
            "run_rate": self.run_rate,
            "window_state": self.window_state.value,
            "innings_complete": self.innings_complete,
        }


# ─────────────────────────────────────────────────────────────
# Team Data — Two IPL-style squads
# ─────────────────────────────────────────────────────────────

def build_teams() -> dict[str, list[Player]]:
    """Returns {team_id: [Player, ...]} for both teams."""

    # ── Team A: Mumbai Mavricks ───────────────────────────────
    mm = [
        Player("mm_rohit",   "Rohit Sharma",     "mm", True,  False, batting_style=BattingStyle.RHB, batting_type=BattingType.ANCHOR),
        Player("mm_prithvi", "Prithvi Kumar",    "mm", True,  False, batting_style=BattingStyle.RHB, batting_type=BattingType.AGGRESSOR),
        Player("mm_surya",   "Surya Mehta",      "mm", True,  False, batting_style=BattingStyle.RHB, batting_type=BattingType.AGGRESSOR),
        Player("mm_tilak",   "Tilak Varma",      "mm", True,  False, batting_style=BattingStyle.LHB, batting_type=BattingType.ANCHOR),
        Player("mm_hardik",  "Hardik Kumar",     "mm", True,  True,  BowlerType.PACE_MEDIUM, 4, BattingStyle.RHB, BattingType.AGGRESSOR, 8.8, 0.10),
        Player("mm_tim",     "Tim Santos",       "mm", True,  False, batting_style=BattingStyle.RHB, batting_type=BattingType.AGGRESSOR),
        Player("mm_ishan",   "Ishan Shah",       "mm", True,  False, batting_style=BattingStyle.LHB, batting_type=BattingType.AGGRESSOR),
        Player("mm_krunal",  "Krunal Bhai",      "mm", True,  True,  BowlerType.OFF_SPIN,    4, BattingStyle.LHB, BattingType.ANCHOR,    7.8, 0.13),
        Player("mm_bumrah",  "Jasprit Roy",      "mm", False, True,  BowlerType.PACE_FAST,   4, economy=6.9, wicket_prob=0.22),
        Player("mm_trent",   "Trent Brook",      "mm", False, True,  BowlerType.PACE_FAST,   4, economy=7.4, wicket_prob=0.17),
        Player("mm_piyush",  "Piyush Tiwari",    "mm", False, True,  BowlerType.LEG_SPIN,    4, economy=7.6, wicket_prob=0.19),
    ]

    # ── Team B: Chennai Cheetahs ──────────────────────────────
    cc = [
        Player("cc_ms",      "MS Raina",         "cc", True,  False, batting_style=BattingStyle.RHB, batting_type=BattingType.AGGRESSOR),
        Player("cc_rutu",    "Ruturaj Kumar",    "cc", True,  False, batting_style=BattingStyle.RHB, batting_type=BattingType.ANCHOR),
        Player("cc_devon",   "Devon Shah",       "cc", True,  False, batting_style=BattingStyle.LHB, batting_type=BattingType.AGGRESSOR),
        Player("cc_shivam",  "Shivam Ambani",    "cc", True,  False, batting_style=BattingStyle.RHB, batting_type=BattingType.AGGRESSOR),
        Player("cc_moeen",   "Moeen Roy",        "cc", True,  True,  BowlerType.OFF_SPIN,    4, BattingStyle.LHB, BattingType.ANCHOR,    7.5, 0.15),
        Player("cc_jadeja",  "Jadeja Singh",     "cc", True,  True,  BowlerType.OFF_SPIN,    4, BattingStyle.LHB, BattingType.ANCHOR,    7.2, 0.17),
        Player("cc_dhoni",   "Dhoni Patel",      "cc", True,  False, batting_style=BattingStyle.RHB, batting_type=BattingType.AGGRESSOR),
        Player("cc_dube",    "Shubman Dube",     "cc", True,  False, batting_style=BattingStyle.LHB, batting_type=BattingType.AGGRESSOR),
        Player("cc_deepak",  "Deepak Roy",       "cc", False, True,  BowlerType.PACE_MEDIUM, 4, economy=7.9, wicket_prob=0.18),
        Player("cc_mustaf",  "Mustafizur Ray",   "cc", False, True,  BowlerType.PACE_FAST,   4, economy=7.6, wicket_prob=0.16),
        Player("cc_theek",   "Maheesh Teeksha",  "cc", False, True,  BowlerType.OFF_SPIN,    4, economy=7.3, wicket_prob=0.21),
    ]

    return {"mm": mm, "cc": cc}


TEAM_META = {
    "mm": {"name": "Mumbai Mavricks",  "short": "MM", "color": "#1d4ed8"},
    "cc": {"name": "Chennai Cheetahs", "short": "CC", "color": "#b45309"},
}


# ─────────────────────────────────────────────────────────────
# Tactical Merit Benchmarks
# ─────────────────────────────────────────────────────────────
# Each entry: ideal bowler types (ordered preference) + ideal field zones
# Used to score fan decisions without real-time field data.
# Built from cricket coaching principles cross-checked against IPL captaincy trends.

BENCHMARKS: dict[str, dict] = {
    # ── Powerplay benchmarks ─────────────────────────────────
    "pace_fast_powerplay_aggressor": {
        "preferred_bowlers": [BowlerType.PACE_FAST, BowlerType.PACE_MEDIUM],
        "field": FieldPlacement(slip_cordon=2, point_cover=2, mid_off=1, mid_on=1,
                                square_leg=1, fine_leg=1, deep_offside=1, deep_onside=1),
        "desc": "Attacking field: 2 slips, ring fielders up, minimal boundary protection",
        "tags": ["attacking", "swing_trap", "catching_cordon"],
    },
    "pace_fast_powerplay_anchor": {
        "preferred_bowlers": [BowlerType.PACE_FAST, BowlerType.PACE_MEDIUM],
        "field": FieldPlacement(slip_cordon=1, point_cover=2, mid_off=1, mid_on=1,
                                square_leg=1, fine_leg=1, deep_offside=2, deep_onside=1),
        "desc": "One slip kept, spread fielders to restrict singles from an anchor",
        "tags": ["balanced", "restrict_singles"],
    },
    "pace_medium_powerplay_aggressor": {
        "preferred_bowlers": [BowlerType.PACE_MEDIUM, BowlerType.PACE_FAST],
        "field": FieldPlacement(slip_cordon=1, point_cover=2, mid_off=1, mid_on=1,
                                square_leg=1, fine_leg=1, deep_offside=2, deep_onside=1),
        "desc": "Single slip, ring fielders, protect the boundary on the offside",
        "tags": ["balanced", "containing"],
    },

    # ── Middle overs benchmarks ──────────────────────────────
    "pace_fast_middle_aggressor": {
        "preferred_bowlers": [BowlerType.PACE_FAST, BowlerType.PACE_MEDIUM],
        "field": FieldPlacement(slip_cordon=1, point_cover=2, mid_off=1, mid_on=1,
                                square_leg=1, fine_leg=1, deep_offside=2, deep_onside=1),
        "desc": "One slip retained; strong ring to cut off easy singles",
        "tags": ["balanced", "wicket_intent"],
    },
    "off_spin_middle_RHB": {
        "preferred_bowlers": [BowlerType.OFF_SPIN, BowlerType.LEG_SPIN],
        "field": FieldPlacement(slip_cordon=1, point_cover=2, mid_off=1, mid_on=0,
                                square_leg=2, fine_leg=1, deep_offside=1, deep_onside=2),
        "desc": "Slip for edge, cover the sweep/slog, packed leg-side for RHB",
        "tags": ["spin_trap", "attack_offside", "leg_side_cover"],
    },
    "off_spin_middle_LHB": {
        "preferred_bowlers": [BowlerType.OFF_SPIN],
        "field": FieldPlacement(slip_cordon=0, point_cover=3, mid_off=1, mid_on=1,
                                square_leg=1, fine_leg=1, deep_offside=1, deep_onside=2),
        "desc": "No slip — LHB drives through off; pack the off-side",
        "tags": ["off_side_heavy", "contain_drive"],
    },
    "leg_spin_middle_RHB": {
        "preferred_bowlers": [BowlerType.LEG_SPIN, BowlerType.OFF_SPIN],
        "field": FieldPlacement(slip_cordon=1, point_cover=2, mid_off=1, mid_on=1,
                                square_leg=1, fine_leg=1, deep_offside=1, deep_onside=2),
        "desc": "1 slip for googly edge, ring fielders protect the gap, deep on-side for sweep",
        "tags": ["balanced", "googly_trap"],
    },
    "leg_spin_middle_anchor": {
        "preferred_bowlers": [BowlerType.LEG_SPIN, BowlerType.OFF_SPIN],
        "field": FieldPlacement(slip_cordon=0, point_cover=2, mid_off=1, mid_on=1,
                                square_leg=1, fine_leg=1, deep_offside=2, deep_onside=2),
        "desc": "No slip — anchor won't edge; spread to force risky shots",
        "tags": ["containing", "force_risk"],
    },
    "pace_medium_middle_aggressor": {
        "preferred_bowlers": [BowlerType.PACE_MEDIUM, BowlerType.PACE_FAST],
        "field": FieldPlacement(slip_cordon=0, point_cover=2, mid_off=1, mid_on=1,
                                square_leg=1, fine_leg=1, deep_offside=2, deep_onside=2),
        "desc": "No slip; protect the boundary, cut off big shots from aggressors",
        "tags": ["defensive", "contain"],
    },

    # ── Death overs benchmarks ───────────────────────────────
    "any_death_low_wickets": {
        "preferred_bowlers": [BowlerType.PACE_FAST, BowlerType.PACE_MEDIUM],
        "field": FieldPlacement(slip_cordon=0, point_cover=1, mid_off=0, mid_on=0,
                                square_leg=1, fine_leg=2, deep_offside=3, deep_onside=3),
        "desc": "Full boundary protection, yorker-length field for death overs",
        "tags": ["full_defensive", "yorker_field", "boundary_save"],
    },
    "any_death_defending": {
        "preferred_bowlers": [BowlerType.PACE_FAST, BowlerType.PACE_MEDIUM],
        "field": FieldPlacement(slip_cordon=0, point_cover=1, mid_off=1, mid_on=0,
                                square_leg=1, fine_leg=1, deep_offside=3, deep_onside=3),
        "desc": "Defend totals: 6 on the boundary, block the big shots",
        "tags": ["defend_total", "boundary_ring"],
    },
    "spin_death_low_wickets": {
        "preferred_bowlers": [BowlerType.OFF_SPIN, BowlerType.LEG_SPIN],
        "field": FieldPlacement(slip_cordon=0, point_cover=1, mid_off=1, mid_on=0,
                                square_leg=0, fine_leg=2, deep_offside=3, deep_onside=3),
        "desc": "Spin in death with wide boundary cover; limit slog-sweep damage",
        "tags": ["spin_death", "slog_protection"],
    },
}


def get_benchmark_key(bowler: Player, state: MatchState, striker: Player) -> str:
    """Map current match context to the closest benchmark key."""
    phase = state.phase
    btype = bowler.bowler_type if bowler.bowler_type else BowlerType.PACE_MEDIUM
    batting_type = striker.batting_type.value

    if phase == Phase.DEATH:
        if state.wickets >= 6:
            return "any_death_low_wickets"
        if btype in (BowlerType.OFF_SPIN, BowlerType.LEG_SPIN):
            return "spin_death_low_wickets"
        return "any_death_defending"

    if phase == Phase.POWERPLAY:
        if btype == BowlerType.PACE_FAST:
            return f"pace_fast_powerplay_{batting_type}"
        return f"pace_medium_powerplay_{batting_type}"

    # Middle overs
    if btype == BowlerType.PACE_FAST:
        return "pace_fast_middle_aggressor"
    if btype == BowlerType.PACE_MEDIUM:
        return "pace_medium_middle_aggressor"
    if btype == BowlerType.OFF_SPIN:
        return f"off_spin_middle_{striker.batting_style.value}"
    if btype == BowlerType.LEG_SPIN:
        return f"leg_spin_middle_{batting_type}"

    return "pace_medium_middle_aggressor"


# ─────────────────────────────────────────────────────────────
# Tactical Merit Scorer
# ─────────────────────────────────────────────────────────────

def score_bowler_choice(fan_bowler: Player, captain_bowler: Player,
                         benchmark_key: str) -> dict:
    """
    Score the fan's bowler selection (0–50 pts).

    Components:
    - Type merit    (0–35): how well the bowler type fits the situation
    - Captain match (0–10): bonus if fan picked same bowler as captain
    - Wicket bonus  (0–5):  added post-over if bowler took a wicket
    """
    benchmark = BENCHMARKS.get(benchmark_key)
    preferred = benchmark["preferred_bowlers"] if benchmark else [BowlerType.PACE_FAST]

    fan_type    = fan_bowler.bowler_type
    cap_type    = captain_bowler.bowler_type

    # Type merit
    if fan_type == preferred[0]:
        type_score = 35
    elif len(preferred) > 1 and fan_type == preferred[1]:
        type_score = 22
    elif fan_type == cap_type:
        type_score = 18
    else:
        type_score = 8

    # Captain match bonus
    captain_bonus = 10 if fan_bowler.id == captain_bowler.id else 0

    return {
        "type_score": type_score,
        "captain_bonus": captain_bonus,
        "wicket_bonus": 0,   # filled in post-over
        "total": type_score + captain_bonus,
    }


def score_field_placement(fan_field: FieldPlacement, captain_field: FieldPlacement,
                           benchmark_key: str) -> dict:
    """
    Score the fan's field placement (0–50 pts).

    Components:
    - Merit score       (0–30): similarity to the benchmark ideal field
    - Captain similarity(0–20): similarity to captain's actual field
    """
    benchmark = BENCHMARKS.get(benchmark_key)
    bench_field = benchmark["field"] if benchmark else FieldPlacement()

    def zone_vec(fp: FieldPlacement) -> list[int]:
        return [fp.slip_cordon, fp.point_cover, fp.mid_off, fp.mid_on,
                fp.square_leg, fp.fine_leg, fp.deep_offside, fp.deep_onside]

    fan_vec   = zone_vec(fan_field)
    bench_vec = zone_vec(bench_field)
    cap_vec   = zone_vec(captain_field)

    # L1 distance normalised by max possible divergence (≈ 20)
    merit_dist = sum(abs(a - b) for a, b in zip(fan_vec, bench_vec))
    merit_score = max(0, round(30 - merit_dist * 2.5))

    # Captain similarity
    cap_dist    = sum(abs(a - b) for a, b in zip(fan_vec, cap_vec))
    cap_score   = max(0, round(20 - cap_dist * 2.5))

    return {
        "merit_score":   merit_score,
        "captain_score": cap_score,
        "total": merit_score + cap_score,
    }


def compute_full_score(bowler_scores: dict, field_scores: dict,
                        over_result: OverResult, fan_bowler_id: str) -> dict:
    """Assemble the complete per-over score."""
    # Wicket bonus
    wicket_bonus = 0
    if over_result.wickets > 0 and fan_bowler_id == over_result.bowler_id:
        wicket_bonus = 5

    bowler_scores["wicket_bonus"] = wicket_bonus
    bowler_total = bowler_scores["type_score"] + bowler_scores["captain_bonus"] + wicket_bonus
    bowler_scores["total"] = bowler_total

    total = bowler_total + field_scores["total"]

    return {
        "bowler": bowler_scores,
        "field":  field_scores,
        "total":  total,
        "grade":  "S" if total >= 85 else "A" if total >= 70 else "B" if total >= 50 else "C",
    }


# ─────────────────────────────────────────────────────────────
# Ball Simulation
# ─────────────────────────────────────────────────────────────

COMMENTARY_TEMPLATES = {
    0: [
        "{bowler} beats {batsman} outside off! Dot ball.",
        "{bowler} nails a perfect yorker — dot!",
        "Good length from {bowler}, defended solidly by {batsman}.",
        "{batsman} misses the pull — dot ball from {bowler}.",
    ],
    1: [
        "{batsman} works {bowler} to mid-wicket for a single.",
        "Nudged off the pads by {batsman} — one run.",
        "{batsman} pushes {bowler} towards cover for a single.",
        "A quick single taken by {batsman}.",
    ],
    2: [
        "{batsman} drives through the gap — two runs!",
        "Good running between the wickets — two!",
        "{batsman} cuts {bowler} to third man — a couple.",
    ],
    3: [
        "Excellent placement by {batsman} — three runs!",
        "Poor fielding! Three from a misfield.",
    ],
    4: [
        "{batsman} drives {bowler} through the covers — FOUR!",
        "FOUR! {batsman} pulls that over mid-wicket!",
        "Beautifully timed by {batsman} — four through extra cover!",
        "{bowler} overpitches and {batsman} drives hard — FOUR!",
        "Swept to the fine-leg boundary — FOUR!",
    ],
    6: [
        "SIX! {batsman} launches {bowler} over long-on!",
        "MASSIVE HIT! {batsman} deposits {bowler} into the stands!",
        "{batsman} slog-sweeps {bowler} — SIX over square leg!",
        "SIX! {batsman} goes over the top — what a shot!",
    ],
    "W": [
        "WICKET! {bowler} gets {batsman} — caught in the deep!",
        "WICKET! {batsman} holes out to mid-off off {bowler}!",
        "CLEAN BOWLED! {bowler} rattles {batsman}'s stumps!",
        "OUT! {batsman} edges {bowler} to slip — brilliant!",
        "LBW! Plumb in front — {batsman} has to go off {bowler}!",
    ],
}

def _commentary(runs_or_wicket, bowler_name: str, batsman_name: str) -> str:
    key = "W" if runs_or_wicket == "W" else runs_or_wicket
    templates = COMMENTARY_TEMPLATES.get(key, ["{bowler} to {batsman}."])
    return random.choice(templates).format(bowler=bowler_name, batsman=batsman_name)


def simulate_ball(bowler: Player, striker: Player,
                  phase: Phase, over_num: int) -> BallResult:
    """Simulate one delivery. Returns a BallResult."""

    # Wicket probability modifiers
    wicket_prob = bowler.wicket_prob
    if phase == Phase.POWERPLAY:
        wicket_prob *= 0.6    # fewer wickets in powerplay
    if phase == Phase.DEATH:
        wicket_prob *= 1.4    # more wickets in death overs

    # Check wicket first
    if random.random() < wicket_prob:
        commentary = _commentary("W", bowler.name, striker.name)
        return BallResult(0, 0, True, commentary)

    # Run distribution — vary by batsman type and phase
    if striker.batting_type == BattingType.AGGRESSOR:
        if phase == Phase.DEATH:
            weights = [20, 18, 8, 4, 22, 16, 12]   # 0,1,2,3,4,6,dot
        elif phase == Phase.POWERPLAY:
            weights = [25, 20, 10, 4, 22, 12,  7]
        else:
            weights = [28, 22, 10, 4, 18, 10,  8]
    else:  # ANCHOR
        if phase == Phase.DEATH:
            weights = [25, 25, 12, 5, 18, 10,  5]
        else:
            weights = [35, 30, 12, 4, 12,  5,  2]

    runs_options = [0, 1, 2, 3, 4, 6, 0]   # last 0 = "dot ball" category
    run_weights  = weights

    runs = random.choices(runs_options, weights=run_weights, k=1)[0]
    commentary = _commentary(runs, bowler.name, striker.name)
    return BallResult(0, runs, False, commentary)


# ─────────────────────────────────────────────────────────────
# Captain Heuristic
# ─────────────────────────────────────────────────────────────

def captain_choose_bowler(bowlers: list[Player], state: MatchState,
                           striker: Player) -> Player:
    """
    Simple captain heuristic for bowler selection.
    Picks the most appropriate available bowler for the phase,
    avoiding the previous bowler and respecting over limits.
    """
    available = [b for b in bowlers if b.can_bowl() and b.id != state.last_bowler_id]
    if not available:
        # Fallback: anyone who can still bowl
        available = [b for b in bowlers if b.can_bowl()]

    phase = state.phase

    # Prioritise bowler types by phase
    if phase == Phase.POWERPLAY:
        preferred_types = [BowlerType.PACE_FAST, BowlerType.PACE_MEDIUM]
    elif phase == Phase.DEATH:
        preferred_types = [BowlerType.PACE_FAST, BowlerType.PACE_MEDIUM]
    else:
        # Middle overs — use spinner if available and match situation warrants
        if state.wickets < 3 and state.run_rate < 8.5:
            preferred_types = [BowlerType.OFF_SPIN, BowlerType.LEG_SPIN,
                               BowlerType.PACE_MEDIUM, BowlerType.PACE_FAST]
        else:
            preferred_types = [BowlerType.PACE_FAST, BowlerType.PACE_MEDIUM,
                               BowlerType.OFF_SPIN, BowlerType.LEG_SPIN]

    for ptype in preferred_types:
        candidates = [b for b in available if b.bowler_type == ptype]
        if candidates:
            # Among candidates, pick best economy
            return min(candidates, key=lambda b: b.match_economy())

    # Fallback: best available economy
    return min(available, key=lambda b: b.match_economy())


def captain_set_field(bowler: Player, state: MatchState,
                       striker: Player) -> FieldPlacement:
    """Return the captain's field placement using the benchmark for the situation."""
    key = get_benchmark_key(bowler, state, striker)
    bench = BENCHMARKS.get(key)
    if not bench:
        return FieldPlacement()

    # Add a tiny bit of variance to make it feel realistic
    base = deepcopy(bench["field"])

    # Small random adjustment (±1 fielder in one zone)
    if random.random() < 0.35:
        zones = ["deep_offside", "deep_onside", "point_cover"]
        zone = random.choice(zones)
        delta = random.choice([-1, 1])
        val = getattr(base, zone)
        new_val = max(0, min(3, val + delta))
        setattr(base, zone, new_val)
        # Compensate to keep total = 10
        diff = base.total() - 10
        if diff != 0:
            comp_zone = "deep_onside" if zone != "deep_onside" else "deep_offside"
            comp_val = getattr(base, comp_zone)
            setattr(base, comp_zone, max(0, comp_val - diff))

    return base


# ─────────────────────────────────────────────────────────────
# Over Simulator
# ─────────────────────────────────────────────────────────────

def simulate_over(bowler: Player, batting_players: dict[str, Player],
                  state: MatchState, captain_field: FieldPlacement) -> OverResult:
    """Simulate a full over (6 deliveries) and return the result."""
    balls: list[BallResult] = []
    total_runs = 0
    wickets = 0
    striker_id = state.current_striker_id

    for ball_num in range(1, 7):
        striker = batting_players[striker_id]
        result  = simulate_ball(bowler, striker, state.phase, state.current_over)
        result.ball_num = ball_num

        balls.append(result)

        if result.is_wicket:
            wickets += 1
            striker.is_out = True
            # Bring in next batsman from batting order
            order = state.batting_order
            for pid in order:
                if pid not in (state.current_striker_id, state.current_nonstriker_id):
                    p = batting_players.get(pid)
                    if p and not p.is_out and pid != striker_id:
                        striker_id = pid
                        break
        else:
            total_runs += result.runs
            striker.runs_scored += result.runs
            striker.balls_faced  += 1
            # Rotate strike on odd runs
            if result.runs % 2 == 1:
                striker_id, state.current_nonstriker_id = \
                    state.current_nonstriker_id, striker_id

    # End-of-over: non-striker becomes striker
    state.current_striker_id    = state.current_nonstriker_id
    state.current_nonstriker_id = striker_id

    # Update bowler stats
    bowler.overs_bowled += 1
    bowler.runs_given   += total_runs
    bowler.wickets_taken += wickets

    return OverResult(
        over_num      = state.current_over,
        bowler_id     = bowler.id,
        total_runs    = total_runs,
        wickets       = wickets,
        balls         = balls,
        captain_field = captain_field,
    )


# ─────────────────────────────────────────────────────────────
# Match Factory
# ─────────────────────────────────────────────────────────────

def init_match() -> tuple[MatchState, dict, dict, dict]:
    """
    Initialise a new match.
    Returns (state, all_players_by_id, batting_dict, bowling_dict)
    where batting/bowling dicts map player_id → Player for each side.
    """
    teams = build_teams()

    # Team MM bats first, CC bowls
    batting_side  = {p.id: p for p in teams["mm"]}
    bowling_side  = {p.id: p for p in teams["cc"]}
    all_players   = {**batting_side, **bowling_side}

    # Batting order for MM
    batting_order = ["mm_rohit", "mm_prithvi", "mm_surya", "mm_tilak",
                     "mm_hardik", "mm_tim", "mm_ishan", "mm_krunal",
                     "mm_bumrah", "mm_trent", "mm_piyush"]

    state = MatchState(
        team_batting_id      = "mm",
        team_bowling_id      = "cc",
        batting_order        = batting_order,
        current_striker_id   = batting_order[0],
        current_nonstriker_id= batting_order[1],
    )

    return state, all_players, batting_side, bowling_side
