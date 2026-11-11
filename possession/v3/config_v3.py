"""Every number of version 3 that is not in version 2. Params3 = Params2 + these fields.

The defaults are the decided version (MT-11 / MT-24). None of them was fitted to the evaluation games.
"""
from dataclasses import dataclass

from ..v2.config_v2 import Params2


@dataclass
class Params3(Params2):
    # ---- [D] the play-stopped model (its P(dead) is the "core" of the hybrid detector) ----------
    dead_ball_motion: bool = False       # False: never use how the ball moves (speed, in flight, still)
    dead_set_piece_tags: bool = False    # False: ignore the SET PIECES tags (goal tags are kept)

    # ---- [E] H/A ----------------------------------------------------------------------------------
    home_calib_a: float = 1.0            # P(home) calibration sigmoid(a * logit(p) + b); 1 / 0 = off
    home_calib_b: float = 0.0

    # ---- [C2] kick-off preprocessing (髙橋's kickoff.py, MT-xx): find each kick-off and the goal before it
    #      from the players, BEFORE D is decided. Laws-of-the-Game values, nothing read off this match.
    kickoff_pre: bool = True             # False: the hybrid D detector uses its own tag-based kick-off search
    ko_search_before_s: float = 5.0      # search window around a kick-off tag ...
    ko_search_after_s: float = 20.0      # ... (wide on purpose: the offset is FOUND, not assumed)
    ko_half_margin_m: float = 1.0        # "own half" tolerance (tracking noise on the halfway line)
    ko_half_frac: float = 0.8            # this fraction of each team in its own half
    ko_spot_m: float = 5.0               # "on the centre spot"
    ko_circle_margin_m: float = 1.0      # opponents outside 9.15 m minus this
    ko_circle_frac: float = 0.8          # this fraction of opponents outside the circle
    ko_still_speed: float = 1.0          # m/s: mean player speed while waiting for the whistle
    ko_min_players: int = 7              # per team, so a half-tracked frame cannot pass
    ko_formation_min_s: float = 1.0      # formation must hold this long
    ko_refine_s: float = 2.0             # look for the movement onset this far around its end
    ko_move_speed: float = 2.0           # m/s: mean speed above this = play has started
    ko_goal_pre_s: float = 5.0           # "before the goal" reference window
    ko_celebrate_s: float = 30.0         # celebration window after the goal
    ko_celebrate_scale_m: float = 10.0   # metres away from the spot worth one "player-fraction"
    ko_celebrate_min_score: float = 0.2  # below this the pattern does not name a scorer
    ko_return_m: float = 25.0            # conceding team "back near the centre": mean distance
    ko_after_goal_max_s: float = 150.0   # without a kick-off tag: search this long after a goal
    ko_goal_pair_max_s: float = 360.0    # a kick-off tag this long after a goal still belongs to it
    ko_goal_search_s: float = 360.0      # look back at most this far for the goal (never past the previous kick-off)
    ko_goal_min_step: float = 0.15       # the 'just after a goal' signal must step up by this (0..1)
    ko_goal_min_asym: float = 0.1        # ... and be ONE-SIDED (0 = both teams alike = a stoppage, not a goal)
    ko_goal_ball_before_s: float = 30.0  # look for the ball in the goal mouth this long before the step
    ko_goal_mouth_m: float = 5.0         # "in the goal mouth": this close to the goal line ...
    ko_goal_half_width_m: float = 5.0    # ... and this close to the middle of the goal
    ko_goal_attack_m: float = 16.5       # no ball: a scorer must have been this close to the goal (the box)
    ko_tag_match_s: float = 60.0         # a GOALS tag this close to a found goal = the same goal
    ko_goal_moment: str = "stop"         # the goal moment reported: shot / onset / stop
    ko_goal_third: str = "oneside"       # goal signal, 3rd part: oneside (hump of one-sided movement) / none / still
    ko_goal_move_pct: float = 75.0       # still only: the conceders' in-play speed = this percentile in the window
    ko_goal_still_frac: float = 0.5      # still only: "not playing" = below this share of that speed
    ko_dead_bonus: float = 3.0           # (v2 HMM only, unused in v3)
    ko_live_bonus: float = 3.0
    ko_team_bonus: float = 1.0
    ko_live_s: float = 2.0
