"""Every number of version 3 that is not in version 2. Params3 = Params2 + these fields.

The defaults are the decided version (MT-11 / MT-24). None of them was fitted to the evaluation games.
"""
from dataclasses import dataclass, field

from ..v2.config_v2 import Params2


@dataclass
class Params3(Params2):
    # ---- [D] the play-stopped model (its P(dead) is the "core" of the hybrid detector) ----------
    dead_ball_motion: bool = False       # False: never use how the ball moves (speed, in flight, still)
    dead_set_piece_tags: bool = False    # False: ignore the SET PIECES tags (goal tags are kept)

    # ---- [D] H/A team model trained on other games' answer keys (team_external.py, --team-train) ---
    # chosen on the external EVAL games only (leave-one-game-out), never on the provided game
    team_model_external: str = "single"  # one LightGBM on every feature ("additive" = base + ball add-on)
    lgbm_external: dict = field(default_factory=lambda: dict(
        n_estimators=1200, learning_rate=0.03, num_leaves=31, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
        random_state=0, deterministic=True, force_row_wise=True))   # same file on every run
    team_lasttouch: bool = True          # add "which team controlled the ball last" features
    lt_dist_m: float = 1.5               # control: a player this close to the ball ...
    lt_opp_m: float = 1.0                # ... the nearest opponent at least this far ...
    lt_min_s: float = 0.2                # ... for at least this long
    lt_cap_s: float = 20.0               # "seconds since the last control" is capped here

    # ---- [E] H/A ----------------------------------------------------------------------------------
    home_calib_a: float = 1.0            # P(home) calibration sigmoid(a * logit(p) + b); 1 / 0 = off
    home_calib_b: float = 0.0
    goal_lead_s: float = 0.0             # this long before every GOALS tag = the scoring team's possession (0 = off)
                                         # (a goal comes from the scoring team's shot; if used, 2 s: on EVAL it has
                                         #  the fewest wrong frames of 0.5-4 s, spellF1 0.6146 vs 0.6151 off)
    goal_lead_shift_s: float = 0.0       # ... ending this long AFTER the tag start (not used: shifts did not help)
    restart_goal_len_s: float = 2.0      # a restart (PK, direct FK) that scores at once: this long after the kick
                                         # = the scorer's possession (0 = off)
    restart_goal_before_s: float = 3.0   # ... when a GOALS tag lies from this long before the kick ...
    restart_goal_after_s: float = 5.0    # ... to this long after it
