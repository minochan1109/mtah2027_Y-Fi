"""Every tunable number in one place.

RULE REMINDER (RULES.md, updated 2026-10-23)
--------------------------------------------
* No hand labels on the provided game, for any purpose (not for training, not for validation).
* The public 5-minute ground truth is for checking the pipeline only: NOT training data and
  NOT a tuning set. Nothing in this file may be chosen by looking at its score.
* No rules or constants tied to specific frames of the game. Everything below is expressed in
  seconds, metres or probabilities, and applies to any match.

The defaults are football-generic starting points. Change them only from reasoning, from
unsupervised statistics of the match itself, or from labelled data on OTHER games.
"""
from dataclasses import dataclass, field


@dataclass
class Params:
    # ---- data -------------------------------------------------------------------------------
    fps: int = 25
    pitch_length: float = 105.3          # metres (games.py)
    pitch_width: float = 68.0
    image_size: tuple = (1920, 1080)     # tactical video, for the camera aim point

    # ---- preprocessing ----------------------------------------------------------------------
    velocity_half_window_s: float = 0.2  # centred difference over +/- this many seconds
    ball_max_interp_gap_s: float = 0.5   # ball gaps up to this long are linearly interpolated
    out_of_pitch_margin_m: float = 0.5   # ball beyond the line by more than this = outside

    # ---- feature windows (seconds, past and future) ------------------------------------------
    windows_s: tuple = (0.5, 1.0, 2.0)
    near_radius_m: float = 2.0           # "players within radius of the ball"
    region_radius_m: float = 10.0        # "ball region" used when the ball is not visible
    sprint_speed: float = 5.0            # m/s
    player_max_speed: float = 7.0        # m/s, time-to-ball model
    reaction_time_s: float = 0.3

    # ---- pseudo labels (algorithmic, no human decisions) -------------------------------------
    control_dist_m: float = 1.2          # player this close to the ball ...
    control_rel_speed: float = 2.0       # ... moving with it (m/s difference) ...
    control_ball_speed_max: float = 8.0  # ... and the ball is not flying away (m/s)
    control_opp_min_dist_m: float = 2.5  # nearest opponent at least this far
    control_min_s: float = 0.4           # sustained this long
    dead_walk_speed: float = 1.6         # mean player speed below this = walking (m/s)
    dead_min_s: float = 3.0              # walking for at least this long
    alive_run_speed: float = 3.0         # mean player speed above this = clearly live play
    set_piece_lead_s: float = 4.0        # frames just before a tagged set piece are likely dead

    # ---- state model (module 4) ----------------------------------------------------------------
    # Costs are in nats (negative log probability) paid once per switch.
    cost_switch_team: float = 12.0       # H <-> A while the ball is alive
    cost_enter_dead: float = 8.0         # H/A -> D
    cost_leave_dead: float = 8.0         # D -> H/A
    min_dead_s: float = 2.0              # a stoppage lasts at least this long
    prob_floor: float = 1e-4             # emissions are clipped to [floor, 1 - floor]
    tagging_weight: float = 0.15         # how much a Smart Tagging team tag shifts P(H)

    # ---- post-processing ------------------------------------------------------------------------
    transition_shift_frames: int = 0     # systematic lag correction; NOT tuned on the public GT

    # ---- models -------------------------------------------------------------------------------------
    cv_block_s: float = 60.0             # time blocks for cross-validation (seconds)
    n_folds: int = 5
    self_training_rounds: int = 1        # 0 = no self-training
    self_training_conf: float = 0.9      # add frames whose predicted prob exceeds this
    lgbm: dict = field(default_factory=lambda: dict(
        n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
        random_state=0, deterministic=True, force_row_wise=True))   # same file on every run

    def frames(self, seconds):
        return max(1, int(round(seconds * self.fps)))
