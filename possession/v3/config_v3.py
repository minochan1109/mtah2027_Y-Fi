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
    dead_external: bool = True           # True: P(dead) learned from other games' true D (--team-train files
                                         # + their *_dead.npz) instead of this game's pseudo labels
    dead_train_unseen_s: float = 3.0     # >0: in those training labels, play with the answer's ball unknown
                                         # this long = D (event-data answers miss some stoppages)

    # ---- [D] H/A team model trained on other games' answer keys (team_external.py, --team-train) ---
    # chosen on the external EVAL games only (leave-one-game-out), never on the provided game
    team_model_external: str = "single"  # one LightGBM on every feature ("additive" = base + ball add-on)
    lgbm_external: dict = field(default_factory=lambda: dict(
        n_estimators=1200, learning_rate=0.03, num_leaves=31, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
        random_state=0, deterministic=True, force_row_wise=True))   # same file on every run
    team_seeds: int = 5                  # average P(home) of random seeds 0..4 (one seed alone moves spellF1
                                         # by about 0.01 on EVAL; 1 = a single model with random_state above)
    team_lasttouch: bool = True         # add "which team controlled the ball last" features
    team_self_train_conf: float = 0.0    # >0: retrain adding this game's frames with P >= this (no answers used)
    team_self_train_weight: float = 0.5  # ... at this weight
    team_blend_pseudo: float = 0.0       # >0: blend with this game's pseudo-label model (version 2) at this weight
    team_nexttouch: bool = False        # ... and "which team controls the ball next" (needs team npz built with it)
    lt_dist_m: float = 1.5               # control: a player this close to the ball ...
    lt_opp_m: float = 1.0                # ... the nearest opponent at least this far ...
    lt_min_s: float = 0.2                # ... for at least this long
    lt_cap_s: float = 20.0               # "seconds since the last control" is capped here
    team_smooth_s: float = 0.5           # P(home) of the team model averaged over this window (centred; 0 = off)
    team_smooth_unseen_s: float = 0.64   # longer centered window without trusted ball tracking
                                         # (EVAL, leave-one-game-out: one-frame flips stop splitting possessions)

    # ---- [E] H/A ----------------------------------------------------------------------------------
    home_calib_a: float = 1.0            # P(home) calibration sigmoid(a * logit(p) + b); 1 / 0 = off
    home_calib_b: float = 0.0
    restart_team_delay_s: float = 2.0   # delay only the restarting team's H/A evidence
    cost_switch_team: float = 13.0      # H <-> A switch cost in play before recovery evidence
    switch_signal_mid: float = 0.8      # recovery onset at the midpoint of the switch signal
    switch_signal_drop: float = 0.75    # maximum fractional discount when recovery is clear
    unseen_ball_switch_cost_scale: float = 1.5  # keep the last team when the ball position is unreliable
    ball_flight_switch_cost_scale: float = 1.25  # keep possession through tracked ball flight
    ball_flight_switch_tail_s: float = 0.12     # continue the flight penalty for this long
    switch_model: bool = False           # True: the switch signal = P(switch) learned from other games (--team-train)
    switch_model_win_s: float = 0.4      # "switch moment" = within this long of a true H<->A switch
    switch_model_scale: float = 2.0      # signal = min(1, scale * P(switch))
    switch_evidence_max_s: float = 1.6   # a possession this short or shorter ... (0 = off)
    switch_evidence_m: float = 0.75      # ... in which no player of that team is ever this close to the (trusted)
                                         # ball, nearer than every opponent, is not a possession: the team before
                                         # it keeps the ball (a switch needs a touch; fixes switches while the ball
                                         # is in flight / loose)
    turnover_backtrack_s: float = 2.5    # look for sustained control before a predicted switch
    turnover_control_s: float = 0.4     # require continuous ball proximity for at least this long
    restore_brief_turnovers: bool = True  # recover short possessions bracketed by ball control
    turnover_slowdown_backtrack_s: float = 1.4  # look for a slowdown before a predicted switch
    turnover_redirect_search_s: float = 0.16  # look for a redirection after a predicted switch
    weak_turnover_merge_s: float = 4.0  # bridge unsupported reversals of at most this duration
    recover_short_ball_out: bool = True  # restore short ball-out stops corroborated by a restart tag
    stop_control_search_s: float = 0.6  # delay a stoppage onset when control continues
    sustained_contact_log_odds: float = 0.0   # weak H/A decoding vote for sustained ball control (0 = off: +0.0003 on EVAL)
    goal_lead_s: float = 0.0             # this long before every GOALS tag = the scoring team's possession (0 = off)
                                         # (a goal comes from the scoring team's shot; if used, 2 s: on EVAL it has
                                         #  the fewest wrong frames of 0.5-4 s, spellF1 0.6146 vs 0.6151 off)
    goal_lead_shift_s: float = 0.0       # ... ending this long AFTER the tag start (not used: shifts did not help)
    restart_goal_len_s: float = 2.0      # a restart (PK, direct FK) that scores at once: this long after the kick
                                         # = the scorer's possession (0 = off)
    restart_goal_before_s: float = 3.0   # ... when a GOALS tag lies from this long before the kick ...
    restart_goal_after_s: float = 5.0    # ... to this long after it

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
