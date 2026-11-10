"""Every number used by version 2, grouped by the file that uses it.

Same rules as config.py: generic football values, changed only from reasoning, from statistics of
the match that use no labels, or from scores on OTHER games (tools/convert_metrica.py + --external).
"""
from dataclasses import dataclass

from ..config import Params


@dataclass
class Params2(Params):
    # ---- [A] clean_players.py -----------------------------------------------------------------
    goal_line_margin_m: float = 3.0      # further than this behind a goal line = not a player
    side_line_margin_m: float = 3.0      # further than this outside a touch line = not a player
    static_speed: float = 0.3            # m/s: "not moving"
    static_near_goal_s: float = 15.0     # still this long within static_goal_zone_m of a goal line = remove
    static_goal_zone_m: float = 2.0
    static_anywhere_s: float = 60.0      # still this long anywhere = remove
    static_live_speed: float = 1.0       # ... but only while the other players move (not in a pause)
    penalty_depth_m: float = 16.5        # penalty area depth (from the goal line)
    penalty_half_width_m: float = 20.16  # penalty area half width (40.32 m / 2)
    penalty_margin_m: float = 10.0       # ... plus this margin around it
    static_pen_s: float = 10.0           # still this long in the penalty area + margin (both ends) = remove
    event_skip: bool = False             # rule 1 ignores frames inside set-piece / goal tags (needs the tags)
    event_pad_s: float = 0.0             # ... widened by this many s before the tag starts
    static_opp_pen_s: float = 5.0        # still this long near the penalty area of the goal the team attacks = remove
    static_gap_s: float = 0.0            # a still run may be interrupted by this many s of speed noise (0 = strict)
    use_keeper_gt: bool = True          # use the known goalkeeper box (if given) to fix team swaps
    use_keeper_candidate: bool = True    # no GT box: protect the deepest object of the keeper zone from rules 1 and 2
    det_box_m: float = 1.0               # nominal detection box size (m) around a player position
    keeper_iou_min: float = 0.1          # min overlap with the GT keeper box to be the keeper
    keeper_dup_iou: float = 0.5          # another detection this close to the GT box = duplicate, removed
    keeper_zone_depth_m: float = 10.0    # goalkeeper zone: this close to the goal line ...
    keeper_zone_half_width_m: float = 10.0   # ... and this close to the centre line of the goal
    keeper_min_s: float = 3.0            # wrong-team "keeper" must persist this long to be re-assigned
    keeper_max_speed: float = 3.0        # ... and move slower than this on average
    keeper_follow_depth_m: float = 20.0  # a re-assigned keeper keeps the team while within this of his goal line (0 = off)
    keeper_follow_gap_s: float = 1.0     # ... across tracking gaps up to this long
    keeper_follow_max_mps: float = 12.0  # ... but a jump faster than this between frames = another identity
    max_per_team: int = 11
    lone_gk: bool = True                 # ONE detection alone in the penalty area (+ margin) = keeper
    lone_margin_m: float = 10.0          # ... penalty area plus this margin
    lone_gk_s: float = 0.0               # ... for at least this long (0 = as soon as it is alone)
    drop_rule: str = "still"             # which detection goes first when a team has more than max_per_team:
                                         #   "still" | "motion" | "overlap" | "both"
    drop_window_s: float = 3.0           # motion: window (+- s) in which the movement is measured
    drop_overlap_m: float = 1.0          # both: a team mate closer than this = duplicate, dropped first
    swap_link: bool = False              # a short track starting/ending next to another slot's track gets its team
    swap_max_s: float = 30.0             # only tracks up to this long are re-assigned
    swap_link_gap_s: float = 0.2         # ... the other track ends/starts within this many s
    swap_link_dist_m: float = 1.5        # ... and within this distance
    swap_need_both: bool = False         # True: both ends (before and after) must link

    # ---- [B] ball_quality.py ------------------------------------------------------------------
    ball_outside_reject_m: float = 5.0   # detection this far outside the pitch = rejected
    ball_link_gap_s: float = 0.2         # detections this close in time can belong to one tracklet
    ball_max_speed: float = 40.0         # m/s; faster = cannot be the same ball
    ball_link_margin_m: float = 1.0
    ball_spike_frames: int = 2           # tracklets this short are spikes = rejected
    ball_short_s: float = 0.2            # short tracklet ...
    ball_isolated_m: float = 5.0         # ... with no player this close = rejected
    ball_static_s: float = 4.0           # tracklet not moving for this long ...
    ball_static_spread_m: float = 0.5
    ball_static_no_player_m: float = 4.0 # ... with no player near it ...
    ball_static_live_speed: float = 2.0  # ... while players run = rejected (spot, line, spare ball)
    ball_interp_max_s: float = 1.0       # gaps between trusted detections filled up to this long
    ball_interp_max_speed: float = 35.0
    ball_flight_speed: float = 12.0      # m/s: faster than this ...
    ball_flight_min_s: float = 0.3       # ... for this long ...
    ball_flight_player_m: float = 1.5    # ... with no player this close = ball in the air (probably)
    ball_flight_conf: float = 0.3        # confidence is multiplied by this during a flight
    ball_conf_min: float = 0.3           # ball features are blanked below this confidence

    # ---- [C] tag_policy.py --------------------------------------------------------------------
    tag_logit_scale: float = 1.0         # overall strength of tag evidence (0 = ignore tags)

    # ---- [D] features_v2.py / pseudo_v2.py / model_v2.py ---------------------------------------
    recover_speed: float = 4.0           # m/s towards own goal = running back
    recover_onset_s: float = 1.0         # "started running back within the last ... s"
    reversal_speed: float = 1.0          # team centroid speed (m/s) counted as forward / backward
    containment_pct: float = 10.0        # team box = this..(100-this) percentile of x and y
    possession_hold_s: float = 4.0       # pseudo labels: keep the last controlling team this long
    team_model: str = "additive"         # "additive" or "single"
    ball_weight: float = 1.0             # additive mode: weight of the ball add-on

    # ---- [E] hmm_v2.py --------------------------------------------------------------------------
    switch_signal_drop: float = 0.5      # switch cost x (1 - drop * signal), signal in 0..1
    switch_signal_mid: float = 2.0       # signal = sigmoid(slope * (recover_onset_diff - mid))
    switch_signal_slope: float = 1.5