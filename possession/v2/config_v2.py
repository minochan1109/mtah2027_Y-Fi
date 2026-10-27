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
    keeper_zone_depth_m: float = 6.0     # goalkeeper zone: this close to the goal line ...
    keeper_zone_half_width_m: float = 10.0   # ... and this close to the centre line of the goal
    keeper_min_s: float = 3.0            # wrong-team "keeper" must persist this long to be re-assigned
    keeper_max_speed: float = 3.0        # ... and move slower than this on average
    max_per_team: int = 11

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
