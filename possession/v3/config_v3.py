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
