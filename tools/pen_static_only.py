"""Only one rule: remove detections that stand (almost) still for a while near the penalty area of the
goal their team ATTACKS (= the penalty area that is NOT on their own side). Nothing else (no keeper
handling, no team fix, no 11-per-team limit, own penalty area is left alone).

    python tools/pen_static_only.py --games external/EVAL-G1 external/EVAL-G2
    python tools/pen_static_only.py --games external/EVAL-G1 external/EVAL-G2 --still-s 5 --gap-s 0.5

"near the penalty area" = penalty area + --margin metres around it.
Score (against truth/errors.npz made by tools/make_eval_data.py), same definitions as eval_cleaning.py.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from possession import data, preprocess  # noqa: E402
from possession.v2.config_v2 import Params2  # noqa: E402

ERR_FALSE, ERR_SWAP_IN = 2, 3


def _bridge_gaps(mask, max_gap):
    """Set to True the False-runs of at most max_gap frames that have True on both sides."""
    out = mask.copy()
    if max_gap <= 0:
        return out
    for a, b in preprocess.run_lengths(~mask):
        if b - a <= max_gap and a > 0 and b < len(mask):
            out[a:b] = True
    return out


def remove_still_near_attacked_penalty(px, py, team, attack, P, still_s=10.0, margin_m=10.0, gap_s=0.0):
    """px, py: (T, S) metres, NaN = not tracked.  team: (S,) 0/1 team of each slot.
    attack: (T, 2), attack[:, k] = +1 when team k attacks +x (its own goal is x = 0), -1 otherwise.
    Returns (px, py) with the removed ones set to NaN, and the (T, S) boolean mask of what was removed.
    gap_s: a "still" run may be interrupted by up to this many seconds of faster frames (speed noise).
    0 = strict: every single frame of the run must be below the speed threshold."""
    L, W = P.pitch_length, P.pitch_width
    hw = P.frames(P.velocity_half_window_s)
    speed = np.hypot(preprocess.velocity(px, hw, P.fps), preprocess.velocity(py, hw, P.fps))
    still = np.isfinite(px) & (np.nan_to_num(speed, nan=0.0) < P.static_speed)   # (almost) not moving

    depth = P.penalty_depth_m + margin_m                                          # from the goal line
    half_w = P.penalty_half_width_m + margin_m                                    # from the goal centre
    lateral = np.abs(py - W / 2) < half_w
    near0 = (px < depth) & lateral                                                # around the goal at x = 0
    nearL = (px > L - depth) & lateral                                            # around the goal at x = L

    # the goal each detection's team attacks: x = L when its team attacks +x, else x = 0
    atk = np.where(team[None, :] == 0, attack[:, [0]], attack[:, [1]])            # (T, S)
    attacked_area = np.where(atk > 0, nearL, near0)

    cand = still & attacked_area                 # still AND near the attacked penalty area, frame by frame
    removed = np.zeros_like(cand)
    min_frames = still_s * P.fps
    gap = int(round(gap_s * P.fps))
    for s in range(cand.shape[1]):
        c = _bridge_gaps(cand[:, s], gap)
        for a, b in preprocess.run_lengths(c):
            if b - a >= min_frames:              # the whole run must last long enough
                removed[a:b, s] = True
    px, py = px.copy(), py.copy()
    px[removed] = py[removed] = np.nan
    return px, py, removed


def evaluate(folder, still_s, margin_m, gap_s=0.0):
    with open(os.path.join(folder, "game.json")) as fh:
        cfg = json.load(fh)
    P = Params2()
    P.pitch_length, P.pitch_width = cfg["pitch"]
    n = int(cfg["n_frames"])
    stem = os.path.join(folder, cfg.get("input_dir", ""), cfg["stem"])
    tr = data.load_tracking(stem + "_atd.xml", stem + "_atd.txt", n, tuple(cfg["pitch"]))
    err = np.load(os.path.join(folder, "truth", "errors.npz"))
    blocks = [tuple(b) for b in cfg["blocks"]]
    attack = preprocess.attack_sign(tr, blocks, P.fps)

    _, _, removed = remove_still_near_attacked_penalty(
        tr.px.astype(np.float64), tr.py.astype(np.float64), tr.team, attack, P, still_s, margin_m, gap_s)
    present = np.isfinite(tr.px)
    kept = present & ~removed
    # this rule never touches the team: every detection keeps the team of its slot
    team_mat = np.repeat(tr.team[None, :].astype(np.int8), n, axis=0)
    state, src = err["player_state"], err["player_src"]
    true_team = err["true_team"]
    tt = np.where(src >= 0, true_team[np.clip(src, 0, None)], -1)
    false_obj = present & (state == ERR_FALSE)
    real = present & (src >= 0)
    swap = present & (state == ERR_SWAP_IN)
    return {                                         # same definitions as tools/eval_cleaning.py
        "false_removed": (false_obj & ~kept).sum() / max(1, false_obj.sum()),
        "real_removed": (real & ~kept).sum() / max(1, real.sum()),
        "swaps_fixed": (swap & kept & (team_mat == tt)).sum() / max(1, swap.sum()),
        "team_correct": (real & kept & (team_mat == tt)).sum() / max(1, (real & kept).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--still-s", type=float, default=10.0, help="seconds standing still")
    ap.add_argument("--margin", type=float, default=10.0, help="metres around the penalty area")
    ap.add_argument("--gap-s", type=float, default=0.0, help="allowed interruption inside a still run (s)")
    a = ap.parse_args()
    print(f"rule: still >= {a.still_s} s near the ATTACKED penalty area (+{a.margin} m), gap allowed {a.gap_s} s")
    keys = ["false_removed", "real_removed", "swaps_fixed", "team_correct"]
    print("PLAYERS  " + "".join(f"{k:>14}" for k in keys))
    for g in a.games:
        r = evaluate(g, a.still_s, a.margin, a.gap_s)
        print(f"{os.path.basename(os.path.normpath(g)):<9}" + "".join(f"{r[k]:14.3f}" for k in keys))
    print("higher is better except real_removed")


if __name__ == "__main__":
    main()
