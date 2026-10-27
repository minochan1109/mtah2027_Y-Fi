"""[D] Pseudo labels of version 2 (algorithmic only -- no person decides any of them).

Differences from version 1
  * only TRUSTED ball detections (ball_quality) not in flight can create a control event
  * team membership per frame (team_mat), so a corrected goalkeeper counts for the right team
  * "possession stays with the last team until the other team controls the ball" (CHALLENGE.md):
    after a control event of team k, the following frames keep k until the next control event
    (of either team) starts, for at most possession_hold_s, and never across a dead ball
"""
import numpy as np

from ..preprocess import run_lengths
from .ball_quality import TRUSTED


def control_events_v2(prep, team_mat, bq, P):
    T = prep["n_frames"]
    px, py = prep["px"], prep["py"]
    ok_ball = (bq["status"] == TRUSTED) & (bq["conf"] >= 0.6) & ~bq["flight"]
    bx, by = bq["bx"], bq["by"]
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    rows = np.arange(T)
    best = []
    for k in (0, 1):
        d = np.where(team_mat == k, dist, np.inf)
        i = d.argmin(axis=1)
        best.append((d[rows, i], i))
    out = np.full(T, -1, np.int8)
    for k in (0, 1):
        d_own, i = best[k]
        d_opp, _ = best[1 - k]
        rel = np.hypot(prep["pvx"][rows, i] - prep["bvx"], prep["pvy"][rows, i] - prep["bvy"])
        ok = (ok_ball & (d_own < P.control_dist_m) & (d_opp > P.control_opp_min_dist_m)
              & (np.nan_to_num(rel, nan=99) < P.control_rel_speed)
              & (np.nan_to_num(prep["bspeed"], nan=99) < P.control_ball_speed_max))
        for s, e in run_lengths(ok):
            if e - s >= P.frames(P.control_min_s):
                out[s:e] = k
    return out


def team_labels_v2(control, dead_y, P):
    """1 = home, 0 = away, -1 unknown. Control frames + 'hold until the other team controls'."""
    T = len(control)
    y = np.full(T, -1, np.int8)
    y[control == 0] = 1
    y[control == 1] = 0
    hold = P.frames(P.possession_hold_s)
    runs = [(s, e, int(control[s])) for s, e in run_lengths(control >= 0)]
    for (s, e, k), nxt in zip(runs, runs[1:] + [(T, T, -1)]):
        stop = min(nxt[0], e + hold)
        seg = np.arange(e, stop)
        dead = np.flatnonzero(dead_y[seg] == 1)
        if len(dead):
            seg = seg[:dead[0]]
        y[seg] = 1 if k == 0 else 0
    y[dead_y == 1] = -1
    return y
