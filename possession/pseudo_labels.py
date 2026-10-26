"""Algorithmic pseudo labels: the only labels used for training on the provided game.

Every label here is produced by a fixed rule on the data. No person decides any of them, which
is what RULES.md allows ("pseudo-labels produced by an algorithm"). Values: 1 / 0 / -1 (unknown).
"""
import numpy as np

from .preprocess import rolling, run_lengths


def control_events(prep, P):
    """Per frame: team (0/1) clearly controlling the ball, or -1.

    A team controls the ball when its nearest player is very close, moving with the ball, the
    ball is not flying away, and no opponent is close -- for at least control_min_s.
    """
    T = prep["n_frames"]
    px, py, team = prep["px"], prep["py"], prep["team"]
    bx, by = prep["bx"], prep["by"]
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    dist = np.where(np.isnan(dist), np.inf, dist)
    rows = np.arange(T)
    out = np.full(T, -1, np.int8)
    best = []
    for k in (0, 1):
        d = np.where(team[None, :] == k, dist, np.inf)
        i = d.argmin(axis=1)
        best.append((d[rows, i], i))
    for k in (0, 1):
        d_own, i = best[k]
        d_opp, _ = best[1 - k]
        rel = np.hypot(prep["pvx"][rows, i] - prep["bvx"], prep["pvy"][rows, i] - prep["bvy"])
        ok = (prep["ball_seen"] & (d_own < P.control_dist_m) & (d_opp > P.control_opp_min_dist_m)
              & (np.nan_to_num(rel, nan=99) < P.control_rel_speed)
              & (np.nan_to_num(prep["bspeed"], nan=99) < P.control_ball_speed_max))
        for s, e in run_lengths(ok):
            if e - s >= P.frames(P.control_min_s):
                out[s:e] = k
    return out


def dead_labels(dead_feat, P, control=None):
    """1 = confidently dead, 0 = confidently alive, -1 = unknown.

    Dead needs BOTH a calm game (everybody walking for dead_min_s) AND a ball-side reason:
    the ball left the pitch, a set piece is about to be tagged, or the ball is static/unseen.
    Alive: somebody controls the ball, the ball is travelling inside the pitch, or players run.
    """
    T = len(dead_feat)
    ms = rolling(dead_feat["speed_mean"].values, P.fps, "centre")
    walking = ms < P.dead_walk_speed
    calm = np.zeros(T, bool)
    for s, e in run_lengths(walking):
        if e - s >= P.frames(P.dead_min_s):
            calm[s:e] = True
    since = np.nan_to_num(dead_feat["since_seen"].values, nan=99.0)
    bspeed = np.nan_to_num(dead_feat["ball_speed"].values, nan=0.0)
    out_recent = (dead_feat["last_ball_out"].values > 0)
    before_sp = dead_feat["to_set_piece_s"].values < P.set_piece_lead_s
    ball_still = (since > 2.0) | (bspeed < 0.5)
    y = np.full(T, -1, np.int8)
    y[calm & (out_recent | before_sp | ball_still)] = 1
    inside = dead_feat["last_ball_out"].values == 0
    travelling = (dead_feat["ball_seen"].values > 0) & (bspeed > 3.0) & inside
    alive = travelling | (ms > P.alive_run_speed)
    if control is not None:
        alive |= control >= 0
    y[alive & (y != 1)] = 0
    return y


def team_labels(control, dead_y, P, max_fill_s=8.0):
    """1 = home in possession, 0 = away, -1 = unknown (alive frames only).

    Control events are labelled directly. The frames BETWEEN two control events of the same team
    are filled with that team when the gap is short and nothing dead happens in between -- this
    is what gives module 3 labelled frames where the ball is not visible.
    """
    T = len(control)
    y = np.full(T, -1, np.int8)
    y[control == 0] = 1
    y[control == 1] = 0
    idx = np.flatnonzero(control >= 0)
    if len(idx) < 2:
        return y
    gaps = np.flatnonzero(np.diff(idx) > 1)
    for g in gaps:
        a, b = idx[g], idx[g + 1]
        if control[a] == control[b] and (b - a) <= P.frames(max_fill_s) and not (dead_y[a:b] == 1).any():
            y[a:b] = 1 if control[a] == 0 else 0
    y[dead_y == 1] = -1
    return y
