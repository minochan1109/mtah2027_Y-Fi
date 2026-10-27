"""[A] Remove false player detections and fix the goalkeeper's team.

Input : Tracking (team per SLOT), attack direction per frame (from preprocess.attack_sign).
Output: px, py with false detections set to NaN, and team_mat (T, S): the team of each slot IN
        EACH FRAME (0 home, 1 away, -1 removed). Later modules use team_mat instead of the
        fixed slot team, so a goalkeeper that the tracker put in the wrong team can be corrected.

Steps (all generic rules, the same for every match):
  1. outside   : far behind a goal line or outside a touch line
  2. static    : not moving for a long time near a goal line (goal post, net, ball boy ...)
                 or anywhere for a very long time
  3. keeper    : the deepest object in a goalkeeper zone belongs to the WRONG team for a while
                 -> it is the goalkeeper, give it the defending team
  4. max 11    : still more than 11 in a team -> drop the most suspicious (stillest) ones
"""
import numpy as np

from ..preprocess import run_lengths, velocity


def _still_run_seconds(still, fps):
    """(T, S): length in seconds of the still-run each frame belongs to (0 when moving)."""
    out = np.zeros(still.shape)
    for s in range(still.shape[1]):
        for a, b in run_lengths(still[:, s]):
            out[a:b, s] = (b - a) / fps
    return out


def clean_players(tr, attack, P):
    T, S = tr.px.shape
    L, W = P.pitch_length, P.pitch_width
    px = tr.px.astype(np.float64).copy()
    py = tr.py.astype(np.float64).copy()
    team_mat = np.repeat(tr.team[None, :].astype(np.int8), T, axis=0)
    team_mat[np.isnan(px)] = -1
    rep = {"tracked_before": int(np.isfinite(px).sum())}

    # 1. outside the pitch
    out = ((px < -P.goal_line_margin_m) | (px > L + P.goal_line_margin_m)
           | (py < -P.side_line_margin_m) | (py > W + P.side_line_margin_m))
    rep["removed_outside"] = int(out.sum())
    px[out] = py[out] = np.nan

    # 2. static objects
    hw = P.frames(P.velocity_half_window_s)
    sp = np.hypot(velocity(px, hw, P.fps), velocity(py, hw, P.fps))
    still = np.isfinite(px) & (np.nan_to_num(sp, nan=0.0) < P.static_speed)
    still_s = _still_run_seconds(still, P.fps)
    near_goal = np.minimum(np.abs(px), np.abs(L - px)) < P.static_goal_zone_m
    with np.errstate(invalid="ignore"):
        live = np.nanmean(sp, axis=1) > P.static_live_speed          # the others are playing
    static = np.zeros_like(still)
    for s in range(S):
        for a, b in run_lengths(still[:, s]):
            dur = (b - a) / P.fps
            if not live[a:b].mean() > 0.5:
                continue                                              # whole game paused: keep
            if dur >= P.static_anywhere_s or (dur >= P.static_near_goal_s and near_goal[a:b, s].mean() > 0.5):
                static[a:b, s] = True
    rep["removed_static"] = int(static.sum())
    px[static] = py[static] = np.nan
    team_mat[np.isnan(px)] = -1

    # 3. goalkeeper on the wrong team
    # team k attacks +x (attack[:,k] = +1) -> it defends the goal at x = 0
    reassigned = 0
    for end_x in (0.0, L):
        depth = np.abs(px - end_x)                                  # distance to that goal line
        in_zone = (depth < P.keeper_zone_depth_m) & (np.abs(py - W / 2) < P.keeper_zone_half_width_m)
        d = np.where(np.isfinite(depth), depth, np.inf)
        deepest = d.argmin(axis=1)
        rows = np.arange(T)
        defend = np.where(end_x == 0.0, (attack[:, 1] > 0).astype(int), (attack[:, 1] < 0).astype(int))
        # defend[t] = team that defends this goal (0 or 1)
        slot_team = team_mat[rows, deepest]
        # nobody of the defending team in the zone, and the deepest object there is "attacking"
        def_in_zone = (in_zone & (team_mat == defend[:, None])).any(axis=1)
        wrong = in_zone[rows, deepest] & (slot_team >= 0) & (slot_team != defend) & ~def_in_zone
        for a, b in run_lengths(wrong):
            if b - a < P.frames(P.keeper_min_s):
                continue
            seg = np.arange(a, b)
            if np.nanmean(sp[seg, deepest[seg]]) > P.keeper_max_speed:
                continue
            team_mat[seg, deepest[seg]] = defend[seg]
            reassigned += b - a
    rep["keeper_frames_reassigned"] = int(reassigned)

    # 4. at most max_per_team per team per frame: drop the stillest
    still_s = np.where(np.isfinite(px), still_s, -1)
    dropped = 0
    excess_frames = 0
    for k in (0, 1):
        cnt = (team_mat == k).sum(axis=1)
        for t in np.flatnonzero(cnt > P.max_per_team):
            excess_frames += 1
            slots = np.flatnonzero(team_mat[t] == k)
            order = slots[np.argsort(-still_s[t, slots], kind="stable")]
            drop = order[: cnt[t] - P.max_per_team]
            px[t, drop] = py[t, drop] = np.nan
            team_mat[t, drop] = -1
            dropped += len(drop)
    rep["frames_over_max"] = int(excess_frames)
    rep["removed_over_max"] = int(dropped)
    rep["tracked_after"] = int(np.isfinite(px).sum())
    return px, py, team_mat, rep
