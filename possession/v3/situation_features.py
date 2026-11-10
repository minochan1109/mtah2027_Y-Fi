"""Per-frame features for the situation-based dead-ball detector. PLAYERS AND TAGS ONLY: no ball.

Input: cleaned player positions px, py (T, slots) in metres (0..L, 0..W), team_mat (T, slots) with
0 = home, 1 = away, -1 = no player, attack (T, 2) = +1 if that team attacks towards +x, and the
Smart Tagging list from tag_policy.load_tags. Every column is a plain number per frame so it can be
used in the expressions of params/dead_situations.txt (see tools/dead_situations.py).

Columns (all speeds in m/s, distances in m, times in s)
  movement     sp_mean, sp_max, sp2 (2nd fastest), n_sprint (>5), n_fast_home / n_fast_away (> 3, per team), n_dash_home / n_dash_away (> 5.5, per
               team), n_fast (>3), frac_walk (<2),
               n_decel / n_accel (speed fell / rose by > 1.5 within 1 s), cen_speed (speed of the centre
               of all players), team_back (max over teams of the mean speed towards its own goal),
               team_fwd (same, towards the opponent goal), spread (mean distance to the centre),
               d_spread (spread now - 2 s ago)
  places       n_touch_slow (slow players on/over a side line), taker_side_dist (distance of the slow
               player closest to a side line, negative = outside), n_near_taker (players <= 15 m from
               him), n_goalline (players on/over a goal
               line), n_corner_slow (slow players <= 3 m from a corner flag), n_box_max (players in the
               fullest penalty box), n_box_att (players of the attacking team in that box), gk_6yd
               (a keeper-like deepest player standing in his 6-yard box), opp_in_gk_box (opponents in
               that keeper's penalty box), own_half_min (lowest share of a team inside its own half;
               1 = kick-off shape), n_centre (players <= 9.15 m from the centre spot), n_wall (largest
               group of slow team-mates standing <= 1.5 m from each other), wall_goal_dist (distance
               from that group to the goal it protects)
  ball (position only, never its movement; only when a trusted position exists)
               ball_known (a position now), ball_out (last known position, at most 10 s old, is outside
               the pitch by > 0.5 m), ball_out_side / ball_out_goal (outside a side / goal line),
               ball_corner (last position <= 2 m from a corner flag), ball_centre (<= 1.5 m from the
               centre spot), ball_6yd (inside a 6-yard box), ball_age_s (seconds since the last position),
               n_approach_ball (players > 3 m from the ball running towards it; n_approach_home /
               n_approach_away per team), ball_still (1 still for 1 s / 0 / -1 unknown), ball_fresh (position
               <= 1 s old), keeper_ball (a keeper inside his box has the ball), ball_moving (the ball
               position travels > 5 m/s; use starts(ball_moving > 0, s) to count moves), ball_pk_spot (last
               position <= 1 m from a penalty spot)
  substitution near_side, n_sub_zone (players at the halfway line on the touch line near the camera),
               sub_in_out (a player appears / disappears there)
  picture      centre_visible / corner_visible (1 / 0 / -1 unknown: the centre spot / a corner flag is in
               the picture)
  goal kick    kick_end, opp_in_kick_box (opponents in the kicking side's penalty box), n_in_kick_area
               (players inside the kicking side's goal area),
               kick_area_visible (1 / 0 / -1 unknown: the kicking side's goal area is in the picture)
  tags         tag_sp_to_start_s / tag_sp_since_start_s / tag_sp_to_end_s / tag_sp_since_end_s
               (SET PIECES tags), tag_since_goal_s, tag_throw / tag_corner / tag_goalkick / tag_fk /
               tag_ko / tag_pk (1 when the nearest SET PIECES tag, from 30 s before it to 1 s after it, has that
               Type text)
"""
import numpy as np
import pandas as pd

CAP = 120.0


def _smooth(a, k):
    return pd.DataFrame(a).rolling(k, center=True, min_periods=max(2, k // 3)).mean().to_numpy()


def _shift(a, k):
    """a[t - k] (k > 0 looks back), NaN outside."""
    out = np.full_like(a, np.nan, dtype=float)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def _runs_bool(m):
    d = np.diff(np.r_[False, m, False].astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _nanmean(a, axis=1):
    with np.errstate(invalid="ignore"):
        return np.nanmean(a, axis=axis)


def player_features(px, py, team_mat, attack, pitch, fps=25):
    L, W = pitch
    T, S = px.shape
    f = {}
    k = max(3, int(0.4 * fps) | 1)
    sx, sy = _smooth(px, k), _smooth(py, k)
    h = int(0.2 * fps)
    vx = (_shift(sx, -h) - _shift(sx, h)) / (2 * h / fps)
    vy = (_shift(sy, -h) - _shift(sy, h)) / (2 * h / fps)
    spd = np.hypot(vx, vy)
    spd[~np.isfinite(px)] = np.nan
    spd = np.clip(spd, 0, 12)
    present = np.isfinite(spd)
    n_present = np.maximum(present.sum(1), 1)

    f["sp_mean"] = _nanmean(spd)
    srt = np.sort(np.nan_to_num(spd, nan=-1.0), axis=1)
    f["sp_max"] = np.clip(srt[:, -1], 0, None)
    f["sp2"] = np.clip(srt[:, -2], 0, None)
    f["n_sprint"] = (np.nan_to_num(spd) > 5.0).sum(1).astype(float)
    fast = np.nan_to_num(spd) > 3.0
    f["n_fast_home"] = (fast & (team_mat == 0)).sum(1).astype(float)
    f["n_fast_away"] = (fast & (team_mat == 1)).sum(1).astype(float)
    dash = np.nan_to_num(spd) > 5.5                                   # "dash"
    f["n_dash_home"] = (dash & (team_mat == 0)).sum(1).astype(float)
    f["n_dash_away"] = (dash & (team_mat == 1)).sum(1).astype(float)
    f["n_fast"] = (np.nan_to_num(spd) > 3.0).sum(1).astype(float)
    f["frac_walk"] = ((spd < 2.0) & present).sum(1) / n_present
    ds = spd - _shift(spd, fps)
    f["n_decel"] = (np.nan_to_num(ds) < -1.5).sum(1).astype(float)
    f["n_accel"] = (np.nan_to_num(ds) > 1.5).sum(1).astype(float)

    cx, cy = _nanmean(sx), _nanmean(sy)
    cvx = (_shift(cx, -h) - _shift(cx, h)) / (2 * h / fps)
    cvy = (_shift(cy, -h) - _shift(cy, h)) / (2 * h / fps)
    f["cen_speed"] = np.nan_to_num(np.hypot(cvx, cvy))
    dist_c = np.hypot(sx - cx[:, None], sy - cy[:, None])
    f["spread"] = _nanmean(dist_c)
    f["d_spread"] = np.nan_to_num(f["spread"] - _shift(f["spread"], 2 * fps))

    back, fwd = [], []
    for tm in (0, 1):
        m = team_mat == tm
        towards_opp = vx * attack[:, tm][:, None]
        back.append(_nanmean(np.where(m, -towards_opp, np.nan)))
        fwd.append(_nanmean(np.where(m, towards_opp, np.nan)))
    f["team_back"] = np.nan_to_num(np.nanmax(np.c_[back[0], back[1]], axis=1))
    f["team_fwd"] = np.nan_to_num(np.nanmax(np.c_[fwd[0], fwd[1]], axis=1))

    slow = np.nan_to_num(spd, nan=99) < 2.0
    side = (py < 1.0) | (py > W - 1.0)
    f["n_touch_slow"] = (side & slow).sum(1).astype(float)
    # throw-in taker candidate: the slow player closest to a side line (negative = outside)
    dside = np.where(slow, np.minimum(py, W - py), np.inf)
    j = np.argmin(dside, axis=1)
    r = np.arange(T)
    f["taker_side_dist"] = np.where(np.isfinite(dside[r, j]), dside[r, j], 99.0)
    dj = np.hypot(px - px[r, j][:, None], py - py[r, j][:, None])
    f["n_near_taker"] = np.where(np.isfinite(dside[r, j]), (np.nan_to_num(dj, nan=99) < 15).sum(1) - 1, 0)
    f["n_goalline"] = ((px < 0.5) | (px > L - 0.5)).sum(1).astype(float)
    corners = [(0, 0), (0, W), (L, 0), (L, W)]
    dcorner = np.min([np.hypot(px - a, py - b) for a, b in corners], axis=0)
    f["n_corner_slow"] = ((dcorner < 3.0) & slow).sum(1).astype(float)

    in_y = np.abs(py - W / 2) < 20.16
    box_lo, box_hi = in_y & (px < 16.5), in_y & (px > L - 16.5)
    n_lo, n_hi = box_lo.sum(1), box_hi.sum(1)
    f["n_box_max"] = np.maximum(n_lo, n_hi).astype(float)
    # attacking team at an end = the team that attacks towards it
    att_hi = np.where(attack[:, 0] > 0, 0, 1)        # team attacking the +x goal
    att_lo = 1 - att_hi
    na_hi = (box_hi & (team_mat == att_hi[:, None])).sum(1)
    na_lo = (box_lo & (team_mat == att_lo[:, None])).sum(1)
    f["n_box_att"] = np.where(n_hi >= n_lo, na_hi, na_lo).astype(float)

    # keeper-like player: deepest player of the team defending that end
    gk6, opp_box = np.zeros(T), np.full(T, 99.0)
    for end in ("lo", "hi"):
        dfd = att_hi if end == "lo" else att_lo        # team defending this end
        depth = px if end == "lo" else L - px
        mine = team_mat == dfd[:, None]
        dmask = np.where(mine, depth, np.inf)
        j = np.argmin(dmask, axis=1)
        r = np.arange(T)
        dj, yj, sj = dmask[r, j], py[r, j], np.nan_to_num(spd[r, j], nan=99)
        ok = np.isfinite(dj) & (dj < 5.5) & (np.abs(yj - W / 2) < 9.16) & (sj < 1.5)
        box = box_lo if end == "lo" else box_hi
        opp = (box & ~mine & (team_mat >= 0)).sum(1).astype(float)
        gk6 = np.where(ok, 1.0, gk6)
        opp_box = np.where(ok, np.minimum(opp_box, opp), opp_box)
    f["gk_6yd"] = gk6
    f["opp_in_gk_box"] = np.where(gk6 > 0, opp_box, 99.0)

    own = []
    for tm in (0, 1):
        m = team_mat == tm
        in_own = ((px - L / 2) * attack[:, tm][:, None]) < 0
        own.append((m & in_own).sum(1) / np.maximum(m.sum(1), 1))
    f["own_half_min"] = np.minimum(own[0], own[1])
    # kick-off shape: players more than 0.5 m inside the OTHER half, the larger of the two teams
    wrong = [((team_mat == tm) & (((px - L / 2) * attack[:, tm][:, None]) > 0.5)).sum(1) for tm in (0, 1)]
    f["n_wrong_half"] = np.maximum(wrong[0], wrong[1]).astype(float)
    f["n_centre"] = (np.hypot(px - L / 2, py - W / 2) < 9.15).sum(1).astype(float)

    # penalty kick: in one penalty box only the keeper and the taker (a player near the penalty spot),
    # the keeper on his goal line, the others waiting just outside the box
    pk_two, pk_edge, gk_line = np.zeros(T), np.zeros(T), np.zeros(T)
    r = np.arange(T)
    for end in ("lo", "hi"):
        depth = px if end == "lo" else L - px
        dfd = att_hi if end == "lo" else att_lo
        inb = (depth < 16.5) & (depth > -1.0) & (np.abs(py - W / 2) < 20.16)
        n_in = inb.sum(1)
        dm = np.where(team_mat == dfd[:, None], depth, np.inf)
        jj = np.argmin(dm, axis=1)
        keeper_in = np.isfinite(dm[r, jj]) & (dm[r, jj] < 16.5)
        on_line = np.isfinite(dm[r, jj]) & (dm[r, jj] < 1.5) & (np.abs(py[r, jj] - W / 2) < 4.0)
        attacker = (team_mat >= 0) & (team_mat != dfd[:, None])
        taker = (attacker & (np.hypot(depth - 11.0, py - W / 2) < 2.0)).any(1)     # attacker at the spot
        edge = ((depth >= 16.5) & (depth < 25.0) & (np.abs(py - W / 2) < 26.0)).sum(1)
        pk_two = np.where((n_in <= 2) & taker & keeper_in, 1.0, pk_two)
        pk_edge = np.where((n_in <= 3) & taker, np.maximum(pk_edge, edge), pk_edge)
        gk_line = np.where(on_line & taker, 1.0, gk_line)
    f["pk_two"] = pk_two
    f["n_pk_edge"] = pk_edge
    f["gk_on_line"] = gk_line

    # wall: largest group of slow team-mates <= 1.5 m apart (connected through neighbours, 1 step)
    n_wall, wall_d = np.zeros(T), np.full(T, 99.0)
    step = 4000
    for a in range(0, T, step):
        b = min(T, a + step)
        x, y, tmx, sl = px[a:b], py[a:b], team_mat[a:b], slow[a:b]
        d = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
        same = (tmx[:, :, None] == tmx[:, None, :]) & (tmx[:, :, None] >= 0)
        both = sl[:, :, None] & sl[:, None, :]
        adj = (np.nan_to_num(d, nan=99) < 1.5) & same & both
        adj[:, np.arange(S), np.arange(S)] = False
        deg = adj.sum(2)                                   # neighbours of each player
        j = np.argmax(deg, axis=1)
        r = np.arange(b - a)
        nw = np.where(deg[r, j] > 0, deg[r, j] + 1, 0)
        n_wall[a:b] = nw
        xj = x[r, j]
        gx = np.where(np.abs(xj) < np.abs(L - xj), 0.0, L)
        wall_d[a:b] = np.where(nw > 0, np.hypot(xj - gx, y[r, j] - W / 2), 99.0)
    f["n_wall"] = n_wall
    f["wall_goal_dist"] = wall_d
    return {k: np.nan_to_num(np.asarray(v, float), nan=0.0) for k, v in f.items()}


def _type_of(tag):
    t = " ".join(txt.lower() for g, txt in tag["labels"] if g.lower() == "type")
    if "throw" in t:
        return "throw"
    if "corner" in t:
        return "corner"
    if "goal kick" in t:
        return "goalkick"
    if "penalty" in t or t.strip() in ("pk", "pk kick"):
        return "pk"
    if "free kick" in t:
        return "fk"
    if "kick off" in t or "kick-off" in t:
        return "ko"
    return "other"


def tag_features(tags, T, fps=25):
    t = np.arange(T)
    f = {k: np.full(T, CAP) for k in ("tag_sp_to_start_s", "tag_sp_since_start_s", "tag_sp_to_end_s",
                                       "tag_sp_since_end_s", "tag_since_goal_s")}
    for k in ("tag_throw", "tag_corner", "tag_goalkick", "tag_fk", "tag_ko", "tag_pk"):
        f[k] = np.zeros(T)
    best = np.full(T, np.inf)
    for g in tags or []:
        s, e = int(g["start"]), int(g["end"])
        if g["code"] == "GOALS":
            m = t >= s
            f["tag_since_goal_s"][m] = np.minimum(f["tag_since_goal_s"][m], (t[m] - s) / fps)
            continue
        if g["code"] != "SET PIECES":
            continue
        for key, ref, sign in (("tag_sp_to_start_s", s, +1), ("tag_sp_since_start_s", s, -1),
                               ("tag_sp_to_end_s", e, +1), ("tag_sp_since_end_s", e, -1)):
            m = (t <= ref) if sign > 0 else (t >= ref)
            f[key][m] = np.minimum(f[key][m], sign * (ref - t[m]) / fps)
        a, b = max(0, s - 30 * fps), min(T, e + fps)
        dist = np.abs(t[a:b] - s)
        win = dist < best[a:b]
        best[a:b] = np.where(win, dist, best[a:b])
        ty = _type_of(g)
        for k in ("throw", "corner", "goalkick", "fk", "ko", "pk"):
            col = f["tag_" + k]
            col[a:b] = np.where(win, float(ty == k), col[a:b])
    return {k: np.minimum(v, CAP) for k, v in f.items()}


def ball_features(bx, by, pitch, fps=25, hold_s=10.0):
    """Where the ball is / was last seen. Movement (speed, flight, still) is deliberately not used."""
    L, W = pitch
    T = len(bx)
    known = np.isfinite(bx)
    idx = np.where(known, np.arange(T), -1)
    idx = np.maximum.accumulate(idx)
    age = np.where(idx >= 0, (np.arange(T) - idx) / fps, CAP)
    ok = (idx >= 0) & (age <= hold_s)
    lx = np.where(ok, bx[np.clip(idx, 0, None)], np.nan)
    ly = np.where(ok, by[np.clip(idx, 0, None)], np.nan)
    with np.errstate(invalid="ignore"):
        out_side = (ly < -0.5) | (ly > W + 0.5)
        out_goal = (lx < -0.5) | (lx > L + 0.5)
        dc = np.min([np.hypot(lx - a, ly - b) for a, b in [(0, 0), (0, W), (L, 0), (L, W)]], axis=0)
        six = (np.abs(ly - W / 2) < 9.16) & ((lx < 5.5) | (lx > L - 5.5)) & (lx > -0.5) & (lx < L + 0.5)
        f = {"ball_known": known.astype(float), "ball_out": (out_side | out_goal).astype(float),
             "ball_out_side": out_side.astype(float), "ball_out_goal": out_goal.astype(float),
             "ball_corner": (dc < 2.0).astype(float),
             "ball_centre": (np.hypot(lx - L / 2, ly - W / 2) < 1.5).astype(float),
             "ball_6yd": six.astype(float), "ball_age_s": np.minimum(age, CAP),
             "ball_pk_spot": (np.minimum(np.hypot(lx - 11.0, ly - W / 2),
                                         np.hypot(lx - (L - 11.0), ly - W / 2)) < 1.0).astype(float)}
    return f


def ball_player_features(px, py, team_mat, attack, bx, by, pitch, fps=25, near_m=3.0, approach_mps=1.5,
                         keeper_m=2.0):
    """Players around the last known ball POSITION (the ball's own movement is not used).

    n_approach_ball  players further than near_m from the ball who move towards it faster than
                     approach_mps (radial speed); 0 while nobody comes for the ball
    ball_fresh       1 when the ball position is at most 1 s old
    keeper_ball      1 when the ball is <= keeper_m from a keeper-like player (deepest player of the team
                     defending that end) standing inside his own penalty box -> the keeper has the ball
    """
    L, W = pitch
    T = len(bx)
    known = np.isfinite(bx)
    idx = np.maximum.accumulate(np.where(known, np.arange(T), -1))
    age = np.where(idx >= 0, (np.arange(T) - idx) / fps, CAP)
    lx = np.where(idx >= 0, bx[np.clip(idx, 0, None)], np.nan)
    ly = np.where(idx >= 0, by[np.clip(idx, 0, None)], np.nan)
    k = max(3, int(0.4 * fps) | 1)
    sx, sy = _smooth(px, k), _smooth(py, k)
    h = int(0.2 * fps)
    vx = (_shift(sx, -h) - _shift(sx, h)) / (2 * h / fps)
    vy = (_shift(sy, -h) - _shift(sy, h)) / (2 * h / fps)
    dx, dy = lx[:, None] - px, ly[:, None] - py
    d = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        radial = (vx * dx + vy * dy) / d                       # > 0 = moving towards the ball
        appr = (d > near_m) & (radial > approach_mps)
    appr = np.nan_to_num(appr).astype(bool)
    f = {"n_approach_ball": np.where(np.isfinite(lx), appr.sum(1), 99).astype(float),
         "n_approach_home": np.where(np.isfinite(lx), (appr & (team_mat == 0)).sum(1), 99).astype(float),
         "n_approach_away": np.where(np.isfinite(lx), (appr & (team_mat == 1)).sum(1), 99).astype(float),
         "ball_fresh": (age <= 1.0).astype(float)}
    # ball_still: the ball position stays within 0.3 m for 1 s (seen in >= 60 % of that second);
    # 1 = still, 0 = not still, -1 = unknown (ball not seen enough)
    w = fps
    seen = pd.Series(known.astype(float)).rolling(w, min_periods=1).mean().to_numpy()
    rx = pd.Series(bx).rolling(w, min_periods=2)
    ry = pd.Series(by).rolling(w, min_periods=2)
    spread_b = np.hypot((rx.max() - rx.min()).to_numpy(), (ry.max() - ry.min()).to_numpy())
    f["ball_still"] = np.where(seen >= 0.6, (np.nan_to_num(spread_b, nan=99) <= 0.3).astype(float), -1.0)
    # ball_moving: the ball (trusted positions only) travels faster than 5 m/s for >= 0.2 s; runs closer
    # than 0.5 s are one move. Used only as "the ball moved twice -> play" (a stoppage has at most one).
    hb = int(0.2 * fps)
    with np.errstate(invalid="ignore"):
        bsp = np.hypot(_shift(bx, -hb) - _shift(bx, hb), _shift(by, -hb) - _shift(by, hb)) / (2 * hb / fps)
    mv = np.nan_to_num(bsp) > 5.0
    for s0, e0 in _runs_bool(~mv):
        if 0 < s0 and e0 < T and e0 - s0 < int(0.5 * fps):
            mv[s0:e0] = True
    for s0, e0 in _runs_bool(mv):
        if e0 - s0 < hb:
            mv[s0:e0] = False
    f["ball_moving"] = mv.astype(float)
    att_hi = np.where(attack[:, 0] > 0, 0, 1)                   # team attacking the +x goal (defends x = 0)
    kb = np.zeros(T)
    r = np.arange(T)
    for end in ("lo", "hi"):
        dfd = att_hi if end == "lo" else 1 - att_hi
        depth = px if end == "lo" else L - px
        dm = np.where(team_mat == dfd[:, None], depth, np.inf)
        j = np.argmin(dm, axis=1)
        dj = dm[r, j]
        in_box = np.isfinite(dj) & (dj < 16.5) & (np.abs(py[r, j] - W / 2) < 20.16)
        near = np.hypot(lx - px[r, j], ly - py[r, j]) <= keeper_m
        kb = np.where(in_box & np.nan_to_num(near) & (age <= 1.0), 1.0, kb)
    f["keeper_ball"] = kb
    return f


def area_visible(H, valid, pitch, x0, x1, y0, y1, img_w=1920, img_h=1080, need=3):
    """1 = the pitch rectangle [x0,x1] x [y0,y1] (metres) is in the picture (at least `need` of its 4
    corners and centre project inside the image), 0 = not, -1 = unknown (no valid camera model)."""
    L, W = pitch
    T = len(H)
    pts = [(x0, y0), (x0, y1), (x1, y0), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2)]
    inside = np.zeros(T, int)
    ok = np.asarray(valid, bool) & np.isfinite(H).all(axis=(1, 2))
    for x, y in pts:
        u_, v_ = x / L, y / W
        den = H[:, 2, 0] * u_ + H[:, 2, 1] * v_ + H[:, 2, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            u = (H[:, 0, 0] * u_ + H[:, 0, 1] * v_ + H[:, 0, 2]) / den
            v = (H[:, 1, 0] * u_ + H[:, 1, 1] * v_ + H[:, 1, 2]) / den
        inside += (np.nan_to_num(den) > 0) & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    return np.where(ok, (inside >= need).astype(float), -1.0)


def goal_kick_features(px, py, team_mat, attack, bx, by, H, valid, pitch, fps=25):
    """Which end a goal kick would be taken from, opponents in that penalty box, and whether its goal
    area is in the picture.

    kick_end           0 = the x = 0 end, 1 = the x = L end, -1 = none. From the last ball POSITION
                       (<= 10 s old) in a goal area or behind a goal line; otherwise the end whose keeper
                       stands in his goal area (the one with fewer opponents in the box if both)
    opp_in_kick_box    players of the attacking team inside that penalty box (99 = no kick end)
    kick_area_visible  1 / 0 = that goal area is / is not in the picture, -1 = unknown (no camera model,
                       e.g. the evaluation games) -> treated as "not known to be missing"
    """
    L, W = pitch
    T = len(px)
    r = np.arange(T)
    att_hi = np.where(attack[:, 0] > 0, 0, 1)
    k6, opp, in6 = [], [], []
    for end in (0, 1):
        depth = px if end == 0 else L - px
        dfd = att_hi if end == 0 else 1 - att_hi
        dm = np.where(team_mat == dfd[:, None], depth, np.inf)
        j = np.argmin(dm, axis=1)
        k6.append(np.isfinite(dm[r, j]) & (dm[r, j] < 5.5) & (np.abs(py[r, j] - W / 2) < 9.16))
        in6.append(((depth < 5.5) & (depth > -0.5) & (np.abs(py - W / 2) < 9.16)).sum(1))
        inb = (depth < 16.5) & (depth > -1.0) & (np.abs(py - W / 2) < 20.16)
        opp.append((inb & (team_mat >= 0) & (team_mat != dfd[:, None])).sum(1))
    end = np.full(T, -1)
    end = np.where(k6[0] & ~k6[1], 0, end)
    end = np.where(k6[1] & ~k6[0], 1, end)
    both = k6[0] & k6[1]
    end = np.where(both, np.where(opp[0] <= opp[1], 0, 1), end)
    if bx is not None:
        known = np.isfinite(bx)
        idx = np.maximum.accumulate(np.where(known, np.arange(T), -1))
        fresh = (idx >= 0) & ((np.arange(T) - idx) <= 10 * fps)
        lx = np.where(fresh, bx[np.clip(idx, 0, None)], np.nan)
        ly = np.where(fresh, by[np.clip(idx, 0, None)], np.nan)
        with np.errstate(invalid="ignore"):
            iny = np.abs(ly - W / 2) < 9.16
            lo = (lx < -0.5) | (iny & (lx < 5.5))
            hi = (lx > L + 0.5) | (iny & (lx > L - 5.5))
        end = np.where(lo, 0, np.where(hi, 1, end))
    f = {"kick_end": end.astype(float),
         "opp_in_kick_box": np.where(end == 0, opp[0], np.where(end == 1, opp[1], 99)).astype(float),
         "n_in_kick_area": np.where(end == 0, in6[0], np.where(end == 1, in6[1], 0)).astype(float)}
    if H is None:
        f["kick_area_visible"] = np.full(T, -1.0)
    else:
        v0 = area_visible(H, valid, pitch, 0.0, 5.5, W / 2 - 9.16, W / 2 + 9.16)
        v1 = area_visible(H, valid, pitch, L - 5.5, L, W / 2 - 9.16, W / 2 + 9.16)
        f["kick_area_visible"] = np.where(end == 0, v0, np.where(end == 1, v1, -1.0))
    return f


def substitution_features(px, py, pitch, H=None, valid=None, fps=25):
    """Substitutions happen at the halfway line on the touch line NEAR the camera.

    near_side       0 = the y = 0 touch line is nearer the camera, 1 = the y = W one, -1 = unknown (both
                    sides are then used)
    n_sub_zone      players within 8 m of the halfway line and <= 3 m inside / 5 m outside that touch line
    sub_in_out      1 in a frame where a player appears or disappears within 10 m of that point
    """
    L, W = pitch
    T = len(px)
    near = -1
    if H is not None and valid is not None and np.any(valid):
        ok = np.asarray(valid, bool) & np.isfinite(H).all(axis=(1, 2))
        v = []
        for yy in (0.0, 1.0):
            den = H[ok, 2, 0] * 0.5 + H[ok, 2, 1] * yy + H[ok, 2, 2]
            v.append((H[ok, 1, 0] * 0.5 + H[ok, 1, 1] * yy + H[ok, 1, 2]) / den)
        if len(v[0]):
            near = 0 if np.nanmedian(v[0]) > np.nanmedian(v[1]) else 1      # lower in the picture = nearer
    sides = [0.0, W] if near < 0 else [0.0 if near == 0 else W]
    zone = np.zeros(px.shape, bool)
    io = np.zeros(T)
    present = np.isfinite(px)
    change = present ^ np.r_[present[:1], present[:-1]]
    xprev = np.where(present, px, np.r_[px[:1], px[:-1]])
    yprev = np.where(present, py, np.r_[py[:1], py[:-1]])
    for yl in sides:
        inside = (np.abs(px - L / 2) < 8.0) & (((py - yl) * (1 if yl == 0 else -1)) < 3.0) & \
                 (((py - yl) * (1 if yl == 0 else -1)) > -5.0)
        zone |= np.nan_to_num(inside).astype(bool)
        near_pt = np.hypot(xprev - L / 2, yprev - yl) < 10.0
        io = np.maximum(io, (change & np.nan_to_num(near_pt).astype(bool)).any(1).astype(float))
    return {"near_side": np.full(T, float(near)), "n_sub_zone": zone.sum(1).astype(float), "sub_in_out": io}


def situation_features(px, py, team_mat, attack, tags, pitch, fps=25, bx=None, by=None, H=None, valid=None):
    f = player_features(px, py, team_mat, attack, pitch, fps)
    f.update(tag_features(tags, len(px), fps))
    if bx is not None:
        f.update(ball_features(np.asarray(bx, float), np.asarray(by, float), pitch, fps))
        f.update(ball_player_features(px, py, team_mat, attack, np.asarray(bx, float), np.asarray(by, float), pitch, fps))
    f.update(goal_kick_features(px, py, team_mat, attack, None if bx is None else np.asarray(bx, float),
                                None if by is None else np.asarray(by, float), H, valid, pitch, fps))
    f.update(substitution_features(px, py, pitch, H, valid, fps))
    # what is in the picture: the centre spot (kick-off) and the corner flags (corner kick); -1 = unknown
    L, W = pitch
    if H is None:
        f["centre_visible"] = np.full(len(px), -1.0)
        f["corner_visible"] = np.full(len(px), -1.0)
    else:
        f["centre_visible"] = area_visible(H, valid, pitch, L / 2 - 1, L / 2 + 1, W / 2 - 1, W / 2 + 1)
        cv = [area_visible(H, valid, pitch, x0, x0 + 1, y0, y0 + 1)
              for x0, y0 in ((0, 0), (0, W - 1), (L - 1, 0), (L - 1, W - 1))]
        f["corner_visible"] = np.where(cv[0] < 0, -1.0, np.max(cv, axis=0))
    return f
