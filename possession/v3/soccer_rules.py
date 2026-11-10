"""H / A corrections from plain football facts (MT-24, no fitting to any data).

Three facts are turned into strong evidence for one team, written over P(home) before the HMM:

  1. restart  after a stoppage the team that restarts has the ball:
                throw-in      the player standing at / over the side line
                corner kick   the player at a corner flag
                goal kick     the team that defends that goal (its keeper / defender kicks)
                kick-off      the player at the centre spot (the other team must be 9.15 m away)
                free kick, PK, others   the player nearest the ball's last position (if the ball is seen)
              -> that team for the first `restart_s` seconds of play
  2. holder   the ball (trusted position) is within holder_m of a player and no opponent is within
              holder_free_m, for holder_min_s -> that player's team
  3. keeper   a keeper holds the ball inside his own penalty box -> the keeper's team

Strength: `confidence` (default 0.95, i.e. "almost certain") for every corrected frame. The values are
football common sense, not tuned on the evaluation games.
"""
import numpy as np

from .expr import FPS, _runs

DEFAULTS = {"confidence": 0.95, "restart_s": 1.5, "restart_look_s": 1.5, "holder_m": 1.0, "holder_free_m": 2.0,
            "holder_min_s": 0.4, "keeper_m": 2.0, "ball_fresh_s": 0.2}


def _last_known(bx, by, max_age):
    T = len(bx)
    idx = np.maximum.accumulate(np.where(np.isfinite(bx), np.arange(T), -1))
    ok = (idx >= 0) & ((np.arange(T) - idx) <= max_age)
    return np.where(ok, bx[np.clip(idx, 0, None)], np.nan), np.where(ok, by[np.clip(idx, 0, None)], np.nan)


def _team_nearest(px, py, team_mat, frames, tx, ty):
    """Team of the player nearest the point (tx, ty) over the frames (majority), or -1."""
    votes = []
    for t in frames:
        d = np.hypot(px[t] - tx, py[t] - ty)
        d = np.where(np.isfinite(d) & (team_mat[t] >= 0), d, np.inf)
        j = int(np.argmin(d))
        if np.isfinite(d[j]) and d[j] < 5.0:
            votes.append(int(team_mat[t, j]))
    if not votes:
        return -1
    return int(np.bincount(votes).argmax())


def restart_teams(st, mask, stretches, sits, cols, opt):
    """[(play_start, team, reason)] for every play stretch that follows a D stretch."""
    px, py, tm = st["px"], st["py"], st["team_mat"]
    L, W = st["pitch"]
    T = st["n"]
    bx = np.asarray(st.get("bx", np.full(T, np.nan)), float)
    by = np.asarray(st.get("by", np.full(T, np.nan)), float)
    lx, ly = _last_known(bx, by, int(3 * FPS))
    att_hi = np.where(st["attack"][:, 0] > 0, 0, 1)          # team attacking the x = L goal (defends x = 0)
    out = []
    for _, a, b in st["blocks"]:
        for s, e in _runs(mask[a:b + 1]):
            s, e = s + a, e + a
            if e > b:
                continue
            # the situation of this stoppage: the stretch whose end is nearest
            near = min(stretches, key=lambda x: abs(x["end"] - e)) if stretches else None
            sit = sits[near["situation"]]["en"] if near is not None and abs(near["end"] - e) < 5 * FPS else "unknown"
            look = range(max(s, e - int(opt["restart_look_s"] * FPS)), e)
            team, why = -1, ""
            if sit == "throw-in":
                side = np.minimum(py[e - 1], W - py[e - 1])
                j = np.where(np.isfinite(side), side, np.inf)
                k = int(np.argmin(j))
                if np.isfinite(j[k]):
                    team = _team_nearest(px, py, tm, look, px[e - 1, k], py[e - 1, k])
                    why = "スローイン：サイドライン際の選手のチーム"
            elif sit == "corner kick":
                corners = [(0, 0), (0, W), (L, 0), (L, W)]
                best = None
                for cx, cy in corners:
                    d = np.nanmin(np.hypot(px[e - 1] - cx, py[e - 1] - cy))
                    if best is None or d < best[0]:
                        best = (d, cx, cy)
                if best and best[0] < 5:
                    team = _team_nearest(px, py, tm, look, best[1], best[2])
                    why = "CK：コーナーの選手のチーム"
            elif sit == "goal kick":
                ke = int(cols.get("kick_end", np.full(T, -1))[e - 1])
                if ke in (0, 1):
                    team = int(att_hi[e - 1]) if ke == 0 else int(1 - att_hi[e - 1])
                    why = "ゴールキック：そのゴールを守るチーム"
            elif sit == "kick-off":
                team = _team_nearest(px, py, tm, look, L / 2, W / 2)
                why = "キックオフ：センタースポットの選手のチーム"
            if team < 0 and np.isfinite(lx[e - 1]):
                team = _team_nearest(px, py, tm, look, lx[e - 1], ly[e - 1])
                why = why or "再開：ボールの最後の位置に一番近い選手のチーム"
            if team >= 0:
                out.append((e, team, why))
    return out


def holder_team(st, opt):
    """(T,) team (0 / 1) of the player holding the ball, -1 = nobody."""
    px, py, tm = st["px"], st["py"], st["team_mat"]
    T = st["n"]
    bx = np.asarray(st.get("bx", np.full(T, np.nan)), float)
    by = np.asarray(st.get("by", np.full(T, np.nan)), float)
    lx, ly = _last_known(bx, by, int(opt["ball_fresh_s"] * FPS))
    d = np.hypot(px - lx[:, None], py - ly[:, None])
    d = np.where(np.isfinite(d) & (tm >= 0), d, np.inf)
    out = np.full(T, -1, np.int8)
    r = np.arange(T)
    for k in (0, 1):
        own = np.where(tm == k, d, np.inf).min(1)
        opp = np.where(tm == 1 - k, d, np.inf).min(1)
        ok = (own <= opt["holder_m"]) & (opp > opt["holder_free_m"])
        for s, e in _runs(ok):
            if e - s >= int(opt["holder_min_s"] * FPS):
                out[s:e] = k
    del r
    return out


def keeper_team(st, cols, opt):
    """(T,) team of a keeper holding the ball in his own box, -1 = no."""
    T = st["n"]
    kb = np.asarray(cols.get("keeper_ball", np.zeros(T))) > 0
    out = np.full(T, -1, np.int8)
    if not kb.any():
        return out
    L, W = st["pitch"]
    px, py, tm = st["px"], st["py"], st["team_mat"]
    att_hi = np.where(st["attack"][:, 0] > 0, 0, 1)
    bx = np.asarray(st.get("bx", np.full(T, np.nan)), float)
    lx, _ = _last_known(bx, np.asarray(st.get("by", np.full(T, np.nan)), float), FPS)
    for t in np.flatnonzero(kb):
        if np.isfinite(lx[t]):
            out[t] = att_hi[t] if lx[t] < L / 2 else 1 - att_hi[t]   # the team defending that end
    return out


def apply(st, mask, stretches, sits, cols, opt=None):
    """New P(home) (T,) and a report dict. Order (later wins): holder, keeper, restart."""
    o = dict(DEFAULTS)
    o.update(opt or {})
    c = o["confidence"]
    ph = np.asarray(st["p_home"], float).copy()
    why = np.full(st["n"], "", dtype=object)
    play = ~mask
    h = holder_team(st, o)
    sel = play & (h >= 0)
    ph[sel] = np.where(h[sel] == 0, c, 1 - c)
    why[sel] = "ボールを持っている選手のチーム"
    k = keeper_team(st, cols, o)
    sel = play & (k >= 0)
    ph[sel] = np.where(k[sel] == 0, c, 1 - c)
    why[sel] = "キーパーがボールを持っている"
    rs = restart_teams(st, mask, stretches, sits, cols, o)
    n = int(o["restart_s"] * FPS)
    for e, team, reason in rs:
        sl = slice(e, min(st["n"], e + n))
        ph[sl] = c if team == 0 else 1 - c
        why[sl] = reason
    rep = {"holder_frames": int(((h >= 0) & play).sum()), "keeper_frames": int(((k >= 0) & play).sum()),
           "restarts": len(rs)}
    return ph, why, rep
