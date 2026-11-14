"""MT-26 high-priority clues (11/5) turned into features. Used by MT-11 (D) and MT-24 (H/A, HMM).

Each clue is switched on with a Params2 field (all off by default, so nothing changes unless set
in --params):

    clue_score     (MT-24) score and time from the tags (GOALS, kick-off after goal). Per team:
                   s_lead = own goals - opponent goals, and s_cx_norm = team centroid minus the team's
                   usual centroid in the same score state ("the normal position changes with the score")
    clue_backline  (MT-24) the back line: s_back_std = spread (forward direction) of the 4 deepest
                   outfield players (small = level line = probably defending), s_back_width = their
                   sideways spread
    clue_crowd     (MT-24) players crowd the ball right after a change: b_n_near8 = players within 8 m of
                   the trusted ball, b_n_conv = players within 15 m running towards it, b_conv_rise = rise
                   of b_n_conv over the last second
    clue_crowd_switch (MT-24, HMM) the switch signal of hmm_v2 is replaced by the crowding onset
                   (both teams converge on the ball at once)
    clue_align     (MT-11) play stopped -> players move independently of their opponent:
                   align_frac = share of moving players whose nearest moving opponent goes the same way,
                   align_drop = fall of align_frac against the last 3 s (both teams at once)

Only the players and (where trusted) the ball position are used; nothing is tied to frames of a game.
"""
import numpy as np
import pandas as pd

from ..preprocess import rolling, shift


# ---------------------------------------------------------------- clue 1: score state from the tags
def score_state(tags, side_of, n, fps):
    """lead_home (T,) = home goals - away goals so far, and the minute of the match (T,).
    A goal counts from its GOALS tag. The team comes from the tag (club name or Home / Away)."""
    goals = []
    for t in tags:
        if t["code"] != "GOALS":
            continue
        team = t["tags"].get("Team", "")
        side = side_of(team)
        if side is None:
            low = team.lower()
            side = 0 if low == "home" else 1 if low == "away" else None
        if side is not None:
            goals.append((t["start"], side))
    lead = np.zeros(n)
    for start, side in sorted(goals):
        lead[min(max(start, 0), n):] += 1 if side == 0 else -1
    minute = np.arange(n) / fps / 60.0
    return lead, minute, len(goals)


def _per_score_median(x, lead):
    out = np.full(len(x), np.nan)
    for v in np.unique(lead):
        m = lead == v
        med = np.nanmedian(x[m]) if np.isfinite(x[m]).any() else np.nan
        out[m] = x[m] - med
    return out


# ---------------------------------------------------------------- clues 1-3: per team columns
def team_clues(prep, team_mat, bq, P, tags=None, side_of=None):
    """Per-team columns {0: {...}, 1: {...}} for the switched-on clues (names follow features_v2)."""
    T = prep["n_frames"]
    L = P.pitch_length
    px, py, vx, vy = prep["px"], prep["py"], prep["pvx"], prep["pvy"]
    use_ball = bq["conf"] >= P.ball_conf_min
    bx = np.where(use_ball, bq["bx"], np.nan)
    by = np.where(use_ball, bq["by"], np.nan)
    out = {0: {}, 1: {}}
    lead = None
    if getattr(P, "clue_score", False) and tags is not None:
        lead, _, _ = score_state(tags, side_of, T, P.fps)
    for k in (0, 1):
        m = team_mat == k
        sign = prep["attack"][:, k][:, None]
        xn = (px - L / 2) * sign
        mx = np.where(m, xn, np.nan)
        my = np.where(m, py, np.nan)
        f = out[k]
        with np.errstate(invalid="ignore", divide="ignore"):
            if lead is not None:
                own_lead = lead if k == 0 else -lead
                f["s_lead"] = own_lead
                cx = np.nanmean(mx, axis=1)
                f["s_cx_norm"] = _per_score_median(cx, own_lead)
                bl = np.sort(np.where(np.isnan(mx), np.inf, mx), axis=1)[:, 1:5]
                bl[~np.isfinite(bl)] = np.nan
                f["s_back_norm"] = _per_score_median(np.nanmean(bl, axis=1), own_lead)
            if getattr(P, "clue_backline", False):
                order = np.argsort(np.where(np.isnan(mx), np.inf, mx), axis=1)[:, 1:5]   # skip the deepest (keeper)
                rows = np.arange(T)[:, None]
                bxs, bys = mx[rows, order], my[rows, order]
                f["s_back_std"] = np.nanstd(bxs, axis=1)
                f["s_back_width"] = np.nanmax(bys, axis=1) - np.nanmin(bys, axis=1)
                f["s_back_std_p2"] = rolling(f["s_back_std"], P.frames(2.0), "past")
            if getattr(P, "clue_crowd", False):
                dx, dy = bx[:, None] - px, by[:, None] - py
                d = np.hypot(dx, dy)
                towards = (vx * dx + vy * dy) / np.where(d > 0.1, d, np.nan)
                f["b_n_near8"] = np.where(use_ball, np.nansum(np.where(m, d, np.nan) < 8.0, axis=1), np.nan)
                conv = np.where(use_ball, np.nansum(m & (d < 15.0) & (towards > 1.5), axis=1), np.nan).astype(float)
                f["b_n_conv"] = conv
                f["b_conv_rise"] = conv - shift(conv, P.fps)
    return out


def add_team_clues(Xh, Xa, per):
    """own / opp / diff columns like features_v2.team_features_v2, appended to both views."""
    views = []
    for X, (own, opp) in ((Xh, (0, 1)), (Xa, (1, 0))):
        cols = {}
        for name in per[own]:
            grp, rest = name.split("_", 1)
            cols[f"{grp}_own_{rest}"] = per[own][name]
            cols[f"{grp}_opp_{rest}"] = per[opp][name]
            cols[f"{grp}_diff_{rest}"] = per[own][name] - per[opp][name]
        views.append(pd.concat([X, pd.DataFrame(cols, index=X.index)], axis=1) if cols else X)
    return views[0], views[1]


def crowd_switch_signal(prep, team_mat, bq, P, mid=2.0, slope=1.5):
    """0..1 per frame: players of BOTH teams start converging on the trusted ball at once
    (what happens right after the ball changes hands)."""
    per = team_clues(prep, team_mat, bq, type("Q", (), {**{k: getattr(P, k) for k in dir(P) if not k.startswith("__")},
                                                          "clue_score": False, "clue_backline": False,
                                                          "clue_crowd": True})(), None, None)
    rise = np.minimum(np.nan_to_num(per[0]["b_conv_rise"]), np.nan_to_num(per[1]["b_conv_rise"])) \
        + 0.5 * np.maximum(np.nan_to_num(per[0]["b_conv_rise"]), np.nan_to_num(per[1]["b_conv_rise"]))
    rise = np.maximum.reduce([rise, shift(rise, 1, 0.0), shift(rise, -1, 0.0)])
    return 1.0 / (1.0 + np.exp(-slope * (rise - mid)))


# ---------------------------------------------------------------- clue 4: play stopped (D model)
def align_features(prep, team_mat, P, min_speed=1.0, cos_same=0.7):
    """align_frac (T,): among players moving faster than min_speed, the share whose nearest moving
    opponent moves the same way (cos >= cos_same). Per team, then the smaller of the two teams.
    align_drop: past-3 s mean minus now (positive = both teams suddenly stopped mirroring)."""
    px, py, vx, vy = prep["px"], prep["py"], prep["pvx"], prep["pvy"]
    sp = np.hypot(vx, vy)
    T, S = px.shape
    moving = np.isfinite(sp) & (sp > min_speed)
    fracs = []
    for k in (0, 1):
        m = (team_mat == k) & moving
        o = (team_mat == 1 - k) & moving
        n_al = np.zeros(T)
        n_mv = m.sum(axis=1).astype(float)
        ox = np.where(o, px, np.nan)
        oy = np.where(o, py, np.nan)
        with np.errstate(invalid="ignore"):
            for s in range(S):
                rows = np.flatnonzero(m[:, s])
                if len(rows) == 0:
                    continue
                dd = np.hypot(ox[rows] - px[rows, s][:, None], oy[rows] - py[rows, s][:, None])
                dd = np.where(np.isfinite(dd), dd, np.inf)
                j = dd.argmin(axis=1)
                ok = np.isfinite(dd[np.arange(len(rows)), j])
                r, jj = rows[ok], j[ok]
                cos = (vx[r, s] * vx[r, jj] + vy[r, s] * vy[r, jj]) / (sp[r, s] * sp[r, jj])
                n_al[r] += cos >= cos_same
        fracs.append(np.where(n_mv >= 2, n_al / np.maximum(n_mv, 1), np.nan))
    frac = np.fmin(fracs[0], fracs[1])
    frac_s = rolling(frac, P.frames(1.0), "centre")
    past = rolling(frac_s, P.frames(3.0), "past")
    drop = past - frac_s
    n_mv_all = moving.sum(axis=1).astype(float)
    return pd.DataFrame({"align_frac": frac_s, "align_drop": drop,
                         "align_frac_p2": rolling(frac_s, P.frames(2.0), "past"),
                         "align_frac_f2": rolling(frac_s, P.frames(2.0), "future"),
                         "n_moving": n_mv_all})
