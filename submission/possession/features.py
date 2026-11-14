"""Feature tables for the three evidence modules.

dead_features(...)   -> DataFrame, one row per frame             (module 1, alive vs dead)
team_features(...)   -> (X_home_view, X_away_view) DataFrames     (modules 2 and 3)

Team features are written from the point of view of one team ("own") against the other
("opp"). Building both views lets the model learn one symmetric rule ("does OWN have the ball?")
and doubles the training data; it also removes any home/away bias.
x coordinates are normalised to the attacking direction of the team in question, so "forward"
means the same thing in both halves.
"""
import numpy as np
import pandas as pd

from .preprocess import rolling, shift


# --------------------------------------------------------------------------- helpers
def _team_mask(team, k):
    return team == k


def _nearest_two(dist, mask):
    """Per frame: (d1, d2, index of nearest) among slots in mask. inf/-1 when absent."""
    d = np.where(mask[None, :], dist, np.inf)
    d = np.where(np.isnan(d), np.inf, d)
    order = np.argsort(d, axis=1)[:, :2]
    rows = np.arange(d.shape[0])
    d1 = d[rows, order[:, 0]]
    d2 = d[rows, order[:, 1]]
    idx = np.where(np.isfinite(d1), order[:, 0], -1)
    return np.where(np.isfinite(d1), d1, np.nan), np.where(np.isfinite(d2), d2, np.nan), idx


def _add_windows(df, cols, P):
    """Past / future rolling means for each column and window (the 'context' features)."""
    extra = {}
    for c in cols:
        x = df[c].values
        for w_s in P.windows_s:
            w = P.frames(w_s)
            past = rolling(x, w, "past")
            fut = rolling(x, w, "future")
            extra[f"{c}_p{w_s}"] = past
            extra[f"{c}_f{w_s}"] = fut
            extra[f"{c}_d{w_s}"] = fut - past
    return pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)


def _since_run_start(mask):
    """Frames since the current True run started (0 when False)."""
    out = np.zeros(len(mask))
    c = 0
    for i, m in enumerate(mask):
        c = c + 1 if m else 0
        out[i] = c
    return out


# --------------------------------------------------------------------------- module 1
def dead_features(prep, tags, P):
    """Alive/dead evidence. No team perspective."""
    T = prep["n_frames"]
    L, W = P.pitch_length, P.pitch_width
    sp = prep["pspeed"]
    tracked = ~np.isnan(prep["px"])
    team = prep["team"]
    f = pd.DataFrame(index=np.arange(T))
    with np.errstate(invalid="ignore"):
        f["speed_mean"] = np.nanmean(sp, axis=1)
        f["speed_std"] = np.nanstd(sp, axis=1)
        f["speed_mean_home"] = np.nanmean(np.where(team == 0, sp, np.nan), axis=1)
        f["speed_mean_away"] = np.nanmean(np.where(team == 1, sp, np.nan), axis=1)
        f["frac_walking"] = np.nansum(sp < P.dead_walk_speed, axis=1) / np.maximum(tracked.sum(1), 1)
        f["n_sprint"] = np.nansum(sp > P.sprint_speed, axis=1)
    f["n_tracked"] = tracked.sum(axis=1)

    # ball: where it is / was, and whether it has left the pitch
    lx, ly = prep["last_bx"], prep["last_by"]
    f["ball_seen"] = prep["ball_seen"].astype(float)
    f["ball_speed"] = prep["bspeed"]
    f["since_seen"] = prep["since_seen"]
    f["until_seen"] = prep["until_seen"]
    out_dist = np.nanmax(np.stack([-lx, lx - L, -ly, ly - W]), axis=0)
    f["last_ball_out_dist"] = out_dist                       # >0 means outside the pitch
    f["last_ball_out"] = (out_dist > P.out_of_pitch_margin_m).astype(float)
    f["last_ball_line_dist"] = np.nanmin(np.stack([lx, L - lx, ly, W - ly]), axis=0)

    # camera
    f["cam_speed"] = prep["cam_speed"]
    f["cam_off_centre"] = np.hypot(prep["aim_x"] - L / 2, prep["aim_y"] - W / 2)

    # formation: kick-off shape = everybody in their own half
    xn = (prep["px"] - L / 2) * prep["attack"][:, team]          # >0 = in the opponent half
    with np.errstate(invalid="ignore"):
        f["frac_own_half"] = np.nansum(xn < 0, axis=1) / np.maximum(tracked.sum(1), 1)
        cx = [np.nanmean(np.where(team == k, prep["px"], np.nan), axis=1) for k in (0, 1)]
    f["centroid_gap"] = np.abs(cx[0] - cx[1])

    # Smart Tagging: distance in time to the next tagged set piece (a restart)
    nxt = np.full(T, np.inf)
    inside = np.zeros(T)
    for e in tags or []:
        if e["code"] == "SET PIECES":
            s = int(np.clip(e["start"], 0, T - 1))
            nxt[:s + 1] = np.minimum(nxt[:s + 1], (s - np.arange(s + 1)) / P.fps)
            inside[max(0, e["start"]):min(T, e["end"])] = 1
    f["to_set_piece_s"] = np.where(np.isfinite(nxt), np.minimum(nxt, 120.0), 120.0)
    f["in_set_piece_tag"] = inside

    return _add_windows(f, ["speed_mean", "frac_walking", "cam_speed", "ball_seen", "n_sprint"], P)


# --------------------------------------------------------------------------- modules 2 and 3
def _per_team(prep, P):
    """Raw per-team quantities, computed once. Returns dict[k] -> dict of arrays."""
    T = prep["n_frames"]
    L, W = P.pitch_length, P.pitch_width
    px, py, vx, vy = prep["px"], prep["py"], prep["pvx"], prep["pvy"]
    team = prep["team"]
    bx, by = prep["bx"], prep["by"]
    rx, ry = prep["region_x"], prep["region_y"]
    rows = np.arange(T)
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    rdist = np.hypot(px - rx[:, None], py - ry[:, None])
    # time to reach the ball: keep running for the reaction time, then straight at max speed
    rx_ = px + np.nan_to_num(vx) * P.reaction_time_s
    ry_ = py + np.nan_to_num(vy) * P.reaction_time_s
    ttb = P.reaction_time_s + np.hypot(rx_ - bx[:, None], ry_ - by[:, None]) / P.player_max_speed
    ttb = np.where(np.isnan(ttb), np.inf, ttb)

    out = {}
    for k in (0, 1):
        m = _team_mask(team, k)
        o = ~m
        sign = prep["attack"][:, k]
        xn = (px - L / 2) * sign[:, None]          # forward = +, for this team
        vxn = vx * sign[:, None]
        d1, d2, i1 = _nearest_two(dist, m)
        rel = np.full(T, np.nan)
        ok = i1 >= 0
        rel[ok] = np.hypot(vx[rows[ok], i1[ok]] - prep["bvx"][ok], vy[rows[ok], i1[ok]] - prep["bvy"][ok])
        t_min = np.where(m[None, :], ttb, np.inf).min(axis=1)
        mx = np.where(m[None, :], xn, np.nan)
        my = np.where(m[None, :], py, np.nan)
        msp = np.where(m[None, :], prep["pspeed"], np.nan)
        mvx = np.where(m[None, :], vxn, np.nan)
        # nearest opponent distance for each of this team's players
        oppx = np.where(o[None, :], px, np.nan)
        oppy = np.where(o[None, :], py, np.nan)
        nn = np.full((T, int(m.sum())), np.nan)
        for j, s in enumerate(np.flatnonzero(m)):
            dd = np.hypot(oppx - px[:, [s]], oppy - py[:, [s]])
            nn[:, j] = np.nanmin(np.where(np.isnan(dd), np.inf, dd), axis=1)
        nn[~np.isfinite(nn)] = np.nan
        # towards the ball region
        dxr = rx[:, None] - px
        dyr = ry[:, None] - py
        norm = np.hypot(dxr, dyr)
        with np.errstate(invalid="ignore", divide="ignore"):
            approach = (vx * dxr + vy * dyr) / norm            # m/s towards the region
            sorted_x = np.sort(np.where(np.isnan(mx), np.inf, mx), axis=1)
            back4 = sorted_x[:, :4]
            back4[~np.isfinite(back4)] = np.nan
            out[k] = dict(
                d1=d1, d2=d2, rel_speed=rel, ttb=np.where(np.isfinite(t_min), t_min, np.nan),
                n_near=np.nansum(np.where(m[None, :], dist, np.nan) < P.near_radius_m, axis=1),
                n_tracked=np.sum(~np.isnan(np.where(m[None, :], px, np.nan)), axis=1),
                cx=np.nanmean(mx, axis=1), cvx=np.nanmean(mvx, axis=1),
                width=np.nanmax(my, axis=1) - np.nanmin(my, axis=1),
                length=np.nanmax(mx, axis=1) - np.nanmin(mx, axis=1),
                spread=np.nanstd(mx, axis=1) * np.nanstd(my, axis=1),
                back_line=np.nanmean(back4, axis=1),
                n_sprint_fwd=np.nansum((msp > P.sprint_speed) & (mvx > 0), axis=1),
                n_sprint_back=np.nansum((msp > P.sprint_speed) & (mvx < 0), axis=1),
                nn_opp=np.nanmean(nn, axis=1),
                n_region=np.nansum(np.where(m[None, :], rdist, np.nan) < P.region_radius_m, axis=1),
                n_approach=np.nansum(np.where(m[None, :], approach, np.nan) > 1.0, axis=1),
                region_min=np.nanmin(np.where(m[None, :], rdist, np.inf), axis=1),
            )
        out[k]["region_min"][~np.isfinite(out[k]["region_min"])] = np.nan
        out[k]["nn_opp_std"] = np.sqrt(np.maximum(
            rolling(out[k]["nn_opp"] ** 2, P.fps, "centre") - rolling(out[k]["nn_opp"], P.fps, "centre") ** 2, 0))
    return out


BALL_COLS = ["d1", "d2", "rel_speed", "ttb", "n_near"]
SHAPE_COLS = ["cx", "cvx", "width", "length", "spread", "back_line", "n_sprint_fwd",
              "n_sprint_back", "nn_opp", "nn_opp_std", "n_region", "n_approach", "region_min",
              "n_tracked"]


def team_features(prep, P):
    """Two DataFrames with identical columns: the home view and the away view.

    Column groups: 'b_*' ball-based (module 2), 's_*' shape-based (module 3), 'c_*' context.
    """
    per = _per_team(prep, P)
    views = []
    for own, opp in ((0, 1), (1, 0)):
        f = pd.DataFrame(index=np.arange(prep["n_frames"]))
        for c in BALL_COLS:
            f[f"b_own_{c}"] = per[own][c]
            f[f"b_opp_{c}"] = per[opp][c]
            f[f"b_diff_{c}"] = per[own][c] - per[opp][c]
        for c in SHAPE_COLS:
            f[f"s_own_{c}"] = per[own][c]
            f[f"s_opp_{c}"] = per[opp][c]
            f[f"s_diff_{c}"] = per[own][c] - per[opp][c]
        f["c_ball_speed"] = prep["bspeed"]
        f["c_ball_ok"] = prep["ball_ok"].astype(float)
        f["c_since_seen"] = prep["since_seen"]
        f["c_until_seen"] = prep["until_seen"]
        f = _add_windows(f, ["b_diff_d1", "b_diff_ttb", "s_diff_n_region", "s_diff_cx",
                             "s_diff_n_approach", "s_own_cvx"], P)
        views.append(f)
    return views[0], views[1]


def ball_columns(df):
    return [c for c in df.columns if c.startswith(("b_", "c_"))]


def shape_columns(df):
    return [c for c in df.columns if c.startswith(("s_", "c_"))]
