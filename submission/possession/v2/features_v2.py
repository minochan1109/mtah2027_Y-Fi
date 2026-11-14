"""[D] Features of version 2. Every frame gets the BASE features; ball features are ADDED on top
and left blank (NaN) where the ball is not trusted, so the model falls back to the base.

Column groups (own team vs opponent, both views like version 1):
    s_*  shape          centroid, width, length, back line, sprints, distances to opponents ...
    k_*  containment    how many opponents are inside our team's box (attackers spread around
                        the defenders) and the size of the boxes
    r_*  recovery       players running fast towards their OWN goal, how many just started,
                        change of the team's forward speed  (a team that just lost the ball runs back)
    b_*  ball add-on    nearest player distances, time to ball ... only where conf >= ball_conf_min
    c_*  context        ball confidence, flight flag, time since a trusted ball

Team membership is PER FRAME (team_mat from clean_players), not per slot.
"""
import numpy as np
import pandas as pd

from ..features import _add_windows
from ..preprocess import rolling, shift, velocity


def _nearest_two(dist, mask):
    d = np.where(mask & np.isfinite(dist), dist, np.inf)
    order = np.argsort(d, axis=1)[:, :2]
    rows = np.arange(d.shape[0])
    d1, d2 = d[rows, order[:, 0]], d[rows, order[:, 1]]
    idx = np.where(np.isfinite(d1), order[:, 0], -1)
    return np.where(np.isfinite(d1), d1, np.nan), np.where(np.isfinite(d2), d2, np.nan), idx


def _per_team(prep, team_mat, bq, P):
    T = prep["n_frames"]
    L, W = P.pitch_length, P.pitch_width
    px, py, vx, vy = prep["px"], prep["py"], prep["pvx"], prep["pvy"]
    sp = prep["pspeed"]
    rows = np.arange(T)
    use_ball = bq["conf"] >= P.ball_conf_min
    bx = np.where(use_ball, bq["bx"], np.nan)
    by = np.where(use_ball, bq["by"], np.nan)
    hw = P.frames(P.velocity_half_window_s)
    bvx, bvy = velocity(bx, hw, P.fps), velocity(by, hw, P.fps)
    rx, ry = prep["region_x"], prep["region_y"]
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    rdist = np.hypot(px - rx[:, None], py - ry[:, None])
    rx_ = px + np.nan_to_num(vx) * P.reaction_time_s
    ry_ = py + np.nan_to_num(vy) * P.reaction_time_s
    ttb = P.reaction_time_s + np.hypot(rx_ - bx[:, None], ry_ - by[:, None]) / P.player_max_speed
    ttb = np.where(np.isfinite(ttb), ttb, np.inf)
    onset_w = P.frames(P.recover_onset_s)

    out = {}
    for k in (0, 1):
        m = team_mat == k
        o = team_mat == 1 - k
        sign = prep["attack"][:, k][:, None]               # +1: this team attacks +x
        xn = (px - L / 2) * sign                             # forward = +
        mx, my = np.where(m, xn, np.nan), np.where(m, py, np.nan)
        ox, oy = np.where(o, xn, np.nan), np.where(o, py, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = {}
            # ---- base: shape
            f["s_n_tracked"] = m.sum(axis=1).astype(float)
            f["s_cx"] = np.nanmean(mx, axis=1)
            f["s_cvx"] = np.nanmean(np.where(m, vx * sign, np.nan), axis=1)
            f["s_width"] = np.nanmax(my, axis=1) - np.nanmin(my, axis=1)
            f["s_length"] = np.nanmax(mx, axis=1) - np.nanmin(mx, axis=1)
            f["s_spread"] = np.nanstd(mx, axis=1) * np.nanstd(my, axis=1)
            back = np.sort(np.where(np.isnan(mx), np.inf, mx), axis=1)[:, :4]
            back[~np.isfinite(back)] = np.nan
            f["s_back_line"] = np.nanmean(back, axis=1)
            msp = np.where(m, sp, np.nan)
            mvx = np.where(m, vx * sign, np.nan)
            f["s_n_sprint_fwd"] = np.nansum((msp > P.sprint_speed) & (mvx > 0), axis=1)
            f["s_n_sprint_back"] = np.nansum((msp > P.sprint_speed) & (mvx < 0), axis=1)
            nn = np.full((T, px.shape[1]), np.nan)
            for s in range(px.shape[1]):
                if not m[:, s].any():
                    continue
                dd = np.hypot(np.where(o, px, np.nan) - px[:, [s]], np.where(o, py, np.nan) - py[:, [s]])
                v = np.nanmin(np.where(np.isfinite(dd), dd, np.inf), axis=1)
                nn[:, s] = np.where(m[:, s] & np.isfinite(v), v, np.nan)
            f["s_nn_opp"] = np.nanmean(nn, axis=1)
            f["s_n_region"] = np.nansum(np.where(m, rdist, np.nan) < P.region_radius_m, axis=1)
            dxr, dyr = rx[:, None] - px, ry[:, None] - py
            appr = (vx * dxr + vy * dyr) / np.hypot(dxr, dyr)
            f["s_n_approach"] = np.nansum(np.where(m, appr, np.nan) > 1.0, axis=1)
            rmin = np.nanmin(np.where(m, rdist, np.inf), axis=1)
            f["s_region_min"] = np.where(np.isfinite(rmin), rmin, np.nan)

            # ---- containment: opponents inside our box (attackers surround defenders)
            q = P.containment_pct
            x_lo, x_hi = np.nanpercentile(mx, q, axis=1), np.nanpercentile(mx, 100 - q, axis=1)
            y_lo, y_hi = np.nanpercentile(my, q, axis=1), np.nanpercentile(my, 100 - q, axis=1)
            inside = (ox >= x_lo[:, None]) & (ox <= x_hi[:, None]) & (oy >= y_lo[:, None]) & (oy <= y_hi[:, None])
            n_opp = np.isfinite(ox).sum(axis=1)
            f["k_frac_opp_inside"] = np.where(n_opp > 0, inside.sum(axis=1) / np.maximum(n_opp, 1), np.nan)
            f["k_box_area"] = (x_hi - x_lo) * (y_hi - y_lo)
            f["k_box_length"] = x_hi - x_lo
            f["k_box_width"] = y_hi - y_lo

            # ---- recovery: running towards OWN goal (own goal is at x=0 if the team attacks +x)
            gx = np.where(sign > 0, 0.0, L)
            dgx, dgy = gx - px, W / 2 - py
            vgoal = (vx * dgx + vy * dgy) / np.hypot(dgx, dgy)
            n_rec = np.nansum(np.where(m, vgoal, np.nan) > P.recover_speed, axis=1).astype(float)
            f["r_n_recover"] = n_rec
            f["r_recover_onset"] = np.maximum(0.0, n_rec - np.nan_to_num(shift(n_rec, onset_w), nan=0.0))
            f["r_cvx_change"] = f["s_cvx"] - shift(f["s_cvx"], P.fps)

            # ---- ball add-on (blank where the ball is not trusted)
            d1, d2, i1 = _nearest_two(dist, m)
            rel = np.full(T, np.nan)
            ok = i1 >= 0
            rel[ok] = np.hypot(vx[rows[ok], i1[ok]] - bvx[ok], vy[rows[ok], i1[ok]] - bvy[ok])
            tmin = np.where(m, ttb, np.inf).min(axis=1)
            f["b_d1"], f["b_d2"], f["b_rel_speed"] = d1, d2, rel
            f["b_ttb"] = np.where(np.isfinite(tmin), tmin, np.nan)
            f["b_n_near"] = np.where(use_ball, np.nansum(np.where(m, dist, np.nan) < P.near_radius_m, axis=1), np.nan)
        out[k] = f
    return out


def team_features_v2(prep, team_mat, bq, P):
    per = _per_team(prep, team_mat, bq, P)
    T = prep["n_frames"]
    since_trusted = prep["since_seen"]
    views = []
    for own, opp in ((0, 1), (1, 0)):
        cols = {}
        for name in per[own]:
            grp, rest = name.split("_", 1)
            cols[f"{grp}_own_{rest}"] = per[own][name]
            cols[f"{grp}_opp_{rest}"] = per[opp][name]
            cols[f"{grp}_diff_{rest}"] = per[own][name] - per[opp][name]
        cols["c_ball_conf"] = bq["conf"]
        cols["c_flight"] = bq["flight"].astype(float)
        cols["c_since_trusted"] = since_trusted
        f = pd.DataFrame(cols, index=np.arange(T))
        f = _add_windows(f, ["s_diff_n_region", "s_diff_cx", "s_diff_n_approach", "s_own_cvx",
                             "k_diff_frac_opp_inside", "r_diff_n_recover", "r_diff_recover_onset",
                             "b_diff_d1", "b_diff_ttb"], P)
        views.append(f)
    return views[0], views[1]


def base_columns(df):
    return [c for c in df.columns if c.startswith(("s_", "k_", "r_"))]


def ball_columns(df):
    return [c for c in df.columns if c.startswith(("b_", "c_"))]


def all_columns(df):
    return base_columns(df) + ball_columns(df)


def dead_extra(bq, tag_dead, P):
    """Extra columns for the dead-ball model."""
    f = pd.DataFrame({"ball_conf": bq["conf"], "ball_flight": bq["flight"].astype(float)})
    for k, v in tag_dead.items():
        f[k] = v
    f["ball_conf_p1"] = rolling(bq["conf"], P.fps, "past")
    f["ball_conf_f1"] = rolling(bq["conf"], P.fps, "future")
    return f
