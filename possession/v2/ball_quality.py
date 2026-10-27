"""[B] Judge every ball detection BEFORE using or interpolating it.

Output per frame
    bx, by   trusted / interpolated position (NaN otherwise)
    conf     0..1 how much later modules should trust it (0 = do not use)
    status   0 missing, 1 trusted detection, 2 interpolated, 3 rejected detection
    flight   True where the ball is probably in the air (its 2-D position is then unreliable)

Steps (all generic rules):
  1. far outside the pitch                          -> rejected
  2. link detections into tracklets (a ball cannot move faster than ball_max_speed)
  3. reject tracklets that are spikes (1-2 frames), short and far from every player,
     or static for seconds with no player near while the players run (spot, line, spare ball)
  4. interpolate only between trusted tracklets, only short gaps at a plausible speed
  5. flight: fast for a while with no player near -> lower the confidence
"""
import numpy as np

from ..preprocess import run_lengths, velocity

MISSING, TRUSTED, INTERP, REJECTED = 0, 1, 2, 3
REASONS = ("outside", "spike", "isolated", "static")


def _nearest_player(bx, by, px, py):
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    m = d.min(axis=1)
    return np.where(np.isfinite(m), m, np.nan)


def _tracklets(bx, by, P):
    """Label each detection with a tracklet id (-1 where no detection)."""
    T = len(bx)
    tid = np.full(T, -1, np.int64)
    det = np.flatnonzero(np.isfinite(bx))
    gap = P.frames(P.ball_link_gap_s)
    cur, last = -1, -10 ** 9
    for t in det:
        dt = t - last
        if cur >= 0 and dt <= gap:
            dist = np.hypot(bx[t] - bx[last], by[t] - by[last])
            if dist <= P.ball_max_speed * dt / P.fps + P.ball_link_margin_m:
                tid[t] = cur
                last = t
                continue
        cur += 1
        tid[t] = cur
        last = t
    return tid


def assess_ball(bx_raw, by_raw, px, py, P):
    T = len(bx_raw)
    L, W = P.pitch_length, P.pitch_width
    bx = bx_raw.astype(np.float64).copy()
    by = by_raw.astype(np.float64).copy()
    status = np.where(np.isfinite(bx), TRUSTED, MISSING).astype(np.int8)
    reason = {r: 0 for r in REASONS}

    # 1. far outside
    outside = np.isfinite(bx) & ((bx < -P.ball_outside_reject_m) | (bx > L + P.ball_outside_reject_m)
                                 | (by < -P.ball_outside_reject_m) | (by > W + P.ball_outside_reject_m))
    reason["outside"] = int(outside.sum())
    status[outside] = REJECTED
    bx[outside] = by[outside] = np.nan

    # 2-3. tracklets
    tid = _tracklets(bx, by, P)
    near = _nearest_player(bx, by, px, py)
    with np.errstate(invalid="ignore"):
        live = np.nanmean(np.hypot(*(velocity(a, P.frames(P.velocity_half_window_s), P.fps)
                                     for a in (px, py))), axis=1)
    n_tr = tid.max() + 1
    order = np.argsort(tid, kind="stable")
    tsorted = tid[order]
    starts = np.searchsorted(tsorted, np.arange(n_tr))
    ends = np.searchsorted(tsorted, np.arange(n_tr), side="right")
    tr_len = np.zeros(max(n_tr, 1))
    for k in range(n_tr):
        fr = order[starts[k]:ends[k]]
        n = len(fr)
        dur = (fr[-1] - fr[0] + 1) / P.fps
        tr_len[k] = dur
        why = None
        if n <= P.ball_spike_frames:
            why = "spike"
        elif dur < P.ball_short_s and np.nanmin(np.r_[near[fr], np.inf]) > P.ball_isolated_m:
            why = "isolated"
        elif dur >= P.ball_static_s:
            spread = np.hypot(np.nanstd(bx[fr]), np.nanstd(by[fr]))
            if (spread < P.ball_static_spread_m and np.nanmedian(near[fr]) > P.ball_static_no_player_m
                    and np.nanmean(live[fr]) > P.ball_static_live_speed):
                why = "static"
        if why:
            reason[why] += n
            status[fr] = REJECTED
            bx[fr] = by[fr] = np.nan

    # confidence of trusted detections grows with the length of their tracklet
    conf = np.zeros(T)
    ok = status == TRUSTED
    conf[ok] = 0.6 + 0.4 * np.minimum(1.0, tr_len[tid[ok]] / 1.0)

    # 4. interpolate between trusted detections only
    maxg = P.frames(P.ball_interp_max_s)
    for a, b in run_lengths(~np.isfinite(bx)):
        if a == 0 or b >= T or (b - a) > maxg:
            continue
        dist = np.hypot(bx[b] - bx[a - 1], by[b] - by[a - 1])
        if dist / ((b - a + 1) / P.fps) > P.ball_interp_max_speed:
            continue
        w = np.arange(1, b - a + 1) / (b - a + 1)
        bx[a:b] = bx[a - 1] + (bx[b] - bx[a - 1]) * w
        by[a:b] = by[a - 1] + (by[b] - by[a - 1]) * w
        status[a:b] = np.where(status[a:b] == REJECTED, REJECTED, INTERP)
        conf[a:b] = 0.1 + 0.5 * (1.0 - (b - a) / maxg)
    # a rejected frame inside an interpolated gap keeps the interpolated position
    status[(status == REJECTED) & np.isfinite(bx)] = INTERP

    # 5. flight
    hw = P.frames(P.velocity_half_window_s)
    bsp = np.hypot(velocity(bx, hw, P.fps), velocity(by, hw, P.fps))
    near = _nearest_player(bx, by, px, py)
    fast = (np.nan_to_num(bsp) > P.ball_flight_speed) & (np.nan_to_num(near, nan=99.0) > P.ball_flight_player_m)
    flight = np.zeros(T, bool)
    for a, b in run_lengths(fast):
        if b - a >= P.frames(P.ball_flight_min_s):
            flight[a:b] = True
    conf = np.where(flight, conf * P.ball_flight_conf, conf)
    conf[~np.isfinite(bx)] = 0.0

    rep = {"detected": int(np.isfinite(bx_raw).sum()), "trusted": int((status == TRUSTED).sum()),
           "interpolated": int((status == INTERP).sum()), "rejected": int((status == REJECTED).sum()),
           "rejected_by_reason": reason, "tracklets": int(n_tr), "flight_frames": int(flight.sum())}
    return dict(bx=bx, by=by, conf=conf, status=status, flight=flight), rep
