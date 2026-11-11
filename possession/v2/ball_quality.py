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
  4. interpolate only between trusted tracklets: follow the player when the same player has the
     ball at both ends of the gap; when a player has it at one end only, keep it at his feet and
     kick it (or receive it) at the speed of the free ball; a pass when different players have it
     at the two ends; otherwise a smooth curve keeping the ball's speed at both ends (or a straight
     line) over short
     gaps at a plausible speed
  5. flight: fast for a while with no player near -> lower the confidence
"""
import numpy as np

from ..preprocess import run_lengths, velocity

MISSING, TRUSTED, INTERP, REJECTED = 0, 1, 2, 3
REASONS = ("outside", "spike", "isolated", "static", "jump")


def _nearest_player(bx, by, px, py):
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    m = d.min(axis=1)
    return np.where(np.isfinite(m), m, np.nan)


def _carrier(px, py, t, x, y, radius):
    """Index of the player within `radius` of (x, y) at frame t, -1 if none."""
    d = np.hypot(px[t] - x, py[t] - y)
    d = np.where(np.isfinite(d), d, np.inf)
    k = int(np.argmin(d))
    return k if d[k] < radius else -1


def _kick(px, py, bx, by, a, b, ka, kb, P):
    """Fill gap a..b-1 when only one end has a carrier: the ball stays at his feet, then travels straight
    at the speed of the free ball seen at the other end (kicked away by ka, or received by kb)."""
    T = len(bx)
    K = P.frames(P.ball_kick_speed_s)
    k = ka if ka >= 0 else kb
    if not np.isfinite(px[a - 1:b + 1, k]).all():
        return None
    t = np.arange(a, b)
    if ka >= 0:
        if b + K >= T or not np.isfinite(bx[b:b + K + 1]).all():
            return None
        v = np.hypot(bx[b + K] - bx[b], by[b + K] - by[b]) * P.fps / K * P.ball_kick_speed_scale
        hx, hy = px[t, k] + bx[a - 1] - px[a - 1, k], py[t, k] + by[a - 1] - py[a - 1, k]
        fits = np.hypot(bx[b] - hx, by[b] - hy) <= v * (b - t) / P.fps
        t0 = t[fits].max() if fits.any() else a          # latest kick that still reaches the free ball
        x0, y0 = (hx[t0 - a - 1], hy[t0 - a - 1]) if t0 > a else (bx[a - 1], by[a - 1])
        w = (t - (t0 - 1)) / (b - (t0 - 1))
        x = np.where(t < t0, hx, x0 + (bx[b] - x0) * w)
        y = np.where(t < t0, hy, y0 + (by[b] - y0) * w)
        return x, y
    if a - 1 - K < 0 or not np.isfinite(bx[a - 1 - K:a]).all():
        return None
    v = np.hypot(bx[a - 1] - bx[a - 1 - K], by[a - 1] - by[a - 1 - K]) * P.fps / K * P.ball_kick_speed_scale
    hx, hy = px[t, k] + bx[b] - px[b, k], py[t, k] + by[b] - py[b, k]
    fits = np.hypot(hx - bx[a - 1], hy - by[a - 1]) <= v * (t - a + 1) / P.fps
    t1 = t[fits].min() if fits.any() else b              # earliest moment the receiver can have it
    x1, y1 = (hx[t1 - a], hy[t1 - a]) if t1 < b else (bx[b], by[b])
    w = (t - (a - 1)) / (t1 - (a - 1))
    x = np.where(t >= t1, hx, bx[a - 1] + (x1 - bx[a - 1]) * w)
    y = np.where(t >= t1, hy, by[a - 1] + (y1 - by[a - 1]) * w)
    return x, y


def _pass(px, py, bx, by, a, b, ka, kb, P):
    """Fill gap a..b-1 when different players have the ball at the two ends: at ka's feet, a straight
    pass at ball_pass_speed in the middle of the gap, then at kb's feet."""
    if not (np.isfinite(px[a - 1:b + 1, ka]).all() and np.isfinite(px[a - 1:b + 1, kb]).all()):
        return None
    t = np.arange(a, b)
    n = b - a
    ax, ay = px[t, ka] + bx[a - 1] - px[a - 1, ka], py[t, ka] + by[a - 1] - py[a - 1, ka]
    cx, cy = px[t, kb] + bx[b] - px[b, kb], py[t, kb] + by[b] - py[b, kb]
    m = n // 2
    f = min(n - 1, max(1, int(round(np.hypot(cx[m] - ax[m], cy[m] - ay[m]) / P.ball_pass_speed * P.fps))))
    i0 = max(0, m - f // 2)
    i1 = min(n - 1, i0 + f)
    i = np.arange(n)
    w = np.clip((i - i0) / (i1 - i0), 0, 1)
    x = np.where(i < i0, ax, np.where(i > i1, cx, ax[i0] + (cx[i1] - ax[i0]) * w))
    y = np.where(i < i0, ay, np.where(i > i1, cy, ay[i0] + (cy[i1] - ay[i0]) * w))
    return x, y


def _curve(bx, by, a, b, P):
    """Fill gap a..b-1 with a cubic (Hermite) curve that matches the ball's position and speed at both
    ends (speed over ball_kick_speed_s of trusted detections on each side); None if not measurable."""
    if not P.ball_use_curve:
        return None
    T = len(bx)
    K = P.frames(P.ball_kick_speed_s)
    if a - 1 - K < 0 or b + K >= T or not (np.isfinite(bx[a - 1 - K:a]).all() and np.isfinite(bx[b:b + K + 1]).all()):
        return None
    n = b - a + 1
    s = np.arange(1, n) / n
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    out = []
    for z in (bx, by):
        m0 = (z[a - 1] - z[a - 1 - K]) / K * n
        m1 = (z[b + K] - z[b]) / K * n
        out.append(h00 * z[a - 1] + h10 * m0 + h01 * z[b] + h11 * m1)
    return out[0], out[1]


def _average(fill, curve):
    """Mean of a kick / pass fill and the curve fill (the fill alone when there is no curve)."""
    if curve is None:
        return fill
    return (fill[0] + curve[0]) / 2, (fill[1] + curve[1]) / 2


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

    # 3b. a short tracklet the ball jumps to and back from, while the tracklets before and after it
    #     join up (a false ball inside the real track: a spot, a line, another player's feet) -> rejected
    if P.ball_jump_max_s > 0:
        first = {k: order[starts[k]] for k in range(n_tr)}
        last = {k: order[ends[k] - 1] for k in range(n_tr)}

        def frozen(k):
            fr = order[starts[k]:ends[k]]
            return len(fr) > 1 and np.hypot(np.nanstd(bx_raw[fr]), np.nanstd(by_raw[fr])) < P.ball_jump_frozen_m

        def jumps(k1, k2, min_m):
            t1, t2 = last[k1], first[k2]
            d = np.hypot(bx_raw[t2] - bx_raw[t1], by_raw[t2] - by_raw[t1])
            return d > max(min_m, P.ball_max_speed * (t2 - t1) / P.fps + P.ball_link_margin_m)

        kept = [k for k in range(n_tr) if status[first[k]] == TRUSTED]
        i = 1
        while i < len(kept) - 1:
            a_, b_, c_ = kept[i - 1], kept[i], kept[i + 1]
            # a ball that does not move at all only needs a short jump to be suspicious
            m = P.ball_jump_frozen_min_m if frozen(b_) else P.ball_jump_min_m
            if (tr_len[b_] <= P.ball_jump_max_s and jumps(a_, b_, m) and jumps(b_, c_, m)
                    and not jumps(a_, c_, P.ball_jump_min_m)):
                fr = order[starts[b_]:ends[b_]]
                reason["jump"] += len(fr)
                status[fr] = REJECTED
                bx[fr] = by[fr] = np.nan
                kept.pop(i)
                i = max(1, i - 1)
            else:
                i += 1

    # confidence of trusted detections grows with the length of their tracklet
    conf = np.zeros(T)
    ok = status == TRUSTED
    conf[ok] = P.ball_conf_floor + (1.0 - P.ball_conf_floor) * np.minimum(1.0, tr_len[tid[ok]] / P.ball_conf_full_s)

    # 4. interpolate between trusted detections only
    maxg = P.frames(P.ball_interp_max_s)
    maxf = P.frames(P.ball_follow_max_s)
    maxk = P.frames(P.ball_kick_max_s)
    followed = np.zeros(T, bool)
    kicked = np.zeros(T, bool)
    maxp = P.frames(P.ball_pass_max_s)
    passed = np.zeros(T, bool)
    maxc = max(maxg, P.frames(P.ball_curve_max_s))
    curved = np.zeros(T, bool)
    maxl = P.frames(P.ball_line_max_s) if P.ball_line_max_s > 0 else 0
    lined = np.zeros(T, bool)
    tbx, tby = bx.copy(), by.copy()                      # trusted detections only
    for a, b in run_lengths(~np.isfinite(bx)):
        if a == 0 or b >= T:
            continue
        ka = _carrier(px, py, a - 1, bx[a - 1], by[a - 1], P.ball_follow_m)
        kb = _carrier(px, py, b, bx[b], by[b], P.ball_follow_m)
        # 4a. the same player has the ball at both ends of the gap -> it carried the ball
        k = ka
        if (b - a) <= maxf and k >= 0 and k == kb and np.isfinite(px[a - 1:b + 1, k]).all():
            w = np.arange(1, b - a + 1) / (b - a + 1)
            ox = bx[a - 1] - px[a - 1, k] + (bx[b] - px[b, k] - bx[a - 1] + px[a - 1, k]) * w
            oy = by[a - 1] - py[a - 1, k] + (by[b] - py[b, k] - by[a - 1] + py[a - 1, k]) * w
            bx[a:b] = px[a:b, k] + ox
            by[a:b] = py[a:b, k] + oy
            status[a:b] = np.where(status[a:b] == REJECTED, REJECTED, INTERP)
            conf[a:b] = P.ball_follow_conf
            followed[a:b] = True
            continue
        # 4a2. otherwise, a gap that is not too long, or where the ball has hardly moved between its two
        #      ends: a straight line at constant speed (no guesses)
        still = (P.ball_line_still_m > 0 and (b - a) <= P.frames(P.ball_rest_max_s)
                 and np.hypot(bx[b] - bx[a - 1], by[b] - by[a - 1]) <= P.ball_line_still_m)
        if (b - a) <= maxl or still:
            w = np.arange(1, b - a + 1) / (b - a + 1)
            bx[a:b] = bx[a - 1] + (bx[b] - bx[a - 1]) * w
            by[a:b] = by[a - 1] + (by[b] - by[a - 1]) * w
            status[a:b] = np.where(status[a:b] == REJECTED, REJECTED, INTERP)
            conf[a:b] = (P.ball_interp_conf_min + (P.ball_interp_conf_max - P.ball_interp_conf_min)
                         * (1.0 - (b - a) / maxg)) if (b - a) <= maxg else P.ball_curve_conf
            lined[a:b] = True
            continue
        # 4b. a player has the ball at one end only, gap too long for a straight line -> kick / receive
        if maxg < (b - a) <= maxk and (ka >= 0) != (kb >= 0):
            r = _kick(px, py, tbx, tby, a, b, ka, kb, P)
            if r is not None:
                bx[a:b], by[a:b] = _average(r, _curve(tbx, tby, a, b, P))
                status[a:b] = np.where(status[a:b] == REJECTED, REJECTED, INTERP)
                conf[a:b] = P.ball_kick_conf
                kicked[a:b] = True
                continue
        # 4c. different players at the two ends, gap too long for a straight line -> a pass
        if maxg < (b - a) <= maxp and ka >= 0 and kb >= 0 and ka != kb:
            r = _pass(px, py, tbx, tby, a, b, ka, kb, P)
            if r is not None:
                bx[a:b], by[a:b] = _average(r, _curve(tbx, tby, a, b, P))
                status[a:b] = np.where(status[a:b] == REJECTED, REJECTED, INTERP)
                conf[a:b] = P.ball_pass_conf
                passed[a:b] = True
                continue
        # 4d. otherwise a smooth curve keeping the ball's speed at both ends (a straight line when the
        #     speed is not known): short gaps, or up to ball_curve_max_s when the curve is possible
        if (b - a) > maxc:
            continue
        dist = np.hypot(bx[b] - bx[a - 1], by[b] - by[a - 1])
        if dist / ((b - a + 1) / P.fps) > P.ball_interp_max_speed:
            continue
        r = _curve(tbx, tby, a, b, P)
        if (b - a) > maxg:
            if r is None:
                continue
            bx[a:b], by[a:b] = r
            status[a:b] = np.where(status[a:b] == REJECTED, REJECTED, INTERP)
            conf[a:b] = P.ball_curve_conf
            curved[a:b] = True
            continue
        if r is not None:
            bx[a:b], by[a:b] = r
        else:
            w = np.arange(1, b - a + 1) / (b - a + 1)
            bx[a:b] = bx[a - 1] + (bx[b] - bx[a - 1]) * w
            by[a:b] = by[a - 1] + (by[b] - by[a - 1]) * w
        status[a:b] = np.where(status[a:b] == REJECTED, REJECTED, INTERP)
        conf[a:b] = P.ball_interp_conf_min + (P.ball_interp_conf_max - P.ball_interp_conf_min) * (1.0 - (b - a) / maxg)
    # 4e. every gap still empty: the smooth curve (a straight line when the speed is not known) between
    #     its ends, holding the last / first position at the start and end of the data, with
    #     confidence ball_rest_conf
    rest = np.zeros(T, bool)
    if P.ball_rest_fill and np.isfinite(bx).any():
        for a, b in run_lengths(~np.isfinite(bx)):
            if P.ball_rest_max_s > 0 and (b - a) > P.frames(P.ball_rest_max_s):
                continue                                # long gap: the ball is probably out of play
            if a == 0:
                bx[:b], by[:b] = bx[b], by[b]
            elif b >= T:
                bx[a:], by[a:] = bx[a - 1], by[a - 1]
            else:
                r = _curve(tbx, tby, a, b, P)
                if r is not None:
                    bx[a:b], by[a:b] = r
                else:
                    w = np.arange(1, b - a + 1) / (b - a + 1)
                    bx[a:b] = bx[a - 1] + (bx[b] - bx[a - 1]) * w
                    by[a:b] = by[a - 1] + (by[b] - by[a - 1]) * w
            status[a:b] = np.where(status[a:b] == REJECTED, REJECTED, INTERP)
            conf[a:b] = P.ball_rest_conf
            rest[a:b] = True

    # 4f. a filled stretch that wanders away from the straight line between its two ends and comes back
    #     (a detour the ball did not make) -> the straight line instead. Not for a ball carried by a player.
    detour = np.zeros(T, bool)
    if P.ball_detour_m > 0:
        for a, b in run_lengths(np.isfinite(bx) & ~np.isfinite(tbx)):
            if a == 0 or b >= T or followed[a:b].any():
                continue
            if P.ball_detour_ends_m > 0 and np.hypot(tbx[b] - tbx[a - 1], tby[b] - tby[a - 1]) > P.ball_detour_ends_m:
                continue                                # only when the ball comes back to about where it was
            w = np.arange(1, b - a + 1) / (b - a + 1)
            lx, ly = tbx[a - 1] + (tbx[b] - tbx[a - 1]) * w, tby[a - 1] + (tby[b] - tby[a - 1]) * w
            if np.nanmax(np.hypot(bx[a:b] - lx, by[a:b] - ly)) > P.ball_detour_m:
                bx[a:b], by[a:b] = lx, ly
                detour[a:b] = True

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
    conf = np.maximum(conf, P.ball_conf_lowest)
    conf[~np.isfinite(bx)] = 0.0

    rep = {"detected": int(np.isfinite(bx_raw).sum()), "trusted": int((status == TRUSTED).sum()),
           "interpolated": int((status == INTERP).sum()), "rejected": int((status == REJECTED).sum()),
           "rejected_by_reason": reason, "tracklets": int(n_tr), "flight_frames": int(flight.sum()),
           "followed": int(followed.sum()), "kicked": int(kicked.sum()),
           "passed": int(passed.sum()), "curved": int(curved.sum()), "lined": int(lined.sum()),
           "rest": int(rest.sum()), "detour": int(detour.sum())}
    return dict(bx=bx, by=by, conf=conf, status=status, flight=flight), rep
