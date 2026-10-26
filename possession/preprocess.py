"""Signal clean-up shared by all modules: velocities, short-gap interpolation, the camera aim
point, attacking direction and NaN-aware rolling windows."""
import numpy as np


def shift(x, k, fill=np.nan):
    """x shifted by k frames along axis 0 (k>0: value from the past)."""
    out = np.full_like(x, fill, dtype=np.float64)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def velocity(x, half_window, fps):
    """Centred difference over +/- half_window frames (m/s). NaN where either end is missing."""
    dt = 2.0 * half_window / fps
    return (shift(x, -half_window) - shift(x, half_window)) / dt


def interp_short_gaps(x, max_gap):
    """Linearly fill NaN runs no longer than max_gap that have valid values on both sides."""
    x = np.asarray(x, dtype=np.float64).copy()
    nan = np.isnan(x)
    if not nan.any():
        return x
    idx = np.flatnonzero(np.diff(np.r_[0, nan.astype(np.int8), 0]))
    for start, stop in zip(idx[::2], idx[1::2]):
        if start == 0 or stop >= len(x) or stop - start > max_gap:
            continue
        x[start:stop] = np.linspace(x[start - 1], x[stop], stop - start + 2)[1:-1]
    return x


def rolling(x, window, side="past", stat="mean"):
    """NaN-aware rolling mean over [t-w, t] ('past'), [t, t+w] ('future') or both ('centre').

    Works on 1-D arrays; O(T) with cumulative sums whatever the window.
    """
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0)
    cs = np.r_[0.0, np.cumsum(v)]
    cn = np.r_[0, np.cumsum(ok)]
    T = len(x)
    t = np.arange(T)
    lo = {"past": t - window, "future": t, "centre": t - window}[side]
    hi = {"past": t, "future": t + window, "centre": t + window}[side]
    lo = np.clip(lo, 0, T)
    hi = np.clip(hi + 1, 0, T)
    n = cn[hi] - cn[lo]
    s = cs[hi] - cs[lo]
    if stat == "count":
        return n.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def run_lengths(mask):
    """[(start, stop)] of True runs (stop exclusive)."""
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(idx[::2], idx[1::2]))


def camera_aim(H, valid, image_size, pitch):
    """Pitch point (metres) at the centre of the tactical image, per frame. NaN when invalid."""
    T = len(valid)
    ax = np.full(T, np.nan)
    ay = np.full(T, np.nan)
    ok = np.flatnonzero(valid & np.isfinite(H).all(axis=(1, 2)))
    if len(ok):
        inv = np.linalg.inv(H[ok])
        cx, cy = image_size[0] / 2.0, image_size[1] / 2.0
        den = inv[:, 2, 0] * cx + inv[:, 2, 1] * cy + inv[:, 2, 2]
        ax[ok] = (inv[:, 0, 0] * cx + inv[:, 0, 1] * cy + inv[:, 0, 2]) / den * pitch[0]
        ay[ok] = (inv[:, 1, 0] * cx + inv[:, 1, 1] * cy + inv[:, 1, 2]) / den * pitch[1]
    return ax, ay


def half_index(n_frames, blocks):
    """Per frame: index into blocks of the half it belongs to (nearest half outside them)."""
    centres = np.array([(a + b) / 2.0 for _, a, b in blocks])
    t = np.arange(n_frames)
    return np.abs(t[:, None] - centres[None, :]).argmin(axis=1)


def attack_sign(tr, blocks, fps, first_s=60.0):
    """(T, 2) array: +1 if that team attacks towards +x in that frame's half, else -1.

    Inferred from data, per half: over the first minute of the half, the team whose players sit
    further towards -x is defending the -x goal and therefore attacks +x.
    """
    half = half_index(tr.n_frames, blocks)
    sign = np.ones((tr.n_frames, 2))
    for h, (_, a, b) in enumerate(blocks):
        seg = slice(a, min(b + 1, a + int(first_s * fps)))
        mx = [np.nanmean(tr.px[seg][:, tr.team == k]) for k in (0, 1)]
        home_attacks_plus = mx[0] < mx[1]
        rows = half == h
        sign[rows, 0] = 1 if home_attacks_plus else -1
        sign[rows, 1] = -sign[rows, 0]
    return sign


def prepare(tr, H, valid, blocks, P):
    """Everything later modules need, as a dict of arrays (all in metres / seconds)."""
    hw = P.frames(P.velocity_half_window_s)
    gap = P.frames(P.ball_max_interp_gap_s)
    bx = interp_short_gaps(tr.bx, gap)
    by = interp_short_gaps(tr.by, gap)
    ball_seen = ~np.isnan(tr.bx)
    ball_ok = ~np.isnan(bx)
    pvx = velocity(tr.px, hw, P.fps)
    pvy = velocity(tr.py, hw, P.fps)
    bvx = velocity(bx, hw, P.fps)
    bvy = velocity(by, hw, P.fps)
    ax, ay = camera_aim(H, valid, P.image_size, (P.pitch_length, P.pitch_width))
    cam_speed = np.hypot(velocity(ax, hw, P.fps), velocity(ay, hw, P.fps))

    # last seen / next seen ball position, used when the ball is missing
    idx = np.where(ball_seen, np.arange(tr.n_frames), -1)
    last = np.maximum.accumulate(idx)
    nxt_idx = np.where(ball_seen, np.arange(tr.n_frames), tr.n_frames)
    nxt = np.minimum.accumulate(nxt_idx[::-1])[::-1]
    t = np.arange(tr.n_frames)
    since_seen = np.where(last >= 0, (t - last) / P.fps, np.nan)
    until_seen = np.where(nxt < tr.n_frames, (nxt - t) / P.fps, np.nan)
    last_c = np.clip(last, 0, None)
    lbx = np.where(last >= 0, tr.bx[last_c], np.nan)
    lby = np.where(last >= 0, tr.by[last_c], np.nan)

    # ball region: ball if visible, else last seen ball if recent, else the camera aim point
    recent = since_seen <= 2.0
    rx = np.where(ball_ok, bx, np.where(recent, lbx, ax))
    ry = np.where(ball_ok, by, np.where(recent, lby, ay))

    return dict(
        px=tr.px.astype(np.float64), py=tr.py.astype(np.float64), pvx=pvx, pvy=pvy,
        pspeed=np.hypot(pvx, pvy), team=tr.team,
        bx=bx, by=by, bvx=bvx, bvy=bvy, bspeed=np.hypot(bvx, bvy),
        ball_seen=ball_seen, ball_ok=ball_ok, since_seen=since_seen, until_seen=until_seen,
        last_bx=lbx, last_by=lby, aim_x=ax, aim_y=ay, cam_speed=cam_speed,
        region_x=rx, region_y=ry, attack=attack_sign(tr, blocks, P.fps),
        n_frames=tr.n_frames,
    )
