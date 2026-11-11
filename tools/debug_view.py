"""Debug view of the tracking: video with boxes on the left, top-down pitch on the right, plus a text report.

    python tools/debug_view.py --repo ../more-than-a-hack-2027 --start 10:00 --seconds 60 \
        --out-video results/debug_1000.mp4 --out-txt results/debug_ball.txt

Video (left)                       Pitch (right, does not overlap the video)
  home players  blue box             home players  blue dot
  away players  red box              away players  red dot
  ball detected yellow box           ball detected yellow dot
  ball interpolated orange box       ball interpolated orange dot

Bottom panel: Smart Tagging. Tags active now are listed by their label text (e.g. Kick Off (Start)), with team, time and code and a
+-15 s timeline shows each tag as a bar (blue = home team, red = away team, grey = no team),
with a white line at "now". The text report also lists every tag with its video time.

Every tracking position is treated as the BOTTOM-CENTRE of its box (the point on the ground).
The tracking has no box size, so the box is drawn with an approximate real size
(player 0.9 m x 1.9 m, ball 0.5 m) converted to pixels with the homography at that spot.

--v2 shows version 2 instead: removed players grey, team colours after the goalkeeper fix,
ball trusted yellow / interpolated orange / rejected magenta (possession/v2/*).

--frames <prefix>_frames.csv (from pipeline_v2 --dead-debug) adds OUR result: the outer frame of the
video is our label (H blue / A red / D grey), and a bottom panel shows P(dead), the pseudo label and
the rule behind it, the features that pushed P(dead), and +-10 s strips of ours / pseudo / P(dead).
--auto short_d|long_d|play|all picks clips from our result (one video per category + an index).

Ball gaps of up to --interp-max-s seconds (default 1.0) with a detection on both sides are filled
by a straight line and drawn in orange. Longer gaps are left empty.

The text report always covers the whole match (graded halves), not only the video range:
detected / interpolated / missing counts, gap lengths, every interpolated and missing stretch
with its video time, and one-frame ball jumps faster than --jump-mps (likely false detections).
No possession labels are used.
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from possession import data  # noqa: E402
from possession.config import Params  # noqa: E402
sys.path.insert(0, HERE)
from tag_io import load_tags  # noqa: E402

FPS = 25
BLUE, RED = (230, 90, 20), (40, 40, 225)            # BGR
YELLOW, ORANGE, MAGENTA = (0, 230, 255), (0, 140, 255), (200, 0, 200)
LINE, GRASS = (235, 235, 235), (60, 110, 50)


# ---------------------------------------------------------------- helpers
def to_frame(text):
    if ":" in str(text):
        m, s = str(text).split(":")
        return int(round((int(m) * 60 + float(s)) * FPS))
    return int(text)


def tstr(f):
    s = f / FPS
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def runs(mask):
    """[(start, end_inclusive)] of True runs."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def interpolate_ball(bx, by, max_gap):
    """Fill NaN runs of <= max_gap frames that have a detection on both sides. Returns x, y, filled mask."""
    x, y = bx.astype(float).copy(), by.astype(float).copy()
    filled = np.zeros(len(x), bool)
    for a, b in runs(~np.isfinite(bx)):
        if a == 0 or b == len(x) - 1 or (b - a + 1) > max_gap:
            continue
        w = np.arange(1, b - a + 2) / (b - a + 2)
        x[a:b + 1] = bx[a - 1] + (bx[b + 1] - bx[a - 1]) * w
        y[a:b + 1] = by[a - 1] + (by[b + 1] - by[a - 1]) * w
        filled[a:b + 1] = True
    return x, y, filled


def project(H, x, y):
    den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den


def box_on_ground(img, H, x, y, L, W, width_m, height_m, colour, thick):
    """Draw a box whose bottom-centre is the pitch point (x, y) in metres."""
    u, v = project(H, x / L, y / W)
    u2, v2 = project(H, (x + 1.0) / L, y / W)
    ppm = max(3.0, float(np.hypot(u2 - u, v2 - v)))          # pixels per metre at that spot
    bw, bh = width_m * ppm, height_m * ppm
    cv2.rectangle(img, (int(u - bw / 2), int(v - bh)), (int(u + bw / 2), int(v)), colour, thick)


def pitch_canvas(h, L, W):
    """Empty top-down pitch of height h. Returns image and a metres->pixels function."""
    mx, top, bottom = 20, 40, 12                              # leave room for the info bar
    s = (h - top - bottom) / W
    w = int(L * s + 2 * mx)
    img = np.full((h, w, 3), GRASS, np.uint8)

    def P(x, y):
        return int(mx + x * s), int(top + y * s)

    cv2.rectangle(img, P(0, 0), P(L, W), LINE, 2)
    cv2.line(img, P(L / 2, 0), P(L / 2, W), LINE, 2)
    cv2.circle(img, P(L / 2, W / 2), int(9.15 * s), LINE, 2)
    for x0, sgn in ((0, 1), (L, -1)):
        cv2.rectangle(img, P(x0, W / 2 - 20.16), P(x0 + sgn * 16.5, W / 2 + 20.16), LINE, 2)
        cv2.rectangle(img, P(x0, W / 2 - 9.16), P(x0 + sgn * 5.5, W / 2 + 9.16), LINE, 2)
    return img, P, s


# ---------------------------------------------------------------- smart tagging
TAG_PANEL_H = 170
GREY = (150, 150, 150)


def tag_colour(tag, side_of):
    side = side_of(tag["tags"].get("Team"))
    return BLUE if side == 0 else (RED if side == 1 else GREY)


def draw_tag_panel(width, t, tags, side_of, window_s=15):
    panel = np.full((TAG_PANEL_H, width, 3), 25, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # left: tags active now
    active = [g for g in tags if g["start"] <= t < g["end"]]
    cv2.putText(panel, f"SMART TAGGING  active now: {len(active)}", (10, 20), font, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    for i, g in enumerate(active[:6]):
        txt = f"{g['name']}  [{g['tags'].get('Team', '-')}]  {tstr(g['start'])}-{tstr(g['end'])}  ({g['code']})"
        cv2.putText(panel, txt[:110], (10, 42 + 20 * i), font, 0.45, tag_colour(g, side_of), 1, cv2.LINE_AA)
    if len(active) > 6:
        cv2.putText(panel, f"... +{len(active) - 6} more", (10, 42 + 20 * 6), font, 0.45, GREY, 1, cv2.LINE_AA)
    # right: timeline +-window_s
    x0, x1 = width // 2 + 10, width - 10
    y0, row_h, n_rows = 30, 22, 6
    span = window_s * FPS
    a, b = t - span, t + span
    near = sorted([g for g in tags if g["end"] > a and g["start"] < b], key=lambda g: g["start"])
    ends = [-1] * n_rows

    def X(f):
        return int(x0 + (min(max(f, a), b) - a) / (b - a) * (x1 - x0))

    for g in near:
        r = next((k for k in range(n_rows) if ends[k] <= g["start"]), None)
        if r is None:
            continue
        ends[r] = g["end"]
        y = y0 + r * row_h
        cv2.rectangle(panel, (X(g["start"]), y), (max(X(g["end"]), X(g["start"]) + 2), y + row_h - 4),
                      tag_colour(g, side_of), -1)
        cv2.putText(panel, g["name"][:28], (X(g["start"]) + 3, y + row_h - 9), font, 0.38, (255, 255, 255), 1,
                    cv2.LINE_AA)
    cv2.line(panel, (X(t), y0 - 8), (X(t), y0 + n_rows * row_h), (255, 255, 255), 2)
    cv2.putText(panel, f"-{window_s}s", (x0, 20), font, 0.4, GREY, 1, cv2.LINE_AA)
    cv2.putText(panel, f"+{window_s}s", (x1 - 35, 20), font, 0.4, GREY, 1, cv2.LINE_AA)
    return panel


def tag_report(add, tags, side_of, team_names):
    add(f"SMART TAGGING: {len(tags)} tags (times are VIDEO time; team: 0 = {team_names[0]}, 1 = {team_names[1]})")
    codes = {}
    for g in tags:
        codes.setdefault(g["code"], []).append(g)
    add("  count by code:")
    for c in sorted(codes, key=lambda c: -len(codes[c])):
        d = sum(g["end"] - g["start"] for g in codes[c]) / FPS
        add(f"    {c:<40} {len(codes[c]):5d}  total {d:8.1f} s")
    names = {}
    for g in tags:
        names.setdefault(g["name"], []).append(g)
    add("  count by label text:")
    for c in sorted(names, key=lambda c: -len(names[c])):
        d = sum(g["end"] - g["start"] for g in names[c]) / FPS
        add(f"    {c:<40} {len(names[c]):5d}  total {d:8.1f} s")
    add("  every tag (start frame, end frame, video time, length, label text, team, code):")
    for g in sorted(tags, key=lambda g: g["start"]):
        side = side_of(g["tags"].get("Team"))
        add(f"    {g['start']:6d}-{g['end']:6d}  {tstr(g['start'])}-{tstr(g['end'])}  "
            f"{(g['end'] - g['start']) / FPS:6.2f} s  {g['name']:<36} "
            f"{g['tags'].get('Team', '-')}({'-' if side is None else side})  {g['code']}")
    add("")


# ---------------------------------------------------------------- our result
RESULT_PANEL_H = 150
RES_COL = {"H": BLUE, "A": RED, "D": (170, 170, 170)}


def draw_result_panel(width, t, res, window_s=10):
    panel = np.full((RESULT_PANEL_H, width, 3), 15, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    r = res["rows"].get(t, {})
    lines = [f"OURS {res['lab'][t] or '-'}   P(dead) {r.get('p_dead', float('nan'))}   "
             f"pseudo {r.get('pseudo_dead', '')} {r.get('pseudo_rule', '') or ''}",
             f"mean speed {r.get('speed_mean', '')}  walking {r.get('frac_walking', '')}  ball seen {r.get('ball_seen', '')}"
             f"  since {r.get('since_seen', '')} s  ball speed {r.get('ball_speed', '')}  last out {r.get('last_ball_out', '')}",
             "pushed by: " + "  ".join(str(r.get(f"top{k}", "")) for k in range(1, 4)),
             "           " + "  ".join(str(r.get(f"top{k}", "")) for k in range(4, 6))]
    for i, s in enumerate(lines):
        cv2.putText(panel, s[:150], (10, 20 + 20 * i), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    span = window_s * FPS
    x0, x1 = 80, width - 10
    xs = np.linspace(x0, x1, 2 * span + 2).astype(int)
    for row, (name, arr, colour) in enumerate((
            ("ours", res["lab"], lambda v: RES_COL.get(v, (40, 40, 40))),
            ("pseudo", res["pseudo"], lambda v: {1: (170, 170, 170), 0: (60, 160, 60)}.get(int(v), (40, 40, 40))),
            ("P(dead)", res["pdead"], lambda v: tuple(int(c) for c in (np.array([60, 160, 60]) * (1 - v)
                                                                       + np.array([170, 170, 170]) * v))
             if np.isfinite(v) else (40, 40, 40)))):
        y0 = 92 + 18 * row
        cv2.putText(panel, name, (10, y0 + 12), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        for k, f in enumerate(range(t - span, t + span + 1)):
            if 0 <= f < len(arr):
                cv2.rectangle(panel, (xs[k], y0), (xs[k + 1], y0 + 14), colour(arr[f]), -1)
    cv2.line(panel, (xs[span], 88), (xs[span], 146), (255, 255, 255), 2)
    return panel


# ---------------------------------------------------------------- report
def write_report(path, tr, bxi, filled, blocks, max_gap, jump_mps, video_range, tags=(), side_of=None):
    det = np.isfinite(tr.bx)
    lines = []
    add = lines.append
    add("BALL REPORT (tactical tracking, whole match, graded halves only)")
    add(f"interpolation: gaps <= {max_gap / FPS:.2f} s with a detection on both sides, straight line")
    add("")
    tot = {"frames": 0, "det": 0, "interp": 0, "miss": 0}
    gaps_all = []
    for half, a, b in blocks:
        sl = slice(a, b + 1)
        n = b - a + 1
        d, f = int(det[sl].sum()), int(filled[sl].sum())
        miss = n - d - f
        for k, v in (("frames", n), ("det", d), ("interp", f), ("miss", miss)):
            tot[k] += v
        add(f"half {half}: frames {a}-{b} ({tstr(a)}-{tstr(b)}), {n} frames")
        add(f"  detected      {d:7d}  {d / n:6.1%}")
        add(f"  interpolated  {f:7d}  {f / n:6.1%}")
        add(f"  missing       {miss:7d}  {miss / n:6.1%}")
        gaps_all += [(s + a, e + a) for s, e in runs(~det[sl])]
    n = tot["frames"]
    add("both halves:")
    add(f"  detected      {tot['det']:7d}  {tot['det'] / n:6.1%}")
    add(f"  interpolated  {tot['interp']:7d}  {tot['interp'] / n:6.1%}")
    add(f"  missing       {tot['miss']:7d}  {tot['miss'] / n:6.1%}")
    add("")

    add("GAP LENGTHS (runs of frames with no detection)")
    lens = np.array([(e - s + 1) / FPS for s, e in gaps_all]) if gaps_all else np.zeros(0)
    edges = [0, 0.2, 0.5, 1, 2, 5, 10, 30, 1e9]
    names = ["<=0.2s", "0.2-0.5s", "0.5-1s", "1-2s", "2-5s", "5-10s", "10-30s", ">30s"]
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        k = (lens > lo) & (lens <= hi)
        add(f"  {nm:>9}: {int(k.sum()):6d} gaps, {lens[k].sum():8.1f} s")
    add("")

    # one-frame jumps between consecutive detections
    v = np.hypot(np.diff(tr.bx), np.diff(tr.by)) * FPS
    jump = np.flatnonzero(np.isfinite(v) & (v > jump_mps)) + 1
    inside = np.zeros(len(det), bool)
    for _, a, b in blocks:
        inside[a:b + 1] = True
    jump = jump[inside[jump]]
    add(f"BALL JUMPS faster than {jump_mps:.0f} m/s between two consecutive frames: {len(jump)}")
    add("  (a real ball rarely exceeds ~35 m/s; these are likely false detections)")
    for f in jump:
        add(f"  frame {f:6d}  {tstr(f)}  {v[f - 1]:6.1f} m/s")
    add("")

    add("INTERPOLATED STRETCHES (start frame, end frame, video time, length)")
    for s, e in runs(filled & inside):
        add(f"  {s:6d}-{e:6d}  {tstr(s)}-{tstr(e)}  {(e - s + 1) / FPS:5.2f} s")
    add("")
    add("MISSING STRETCHES AFTER INTERPOLATION (start frame, end frame, video time, length)")
    for s, e in runs(~np.isfinite(bxi) & inside):
        add(f"  {s:6d}-{e:6d}  {tstr(s)}-{tstr(e)}  {(e - s + 1) / FPS:6.2f} s")
    add("")
    if video_range:
        a, b = video_range
        sl = slice(a, b)
        n = b - a
        add(f"VIDEO RANGE {tstr(a)}-{tstr(b)}: detected {det[sl].mean():.1%}, "
            f"interpolated {filled[sl].mean():.1%}, missing {(~np.isfinite(bxi[sl])).mean():.1%} ({n} frames)")
        add("")
    if tags:
        tag_report(add, tags, side_of, tr.team_names)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--start", default="0", help="video time mm:ss or a frame number")
    ap.add_argument("--seconds", type=float, default=60, help="video length; 0 = to the end")
    ap.add_argument("--out-video", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--interp-max-s", type=float, default=1.0)
    ap.add_argument("--jump-mps", type=float, default=40.0)
    ap.add_argument("--scale", type=float, default=0.5, help="video size (0.5 = half of 1920x1080)")
    ap.add_argument("--game", default="BAR-ATH")
    ap.add_argument("--data-dir")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--v2", action="store_true",
                    help="show what version 2 does: removed players grey, corrected team colours, "
                         "ball by trust (yellow trusted, orange interpolated, magenta rejected)")
    ap.add_argument("--params", help="JSON overriding Params2 fields (with --v2), same as eval_cleaning.py")
    ap.add_argument("--frames", help="<prefix>_frames.csv from pipeline_v2 --dead-debug: show OUR result "
                                     "(outer frame colour = our label, P(dead), pseudo label and why)")
    ap.add_argument("--auto", choices=["short_d", "long_d", "play", "all"],
                    help="with --frames: pick clips automatically instead of --start/--seconds")
    ap.add_argument("--per-category", type=int, default=5)
    ap.add_argument("--pad-s", type=float, default=3.0)
    ap.add_argument("--max-s", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    repo = os.path.abspath(a.repo)
    if not os.path.isfile(os.path.join(repo, "games.py")):
        raise SystemExit(f"--repo does not point at more-than-a-hack-2027 (games.py not found): {repo}")
    sys.path.insert(0, repo)
    import games  # noqa: E402
    P = Params()
    data_dir = a.data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(repo, "data")
    paths = games.asset_paths(a.game, data_dir)
    n = games.source_frames("tactical", a.game)
    if a.game in games.GAMES:
        P.pitch_length, P.pitch_width = games.GAMES[a.game]["pitch"]
    L, W = P.pitch_length, P.pitch_width
    blocks = data.graded_blocks(repo, a.game)

    os.makedirs(a.cache_dir, exist_ok=True)
    print("loading tracking (first time takes a few minutes)", flush=True)
    tr = data.load_tracking(paths["atd_meta"], paths["atd_raw"], n, (L, W),
                            cache=os.path.join(a.cache_dir, f"{a.game}_tracking.npz"))
    Hs, _, valid = data.load_homography(paths["homography"], n)
    tags = load_tags(paths["smart_tagging"], FPS) if os.path.isfile(paths.get("smart_tagging", "")) else []
    side_of = lambda name: data.tag_team_side(name, tr.team_names)  # noqa: E731
    print(f"Smart Tagging: {len(tags)} tags; team mapping "
          f"{ {k: side_of(k) for k in sorted({g['tags'].get('Team') for g in tags} - {None})} }", flush=True)
    max_gap = int(round(a.interp_max_s * FPS))
    bxi, byi, filled = interpolate_ball(tr.bx, tr.by, max_gap)
    team_mat = np.repeat(tr.team[None, :].astype(np.int8), n, axis=0)
    removed = np.zeros(tr.px.shape, bool)
    rejected = np.zeros(n, bool)
    if a.v2:
        from possession import preprocess
        from possession.v2.ball_quality import INTERP, REJECTED
        from possession.v2.ball_quality import assess_ball
        from possession.v2.clean_players import clean_players
        from possession.v2.config_v2 import Params2
        P2 = Params2()
        if a.params:
            import json
            with open(a.params) as fh:
                for k, v in json.load(fh).items():
                    setattr(P2, k, v)
        P2.pitch_length, P2.pitch_width = L, W
        attack = preprocess.attack_sign(tr, blocks, FPS)
        from possession.v2.tag_policy import event_frames
        cpx, cpy, team_mat, rep_a = clean_players(tr, attack, P2, event_mask=event_frames(tags, n, P2))
        removed = np.isfinite(tr.px) & np.isnan(cpx)
        bq, rep_b = assess_ball(tr.bx, tr.by, cpx, cpy, P2)
        rejected = np.isfinite(tr.bx) & (bq["status"] == REJECTED)
        bxi, byi = np.where(rejected, tr.bx, bq["bx"]), np.where(rejected, tr.by, bq["by"])
        filled = bq["status"] == INTERP
        print(f"v2 players: {rep_a}\nv2 ball: {rep_b}", flush=True)

    # ---- our current result (pipeline_v2 --dead-debug <prefix> -> <prefix>_frames.csv)
    res = None
    if a.frames:
        import pandas as pd
        fr = pd.read_csv(a.frames)
        res = {"lab": np.full(n, "", dtype="<U1"), "pdead": np.full(n, np.nan), "pseudo": np.full(n, -1),
               "rows": {int(f): r for f, r in zip(fr["frame"].values, fr.to_dict("records"))}, "fr": fr}
        res["lab"][fr["frame"].values] = fr["label"].astype(str).values
        res["pdead"][fr["frame"].values] = fr["p_dead"].values
        res["pseudo"][fr["frame"].values] = fr["pseudo_dead"].values

    cap = cv2.VideoCapture(paths["video"])
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {paths['video']}")
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), n)
    f0 = to_frame(a.start)
    f1 = total if a.seconds <= 0 else min(total, f0 + int(a.seconds * FPS))

    write_report(a.out_txt, tr, bxi, filled, blocks, max_gap, a.jump_mps, (f0, f1), tags, side_of)
    print(f"wrote {a.out_txt}", flush=True)

    vh = int(1080 * a.scale) // 2 * 2
    vw = int(1920 * a.scale) // 2 * 2
    pitch0, PP, _ = pitch_canvas(vh, L, W)
    out_w = (vw + pitch0.shape[1]) // 2 * 2
    out_h = vh + TAG_PANEL_H + (RESULT_PANEL_H if res else 0)
    team_now = lambda t, k: int(((team_mat[t] == k) & np.isfinite(tr.px[t]) & ~removed[t]).sum())  # noqa: E731

    def draw(t, img, title=""):
        H = Hs[t] if valid[t] and np.isfinite(Hs[t]).all() else None
        pitch = pitch0.copy()
        for s in range(tr.px.shape[1]):
            x, y = tr.px[t, s], tr.py[t, s]
            if not np.isfinite(x):
                continue
            col = GREY if removed[t, s] else (BLUE if team_mat[t, s] == 0 else RED)
            if H is not None:
                box_on_ground(img, H, x, y, L, W, 0.9, 1.9, col, 3)
            cv2.circle(pitch, PP(x, y), 6, col, -1)
        if np.isfinite(bxi[t]):
            col = MAGENTA if rejected[t] else (ORANGE if filled[t] else YELLOW)
            if H is not None:
                box_on_ground(img, H, bxi[t], byi[t], L, W, 0.5, 0.5, col, 3)
            cv2.circle(pitch, PP(bxi[t], byi[t]), 6, col, -1)
            cv2.circle(pitch, PP(bxi[t], byi[t]), 6, (0, 0, 0), 1)
        state = ("REJECTED" if rejected[t] else "detected" if np.isfinite(tr.bx[t])
                 else ("interpolated" if filled[t] else "MISSING"))
        frame = cv2.resize(img, (vw, vh), interpolation=cv2.INTER_AREA)
        if res:                                       # outer frame of the video = our label
            lab = res["lab"][t]
            c = {"H": BLUE, "A": RED, "D": (170, 170, 170)}.get(lab, (0, 0, 0))
            cv2.rectangle(frame, (0, 0), (vw - 1, vh - 1), c, 10)
        canvas = np.zeros((out_h, out_w, 3), np.uint8)
        canvas[:vh, :vw] = frame
        canvas[:vh, vw:vw + pitch.shape[1]] = pitch[:, : out_w - vw]
        canvas[vh:vh + TAG_PANEL_H] = draw_tag_panel(out_w, t, tags, side_of)
        if res:
            canvas[vh + TAG_PANEL_H:] = draw_result_panel(out_w, t, res)
        info = (f"{title}  {tstr(t)}  frame {t}  ball {state}  "
                f"home {team_now(t, 0)}  away {team_now(t, 1)}" + ("  [v2]" if a.v2 else "")
                + (f"  OURS {res['lab'][t] or '-'}  P(dead) {res['pdead'][t]:.2f}" if res else ""))
        cv2.rectangle(canvas, (0, 0), (out_w, 30), (0, 0, 0), -1)
        cv2.putText(canvas, info.strip(), (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def write_range(out, a0, a1, title=""):
        cap.set(cv2.CAP_PROP_POS_FRAMES, a0)
        t = a0
        ok, img = cap.read()
        while ok and t < a1:
            out.write(draw(t, img, title))
            t += 1
            if (t - a0) % (60 * FPS) == 0:
                print(f"  {(t - a0) // FPS} s written", flush=True)
            ok, img = cap.read()
        return t - a0

    os.makedirs(os.path.dirname(os.path.abspath(a.out_video)), exist_ok=True)
    if a.auto:
        if not res:
            raise SystemExit("--auto needs --frames (pipeline_v2 --dead-debug <prefix> writes <prefix>_frames.csv)")
        sys.path.insert(0, HERE)
        from dead_view import pick_clips
        cats, _ = pick_clips(res["fr"], n, a.per_category, np.random.default_rng(a.seed))
        want = list(cats) if a.auto == "all" else [a.auto]
        base, ext = os.path.splitext(a.out_video)
        index = []
        for cat in want:
            clips = cats.get(cat, [])
            if not clips:
                continue
            path = f"{base}_{cat}{ext or '.mp4'}"
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (out_w, out_h))
            index.append(f"[{cat}] -> {path}")
            for i, (s, e) in enumerate(clips):
                r = res["rows"].get(int((s + e) // 2), {})
                index.append(f"  {i + 1}. {tstr(s)}-{tstr(e)} ({(e - s) / FPS:.1f} s)  pseudo {r.get('pseudo_rule', '')}"
                             f"  P(dead) {r.get('p_dead', '')}  top {r.get('top1', '')}")
                card = np.zeros((out_h, out_w, 3), np.uint8)
                cv2.putText(card, f"{cat} #{i + 1}   {tstr(s)} - {tstr(e)}  ({(e - s) / FPS:.1f} s)",
                            (40, out_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                for _ in range(FPS):
                    out.write(card)
                a0 = max(0, int(s - a.pad_s * FPS))
                a1 = min(total, int(min(e + a.pad_s * FPS, a0 + a.max_s * FPS)))
                write_range(out, a0, a1, f"{cat} #{i + 1}")
            out.release()
            print(f"wrote {path} ({len(clips)} clips)", flush=True)
        with open(f"{base}_index.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(index) + "\n")
        print(f"wrote {base}_index.txt")
    else:
        out = cv2.VideoWriter(a.out_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (out_w, out_h))
        k = write_range(out, f0, f1)
        out.release()
        print(f"wrote {a.out_video} ({k / FPS:.0f} s)")
    cap.release()


if __name__ == "__main__":
    main()
