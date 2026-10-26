"""Draw the tracking and our estimate on top of the tactical video, to watch it again and again.

    python -m possession.pipeline --repo ../more-than-a-hack-2027 --out results/v0.csv --dump results/v0_detail.csv
    python tools/overlay_video.py --repo ../more-than-a-hack-2027 --detail results/v0_detail.csv \
        --start 20:00 --seconds 60 --out results/review_2000.mp4

What is drawn
    * outer frame colour = estimated label: H blue, A orange, D grey, X black
    * boxes around players: home blue, away orange. The tracking has only a foot position, so the
      box is an APPROXIMATE size made from the homography (about 0.9 m x 1.9 m), not a detection box
    * ball: yellow ring (hollow grey ring = last place the ball was seen, when it is not tracked)
    * top bar: video time, frame, label, P(dead), P(home), pseudo label
    * bottom strip: labels for 10 s before and after, with a white marker at "now"

Use it to understand the data and to find bugs and general failure patterns. Do NOT write down
"frames X..Y are really H" and use that for training, scoring or picking settings -- that is
hand labelling of the provided game, which the rules forbid. Pick settings on other games
(tools/convert_metrica.py + --external).
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from possession import data  # noqa: E402
from possession.config import Params  # noqa: E402

FPS = 25
# BGR colours
COL = {"H": (176, 108, 43), "A": (32, 107, 221), "D": (150, 150, 150), "X": (0, 0, 0)}
TEAM = {0: COL["H"], 1: COL["A"]}
BALL = (0, 230, 255)


def to_frame(text):
    if ":" in str(text):
        m, s = str(text).split(":")
        return int(round((int(m) * 60 + float(s)) * FPS))
    return int(text)


def project(H, x, y):
    """normalised pitch (0-1) -> pixels; arrays allowed."""
    den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den


def draw_frame(img, t, H, tr, L, W, row, labels, last_ball, border):
    h, w = img.shape[:2]
    if H is not None and np.isfinite(H).all():
        # players: approximate box from foot point and local metres-per-pixel scale
        for s in range(tr.px.shape[1]):
            x, y = tr.px[t, s], tr.py[t, s]
            if not np.isfinite(x):
                continue
            u, v = project(H, x / L, y / W)
            u2, v2 = project(H, (x + 1.0) / L, y / W)
            scale = max(4.0, float(np.hypot(u2 - u, v2 - v)))       # pixels per metre
            bw, bh = 0.9 * scale, 1.9 * scale
            p1 = (int(u - bw / 2), int(v - bh))
            p2 = (int(u + bw / 2), int(v))
            cv2.rectangle(img, p1, p2, TEAM.get(int(tr.team[s]), (255, 255, 255)), 2)
        if np.isfinite(tr.bx[t]):
            u, v = project(H, tr.bx[t] / L, tr.by[t] / W)
            cv2.circle(img, (int(u), int(v)), 9, BALL, 3)
        elif last_ball is not None:
            u, v = project(H, last_ball[0] / L, last_ball[1] / W)
            cv2.circle(img, (int(u), int(v)), 9, (128, 128, 128), 1)

    lab = row["label"]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), COL.get(lab, (0, 0, 0)), border)

    # top bar
    sec = t / FPS
    pseudo = ""
    if "pseudo_dead" in row:
        pseudo = "D" if row["pseudo_dead"] == 1 else ({1: "H", 0: "A"}.get(int(row["pseudo_team"]), "-"))
    text = (f"{int(sec // 60):02d}:{sec % 60:05.2f}  frame {t}  label {lab}  "
            f"P(dead) {row['p_dead']:.2f}  P(home) {row['p_home']:.2f}  pseudo {pseudo}"
            + ("" if np.isfinite(tr.bx[t]) else "  [ball not tracked]"))
    cv2.rectangle(img, (border, border), (w - border, border + 44), (0, 0, 0), -1)
    cv2.putText(img, text, (border + 10, border + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                cv2.LINE_AA)

    # bottom strip: +-10 s of labels
    span = 10 * FPS
    y0, y1 = h - border - 26, h - border - 6
    x0, x1 = border + 10, w - border - 10
    a, b = max(0, t - span), min(len(labels) - 1, t + span)
    xs = np.linspace(x0, x1, 2 * span + 2).astype(int)
    for k, f in enumerate(range(t - span, t + span + 1)):
        if a <= f <= b:
            cv2.rectangle(img, (xs[k], y0), (xs[k + 1], y1), COL.get(labels[f], (0, 0, 0)), -1)
    mid = xs[span]
    cv2.line(img, (mid, y0 - 6), (mid, y1 + 6), (255, 255, 255), 3)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--detail", required=True, help="CSV written by the pipeline with --dump")
    ap.add_argument("--start", default="0", help="video time mm:ss or a frame number")
    ap.add_argument("--seconds", type=float, default=60, help="length; 0 = to the end of the video")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--scale", type=float, default=0.5, help="output size relative to the video (0.5 = half)")
    ap.add_argument("--game", default="BAR-ATH")
    ap.add_argument("--data-dir")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--video", help="override the video path")
    a = ap.parse_args(argv)

    repo = os.path.abspath(a.repo)
    sys.path.insert(0, repo)
    import games  # noqa: E402
    P = Params()
    data_dir = a.data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(repo, "data")
    paths = games.asset_paths(a.game, data_dir)
    n = games.source_frames("tactical", a.game)
    if a.game in games.GAMES:
        P.pitch_length, P.pitch_width = games.GAMES[a.game]["pitch"]
    L, W = P.pitch_length, P.pitch_width

    os.makedirs(a.cache_dir, exist_ok=True)
    tr = data.load_tracking(paths["atd_meta"], paths["atd_raw"], n, (L, W),
                            cache=os.path.join(a.cache_dir, f"{a.game}_tracking.npz"))
    Hs, _, valid = data.load_homography(paths["homography"], n)
    det = pd.read_csv(a.detail)
    labels = det["label"].astype(str).to_numpy()
    video = a.video or paths["video"]

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = to_frame(a.start)
    f1 = total if a.seconds <= 0 else min(total, f0 + int(a.seconds * FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ok, img = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {f0}")
    h, w = img.shape[:2]
    ow, oh = int(w * a.scale) // 2 * 2, int(h * a.scale) // 2 * 2
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (ow, oh))
    border = max(12, int(0.025 * h))

    # last seen ball before the start
    seen = np.flatnonzero(np.isfinite(tr.bx[:f0 + 1]))
    last_ball = (tr.bx[seen[-1]], tr.by[seen[-1]]) if len(seen) else None

    t = f0
    while ok and t < f1:
        if np.isfinite(tr.bx[t]):
            last_ball = (tr.bx[t], tr.by[t])
        H = Hs[t] if valid[t] else None
        draw_frame(img, t, H, tr, L, W, det.iloc[t], labels, last_ball, border)
        out.write(cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA))
        t += 1
        if (t - f0) % (60 * FPS) == 0:
            print(f"  {(t - f0) // FPS} s written", flush=True)
        ok, img = cap.read()
    out.release()
    cap.release()
    print(f"wrote {a.out} ({(t - f0) / FPS:.0f} s)")


if __name__ == "__main__":
    main()
