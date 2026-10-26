"""Convert a Metrica Sports public sample game into the same file formats as the challenge data.

Source (public, labelled by Metrica, NOT the challenge game):
    https://github.com/metrica-sports/sample-data   (data/Sample_Game_1, data/Sample_Game_2)

    python tools/convert_metrica.py --src sample-data/data/Sample_Game_1 --out external/MET-G1

Writes into --out:
    game.json            n_frames, blocks (one per period), pitch, stem
    <stem>_atd.xml       teams and track slots (home first)
    <stem>_atd.txt       "frame:p1;p2;...:ball", groups "x,y,speed", 0-1 normalised
    gt.csv               frame,label (H/A/D/X) built from Metrica's hand-made event data

The sample games are tracked with full coverage, unlike the challenge's tactical camera. By default
the converter hides the ball in extra random bursts (the sample
already misses it on ~39% of frames, mostly when out of play) and hides players far from the ball
some of the time, so the input looks more like the challenge data. Use --no-degrade to keep all.

How gt.csv is built (write these rules in the technical document):
    * team in possession = team of the latest "owning" event (PASS, RECOVERY, BALL LOST, SHOT,
      SET PIECE, CHALLENGE ...-WON) — it lasts until another team's owning event starts
    * D (dead ball) starts at BALL OUT, a SHOT that goes out or is a goal, or FAULT RECEIVED,
      and lasts until the next SET PIECE / goal kick / kick-off
    * X outside each period
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

FPS = 25
OWN_TYPES = {"PASS", "RECOVERY", "BALL LOST", "SHOT", "SET PIECE"}
RESTART_SUBTYPES = ("GOAL KICK", "KICK OFF", "FREE KICK", "THROW IN", "CORNER KICK", "PENALTY")


def read_tracking(path):
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    players = [c for c in cols if c.startswith("Player")]
    xy = {}
    for c in players:
        i = cols.index(c)
        xy[c] = raw.iloc[:, [i, i + 1]].to_numpy(float)
    ib = cols.index("Ball")
    ball = raw.iloc[:, [ib, ib + 1]].to_numpy(float)
    return raw["Period"].to_numpy(int), raw["Frame"].to_numpy(int), xy, ball


def build_gt(events, period, frame, n):
    gt = np.full(n, "X", dtype="<U1")
    ev = events.sort_values("Start Frame").reset_index(drop=True)
    idx = {f: i for i, f in enumerate(frame)}          # Metrica frame -> row
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        a, b = rows[0], rows[-1]
        team, dead = None, False
        e = ev[ev["Period"] == p]
        marks = []                                     # (row, team or None, dead flag)
        for _, r in e.iterrows():
            t, st = str(r["Type"]).upper(), str(r["Subtype"]).upper() if pd.notna(r["Subtype"]) else ""
            side = "H" if str(r["Team"]).lower().startswith("home") else "A"
            s = idx.get(int(r["Start Frame"]))
            if s is None:
                continue
            end = idx.get(int(r["End Frame"]), s)
            is_restart = t == "SET PIECE" or any(k in st for k in RESTART_SUBTYPES)
            if is_restart:
                marks.append((s, side, False))
            elif t == "BALL OUT" or (t == "SHOT" and ("OUT" in st or "GOAL" in st)):
                marks.append((max(s, end), None, True))
            elif t == "FAULT RECEIVED":
                marks.append((s, None, True))
            elif t in OWN_TYPES:
                marks.append((s, side, None))
            elif t == "CHALLENGE" and "WON" in st and "FAULT" not in st:
                marks.append((s, side, None))
        marks.sort(key=lambda m: m[0])
        cur = "D"                                      # each period starts with a kick-off
        last = a
        for s, side, d in marks:
            gt[last:s] = cur
            if d is True:
                cur = "D"
            elif side is not None:                     # restart, or an owning event (play is live)
                cur = side
            last = s
        gt[last:b + 1] = cur
    return gt


def degrade(px, py, bx, by, rng, ball_missing=0.4, far_m=40.0):
    T = len(bx)
    seen = np.ones(T, bool)
    t = 0
    rate = ball_missing / (1.6 * FPS * (1 - ball_missing) + 1e-9)
    while t < T:
        if rng.random() < rate:
            k = int(rng.uniform(0.2, 3.0) * FPS)
            seen[t:t + k] = False
            t += k
        t += 1
    bx, by = bx.copy(), by.copy()
    bx[~seen] = np.nan
    by[~seen] = np.nan
    cx = pd.Series(bx).interpolate(limit_direction="both").rolling(50, min_periods=1, center=True).mean().to_numpy()
    cy = pd.Series(by).interpolate(limit_direction="both").rolling(50, min_periods=1, center=True).mean().to_numpy()
    far = np.hypot(px - cx[:, None], py - cy[:, None]) > far_m
    drop = far & (rng.random(far.shape) < 0.5)
    px, py = px.copy(), py.copy()
    px[drop] = np.nan
    py[drop] = np.nan
    return px, py, bx, by


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="e.g. sample-data/data/Sample_Game_1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", help="file name prefix (default MET-<folder>)")
    ap.add_argument("--pitch", type=float, nargs=2, default=(105.0, 68.0))
    ap.add_argument("--no-degrade", action="store_true")
    ap.add_argument("--ball-missing", type=float, default=0.15,
                    help="EXTRA share of frames to hide the ball (the sample already misses ~39%%)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    name = os.path.basename(os.path.normpath(a.src))
    stem = a.stem or "MET-" + name.replace("Sample_Game_", "G")
    f_home = [f for f in os.listdir(a.src) if f.endswith("Home_Team.csv")][0]
    f_away = [f for f in os.listdir(a.src) if f.endswith("Away_Team.csv")][0]
    f_ev = [f for f in os.listdir(a.src) if f.endswith("EventsData.csv")][0]
    period, frame, home, ball = read_tracking(os.path.join(a.src, f_home))
    _, frame2, away, _ = read_tracking(os.path.join(a.src, f_away))
    assert (frame == frame2).all(), "home/away files do not line up"
    events = pd.read_csv(os.path.join(a.src, f_ev))

    L, W = a.pitch
    n = len(frame)
    names = list(home) + list(away)
    px = np.stack([v[:, 0] for v in list(home.values()) + list(away.values())], 1) * L
    py = np.stack([v[:, 1] for v in list(home.values()) + list(away.values())], 1) * W
    bx, by = ball[:, 0] * L, ball[:, 1] * W
    if not a.no_degrade:
        px, py, bx, by = degrade(px, py, bx, by, np.random.default_rng(a.seed), a.ball_missing)
    sp = np.hypot(np.gradient(px, axis=0), np.gradient(py, axis=0)) * FPS

    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, stem)
    with open(base + "_atd.xml", "w") as fh:
        fh.write('<?xml version="1.0"?>\n<Root><Teams><Team id="H"><Name>Home</Name></Team>'
                 '<Team id="A"><Name>Away</Name></Team></Teams><Players>')
        for i, nm in enumerate(names):
            tid = "H" if i < len(home) else "A"
            fh.write(f'<Player id="{tid}{nm}" teamId="{tid}"><Name>Track {i + 1}</Name></Player>')
        fh.write("</Players></Root>\n")
    with open(base + "_atd.txt", "w") as fh:
        for t in range(n):
            g = ["NaN,NaN,NaN" if np.isnan(px[t, s]) else f"{px[t, s] / L:.5f},{py[t, s] / W:.5f},{sp[t, s]:.2f}"
                 for s in range(px.shape[1])]
            b = "NaN,NaN,NaN" if np.isnan(bx[t]) else f"{bx[t] / L:.5f},{by[t] / W:.5f},NaN"
            fh.write(f"{t + 1}:{';'.join(g)}:{b}\n")

    blocks = []
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        blocks.append([int(p), int(rows[0]), int(rows[-1])])
    with open(os.path.join(a.out, "game.json"), "w") as fh:
        json.dump({"stem": stem, "n_frames": n, "blocks": blocks, "pitch": [L, W],
                   "source": "Metrica Sports sample-data " + name, "degraded": not a.no_degrade}, fh, indent=2)

    gt = build_gt(events, period, frame, n)
    pd.DataFrame({"frame": np.arange(n), "label": gt}).to_csv(os.path.join(a.out, "gt.csv"), index=False)
    inside = np.concatenate([np.arange(s, e + 1) for _, s, e in blocks])
    share = pd.Series(gt[inside]).value_counts(normalize=True).round(3).to_dict()
    print(f"wrote {a.out}: {n} frames, blocks {blocks}, label share {share}, "
          f"ball visible {np.isfinite(bx).mean():.1%}")


if __name__ == "__main__":
    main()
