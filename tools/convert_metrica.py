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


def add_fakes(px, py, bx, by, n_home, period, L, W, rng, n_static=30, n_spikes=300, n_swaps=20):
    """Inject the kinds of errors seen in the challenge data, to test the clean-up modules.

    ball : static false balls (spots / lines) filling missing stretches, and 1-2 frame spikes
    players: a static object at a goal post (extra home slot), and goalkeeper episodes where the
             keeper is tracked in a free slot of the OTHER team
    Returns new arrays (players get one extra column, appended to the home team) and a report.
    """
    T = len(bx)
    bx, by = bx.copy(), by.copy()
    spots = [(11.0, W / 2), (L - 11.0, W / 2), (L / 2, W / 2), (L / 2, 0.0), (0.0, W / 2 + 9.0), (L, W / 2 - 9.0)]
    n_sf = 0
    for _ in range(n_static):
        t0 = int(rng.integers(0, T - 300))
        k = int(rng.uniform(3, 10) * FPS)
        x, y = spots[int(rng.integers(len(spots)))]
        seg = np.arange(t0, t0 + k)
        seg = seg[np.isnan(bx[seg])]
        bx[seg] = x + rng.normal(0, 0.05, len(seg))
        by[seg] = y + rng.normal(0, 0.05, len(seg))
        n_sf += len(seg)
    det = np.flatnonzero(np.isfinite(bx))
    for t in rng.choice(det, size=min(n_spikes, len(det)), replace=False):
        ang, r = rng.uniform(0, 2 * np.pi), rng.uniform(10, 40)
        for u in range(t, min(T, t + int(rng.integers(1, 3)))):
            bx[u] = np.clip(bx[t] + r * np.cos(ang), 0, L)
            by[u] = np.clip(by[t] + r * np.sin(ang), 0, W)
    # static object at a goal post, present in random 60 s windows (about half the match)
    post = np.full(T, np.nan)
    for t0 in range(0, T, 60 * FPS):
        if rng.random() < 0.5:
            post[t0:t0 + 60 * FPS] = 1
    gx, gy = np.where(post == 1, 0.3, np.nan), np.where(post == 1, W / 2 - 3.66, np.nan)
    px = np.concatenate([px[:, :n_home], gx[:, None], px[:, n_home:]], axis=1)
    py = np.concatenate([py[:, :n_home], gy[:, None], py[:, n_home:]], axis=1)
    n_home += 1
    # goalkeeper swaps
    swaps = 0
    first = period == period.min()
    for team_cols, other_cols in ((range(0, n_home - 1), range(n_home, px.shape[1])),
                                  (range(n_home, px.shape[1]), range(0, n_home - 1))):
        cols = list(team_cols)
        mx = np.array([np.nanmean(px[first, c]) if np.isfinite(px[first, c]).any() else np.nan for c in cols])
        dist_goal = np.minimum(np.abs(mx), np.abs(L - mx))
        keeper = cols[int(np.nanargmin(dist_goal))]
        for _ in range(n_swaps // 2):
            t0 = int(rng.integers(0, T - 600))
            k = int(rng.uniform(5, 20) * FPS)
            seg = slice(t0, t0 + k)
            free = [c for c in other_cols if np.isnan(px[seg, c]).all()]
            if not free:
                continue
            c = free[0]
            px[seg, c], py[seg, c] = px[seg, keeper], py[seg, keeper]
            px[seg, keeper] = py[seg, keeper] = np.nan
            swaps += 1
    return px, py, bx, by, n_home, {"static_ball_frames": n_sf, "spikes": n_spikes, "keeper_swaps": swaps,
                                     "post_frames": int(np.isfinite(gx).sum())}


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
    ap.add_argument("--fake", action="store_true",
                    help="also inject false balls, ball spikes, a static object at a goal post and goalkeeper team swaps")
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
    n_home = len(home)
    fake_rep = None
    if a.fake:
        px, py, bx, by, n_home, fake_rep = add_fakes(px, py, bx, by, n_home, period, L, W,
                                                     np.random.default_rng(a.seed + 1))
        names = list(home) + ["FakePost"] + list(away)
    sp = np.hypot(np.gradient(px, axis=0), np.gradient(py, axis=0)) * FPS

    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, stem)
    with open(base + "_atd.xml", "w") as fh:
        fh.write('<?xml version="1.0"?>\n<Root><Teams><Team id="H"><Name>Home</Name></Team>'
                 '<Team id="A"><Name>Away</Name></Team></Teams><Players>')
        for i, nm in enumerate(names):
            tid = "H" if i < n_home else "A"
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
                   "source": "Metrica Sports sample-data " + name, "degraded": not a.no_degrade, "fakes": fake_rep}, fh, indent=2)

    gt = build_gt(events, period, frame, n)
    pd.DataFrame({"frame": np.arange(n), "label": gt}).to_csv(os.path.join(a.out, "gt.csv"), index=False)
    inside = np.concatenate([np.arange(s, e + 1) for _, s, e in blocks])
    share = pd.Series(gt[inside]).value_counts(normalize=True).round(3).to_dict()
    print(f"wrote {a.out}: {n} frames, blocks {blocks}, label share {share}, "
          f"ball visible {np.isfinite(bx).mean():.1%}")


if __name__ == "__main__":
    main()
