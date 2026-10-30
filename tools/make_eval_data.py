"""Build an evaluation game from a Metrica public sample game: INPUT data that looks like the
challenge data, and TRUTH data to score against.

    python tools/make_eval_data.py --src ../sample-data/data/Sample_Game_1 --out external/EVAL-G1
    python tools/make_eval_data.py --src ../sample-data/data/Sample_Game_2 --out external/EVAL-G2 --seed 1

Folder written
    game.json                    sizes, halves, settings, and what was injected (counts)
    input/   (use these, like the challenge data)
        <stem>_atd.xml / _atd.txt          players + ball with missing data and false detections
        <stem>_smart_tagging.xml           tags made by the tagger from the INPUT data
    truth/   (only for scoring)
        <stem>_atd.xml / _atd.txt          true players + ball (Metrica tracking)
        ball.csv                           frame, x_m, y_m of the true ball (empty when not tracked)
        gt.csv                             frame,label  true possession H/A/D (from Metrica events)
        <stem>_smart_tagging.xml           tags made by the SAME tagger from the TRUE data
        errors.npz                         what was changed, frame by frame (for tools/eval_cleaning.py)

INPUT ball   : missing on --ball-missing of ALL frames (default 0.41: the true gaps + extra bursts
               during play), and --ball-false (default 0.10) of the remaining detections replaced by
               false detections (static spots / lines, spikes, a player's feet)
INPUT players: far players dropped like a tactical camera, then --player-false (default 0.10) of the
               remaining detections turned into errors: team swaps (a player tracked in a slot of
               the other team, goalkeepers more often) and false static objects (posts, behind the
               goal, touch line)
Tags         : one rule-based tagger (tag_rules below) applied twice -- to the truth (true ball,
               true possession, exact restart times) and to the input (input ball, a naive
               possession guess, restart times with the offsets seen in the real Smart Tagging).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from convert_metrica import build_gt, read_tracking  # noqa: E402

FPS = 25
ERR_NONE, ERR_REMOVED, ERR_FALSE, ERR_SWAP_IN, ERR_SWAP_OUT = 0, 1, 2, 3, 4


# ------------------------------------------------------------------ helpers
def runs(mask):
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))     # [start, stop)


def write_atd(base, names_home, names_away, px, py, bx, by, L, W, team_names=("Home", "Away")):
    nh = len(names_home)
    with open(base + "_atd.xml", "w") as fh:
        fh.write('<?xml version="1.0"?>\n<Root><Teams>'
                 f'<Team id="H"><Name>{team_names[0]}</Name></Team><Team id="A"><Name>{team_names[1]}</Name></Team>'
                 '</Teams><Players>')
        for i, nm in enumerate(list(names_home) + list(names_away)):
            tid = "H" if i < nh else "A"
            fh.write(f'<Player id="{tid}{i}_{nm}" teamId="{tid}"><Name>Track {i + 1}</Name></Player>')
        fh.write("</Players></Root>\n")
    hw = 5
    sp = np.hypot(px[hw:] - px[:-hw], py[hw:] - py[:-hw]) * FPS / hw
    sp = np.vstack([sp, np.repeat(sp[-1:], hw, axis=0)])
    with open(base + "_atd.txt", "w") as fh:
        for t in range(len(bx)):
            g = ["NaN,NaN,NaN" if not np.isfinite(px[t, s]) else
                 f"{px[t, s] / L:.5f},{py[t, s] / W:.5f},{np.nan_to_num(sp[t, s]):.2f}" for s in range(px.shape[1])]
            b = "NaN,NaN,NaN" if not np.isfinite(bx[t]) else f"{bx[t] / L:.5f},{by[t] / W:.5f},NaN"
            fh.write(f"{t + 1}:{';'.join(g)}:{b}\n")


def write_tags(path, tags):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("<file><ALL_INSTANCES>\n")
        for i, g in enumerate(sorted(tags, key=lambda g: g["start"])):
            labels = "".join(f"<label><group>{k}</group><text>{v}</text></label>" for k, v in g["labels"])
            fh.write(f"<instance><ID>{i + 1}</ID><start>{g['start'] / FPS:.2f}</start><end>{g['end'] / FPS:.2f}</end>"
                     f"<code>{g['code']}</code>{labels}</instance>\n")
        fh.write("</ALL_INSTANCES></file>\n")


# ------------------------------------------------------------------ input ball
def make_input_ball(bx, by, px, py, target_missing, false_rate, L, W, rng):
    T = len(bx)
    ibx, iby = bx.copy(), by.copy()
    state = np.zeros(T, np.int8)                     # per frame: ERR_NONE / ERR_REMOVED / ERR_FALSE
    # extra missing bursts during play until the target share of ALL frames is missing
    det = np.flatnonzero(np.isfinite(ibx))
    while np.isnan(ibx).mean() < target_missing and len(det):
        t = int(rng.choice(det))
        k = int(rng.uniform(0.2, 3.0) * FPS)
        seg = np.arange(t, min(T, t + k))
        seg = seg[np.isfinite(ibx[seg])]
        ibx[seg] = iby[seg] = np.nan
        state[seg] = ERR_REMOVED
        det = np.flatnonzero(np.isfinite(ibx))
    # false detections replacing false_rate of the remaining detections
    n_target = int(false_rate * np.isfinite(ibx).sum())
    spots = [(11.0, W / 2), (L - 11.0, W / 2), (L / 2, W / 2), (L / 2, 0.0), (L / 2, W), (0.0, W / 2 + 9.0),
             (L, W / 2 - 9.0), (16.5, W / 2 - 15), (L - 16.5, W / 2 + 15)]
    kinds = {"static": 0, "spike": 0, "feet": 0}
    done = 0
    while done < n_target:
        det = np.flatnonzero(np.isfinite(ibx) & (state == ERR_NONE))
        if not len(det):
            break
        t = int(rng.choice(det))
        kind = rng.choice(["static", "spike", "feet"], p=[0.4, 0.3, 0.3])
        if kind == "static":
            k = int(rng.uniform(0.5, 5.0) * FPS)
            x, y = spots[int(rng.integers(len(spots)))]
            seg = np.arange(t, min(T, t + k))
            xs, ys = x + rng.normal(0, 0.05, len(seg)), y + rng.normal(0, 0.05, len(seg))
        elif kind == "spike":
            seg = np.arange(t, min(T, t + int(rng.integers(1, 4))))
            ang, r = rng.uniform(0, 2 * np.pi), rng.uniform(5, 40)
            xs = np.clip(ibx[t] + r * np.cos(ang), 0, L) + np.zeros(len(seg))
            ys = np.clip(iby[t] + r * np.sin(ang), 0, W) + np.zeros(len(seg))
        else:                                        # on the feet of a player who is not on the ball
            seg = np.arange(t, min(T, t + int(rng.uniform(0.2, 2.0) * FPS)))
            d = np.hypot(px[t] - bx[t], py[t] - by[t])
            cand = np.flatnonzero(np.isfinite(d) & (d > 5))
            if not len(cand):
                continue
            s = int(rng.choice(cand))
            xs, ys = px[seg, s], py[seg, s]
        seg_ok = np.isfinite(ibx[seg]) & (state[seg] == ERR_NONE) & np.isfinite(xs) & np.isfinite(ys)
        seg, xs, ys = seg[seg_ok], np.asarray(xs)[seg_ok], np.asarray(ys)[seg_ok]
        ibx[seg], iby[seg] = xs, ys
        state[seg] = ERR_FALSE
        kinds[kind] += len(seg)
        done += len(seg)
    return ibx, iby, state, kinds


# ------------------------------------------------------------------ input players
def make_input_players(px, py, n_home, bx, by, false_rate, L, W, rng, n_extra=4):
    """Returns input px, py with n_extra free slots added to EACH team, team array, error state,
    and the true slot each input detection comes from (-1 for false objects)."""
    T, S = px.shape
    # camera coverage: drop players far from where the camera looks (smoothed true ball)
    aimx = pd.Series(bx).interpolate(limit_direction="both").rolling(50, min_periods=1, center=True).mean().to_numpy()
    aimy = pd.Series(by).interpolate(limit_direction="both").rolling(50, min_periods=1, center=True).mean().to_numpy()
    aimx, aimy = np.nan_to_num(aimx, nan=L / 2), np.nan_to_num(aimy, nan=W / 2)
    far = np.hypot(px - aimx[:, None], py - aimy[:, None]) > 40
    dropped = far & (rng.random(far.shape) < 0.5)
    qx, qy = px.copy(), py.copy()
    qx[dropped] = qy[dropped] = np.nan

    # layout: home slots, home extras, away slots, away extras
    na = S - n_home
    E = n_extra
    ix = np.full((T, S + 2 * E), np.nan)
    iy = np.full((T, S + 2 * E), np.nan)
    src = np.full((T, S + 2 * E), -1, np.int16)
    home_cols = list(range(n_home)) + list(range(n_home, n_home + E))
    away_cols = list(range(n_home + E, n_home + E + na)) + list(range(n_home + E + na, S + 2 * E))
    for j in range(n_home):
        ix[:, j], iy[:, j], src[:, j] = qx[:, j], qy[:, j], j
    for j in range(na):
        c = n_home + E + j
        ix[:, c], iy[:, c], src[:, c] = qx[:, n_home + j], qy[:, n_home + j], n_home + j
    src[np.isnan(ix)] = -1
    team = np.array([0] * (n_home + E) + [1] * (na + E), np.int8)
    state = np.zeros(ix.shape, np.int8)
    for j in range(S):                               # players hidden by the camera coverage
        c = j if j < n_home else j + E
        state[dropped[:, j] & np.isfinite(px[:, j]), c] = ERR_REMOVED

    # goalkeepers: the slot of each team closest to a goal line on average in the first minutes
    def keeper(cols):
        m = [np.nanmean(np.minimum(np.abs(px[:3000, c]), np.abs(L - px[:3000, c])))
             if np.isfinite(px[:3000, c]).any() else np.inf for c in cols]
        return cols[int(np.argmin(m))]
    keepers = [keeper(list(range(n_home))), keeper(list(range(n_home, S)))]

    n_target = int(false_rate * np.isfinite(ix).sum())
    done = 0
    kinds = {"swap": 0, "keeper_swap": 0, "static_object": 0}
    objects = [(-0.3, W / 2 - 3.66), (-0.3, W / 2 + 3.66), (L + 0.3, W / 2 - 3.66), (L + 0.3, W / 2 + 3.66),
               (-2.0, W / 2), (L + 2.0, W / 2), (L / 2, -1.5), (L / 3, W + 1.5)]
    tries = 0
    while done < n_target and tries < 100000:
        tries += 1
        t0 = int(rng.integers(0, T - 30 * FPS))
        kind = rng.choice(["swap", "keeper_swap", "static_object"], p=[0.5, 0.3, 0.2])
        if kind == "static_object":
            k = int(rng.uniform(5, 30) * FPS)
            seg = slice(t0, t0 + k)
            side_cols = home_cols if rng.random() < 0.5 else away_cols
            free = [c for c in side_cols if np.isnan(ix[seg, c]).all()]
            if not free:
                continue
            c = free[0]
            x, y = objects[int(rng.integers(len(objects)))]
            ix[seg, c], iy[seg, c] = x + rng.normal(0, 0.05, k), y + rng.normal(0, 0.05, k)
            state[seg, c], src[seg, c] = ERR_FALSE, -1
            kinds[kind] += k
            done += k
        else:
            k = int(rng.uniform(3, 20) * FPS)
            seg = slice(t0, t0 + k)
            if kind == "keeper_swap":
                true_slot = keepers[int(rng.integers(2))]
            else:
                true_slot = int(rng.integers(S))
            # input column that currently holds that true slot
            col = true_slot if true_slot < n_home else true_slot + E
            if not np.isfinite(ix[seg, col]).all() or (state[seg, col] != ERR_NONE).any():
                continue
            other = away_cols if col in home_cols else home_cols
            free = [c for c in other if np.isnan(ix[seg, c]).all()]
            if not free:
                continue
            c = free[0]
            ix[seg, c], iy[seg, c], src[seg, c], state[seg, c] = ix[seg, col], iy[seg, col], true_slot, ERR_SWAP_IN
            ix[seg, col] = iy[seg, col] = np.nan
            state[seg, col], src[seg, col] = ERR_SWAP_OUT, -1
            kinds[kind] += k
            done += k
    return ix, iy, team, state, src, n_home + E, kinds, float(np.mean(np.isfinite(ix[:, :]).sum(1)))


# ------------------------------------------------------------------ tagger (the same rules for truth and input)
SET_TYPES = {"THROW IN": ("Throw-in", False), "CORNER KICK": ("Corner Kick", True), "GOAL KICK": ("Goal Kick", True),
             "FREE KICK": ("Free Kick", True), "KICK OFF": ("Kick Off", False)}


def restarts_from_events(events, frame, period):
    """[(frame_row, type_text, team 'H'/'A', period)] for set pieces, and goal rows."""
    idx = {f: i for i, f in enumerate(frame)}
    out, goals = [], []
    first_ko = set()
    for _, r in events.sort_values("Start Frame").iterrows():
        t = str(r["Type"]).upper()
        st = str(r["Subtype"]).upper() if pd.notna(r["Subtype"]) else ""
        row = idx.get(int(r["Start Frame"]))
        if row is None:
            continue
        side = "H" if str(r["Team"]).lower().startswith("home") else "A"
        if t == "SET PIECE":
            for key, (txt, _) in SET_TYPES.items():
                if key in st:
                    if key == "KICK OFF":
                        p = int(r["Period"])
                        txt = "Kick Off (Start)" if p not in first_ko else "Kick Off (After Goal)"
                        first_ko.add(p)
                    out.append((row, txt, side, int(r["Period"])))
                    break
        elif t == "SHOT" and "GOAL" in st:
            goals.append((row, side, int(r["Period"])))
    return out, goals


def tag_rules(bx, by, poss, restarts, goals, period, attack_home, L, W, jitter=None, rng=None):
    """Tags in the Smart Tagging style from a ball track and a per-frame possession (H/A/D/'').

    jitter=None -> truth timing. jitter=dict -> input timing (restart tags start before the stoppage
    is visible and end ~3 s after the kick; a few tags are missed).
    """
    T = len(bx)
    tags = []
    names = {"H": "Home", "A": "Away"}
    half = lambda p: "1st Half" if p == 1 else "2nd Half"  # noqa: E731
    # ball, gaps up to 1 s bridged
    x = pd.Series(bx).ffill(limit=FPS).to_numpy()
    y = pd.Series(by).ffill(limit=FPS).to_numpy()

    def add(code, s, e, team, p, extra=()):
        if e - s < FPS // 2:
            return
        if jitter and rng.random() < jitter.get("miss", 0.0):
            return
        tags.append({"start": int(s), "end": int(e), "code": code,
                     "labels": [("Team", names.get(team, "N/A"))] + list(extra) + [("Half", half(p))]})

    # phases inside possession runs
    for team in ("H", "A"):
        opp = "A" if team == "H" else "H"
        mine = poss == team
        for s, e in runs(mine):
            if e - s < 2 * FPS:
                continue
            p = int(period[s])
            sign = attack_home[s] if team == "H" else -attack_home[s]
            xn = (x[s:e] if sign > 0 else L - x[s:e])          # 0 = own goal line
            zone = np.full(e - s, -1)
            ok = np.isfinite(xn)
            zone[ok & (xn < L / 3)] = 0
            zone[ok & (xn >= L / 3) & (xn < 2 * L / 3)] = 1
            zone[ok & (xn >= 2 * L / 3)] = 2
            box = ok & (xn > L - 16.5) & (np.abs(np.nan_to_num(y[s:e]) - W / 2) < 20.16)
            for z, code, dcode, mins in ((0, "BUILD UP", "DEFENDING IN ATTACKING THIRD", 6),
                                         (1, "PROGRESSION", "DEFENDING IN MIDDLE THIRD", 6),
                                         (2, "BALL IN FINAL THIRD", "DEFENDING IN DEFENSIVE THIRD", 4)):
                for a, b in runs(zone == z):
                    if b - a >= mins * FPS:
                        add(code, s + a, s + b, team, p)
                        add(dcode, s + a, s + b, opp, p)
            for a, b in runs(box):
                if b - a >= 2 * FPS:
                    add("BALL IN THE BOX", s + a, s + b, team, p)
            # transitions: possession won directly from the other team (no stoppage between)
            if s > 0 and poss[s - 1] == opp and e - s >= 3 * FPS:
                add("ATTACKING TRANSITION", s, min(e, s + 4 * FPS), team, p)
                add("DEFENSIVE TRANSITION", s, min(e, s + 4 * FPS), opp, p)
    # set pieces
    for row, txt, side, p in restarts:
        team_tag = "N/A" if (txt == "Throw-in" or txt.startswith("Kick Off")) else side   # as in the real tags
        if jitter:
            s = row - int(rng.uniform(1.0, 4.0) * FPS)
            e = row + int(rng.uniform(2.0, 4.0) * FPS)
        else:
            dead = np.flatnonzero(poss[:row] != "D")
            s = (dead[-1] + 1) if len(dead) else max(0, row - 3 * FPS)
            s = max(s, row - 30 * FPS)
            e = row + 1
        add("SET PIECES", max(0, s), min(T, e), team_tag, p, extra=[("Type", txt)])
    for row, side, p in goals:
        s = row + (int(rng.uniform(-1, 1) * FPS) if jitter else 0)
        add("GOALS", max(0, s), min(T, s + 5 * FPS), side, p)
    return tags


def naive_possession(bx, by, px, py, team, L, W):
    """A deliberately simple guess from the INPUT data: team of the nearest detection to the ball
    (within 3 m), carried for up to 5 s; '' when unknown."""
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    k = d.argmin(axis=1)
    near = d[np.arange(len(bx)), k] < 3.0
    lab = np.where(near, np.where(team[k] == 0, "H", "A"), "")
    s = pd.Series(lab).replace("", np.nan).ffill(limit=5 * FPS).fillna("")
    return s.to_numpy().astype("<U1")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="e.g. ../sample-data/data/Sample_Game_1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem")
    ap.add_argument("--pitch", type=float, nargs=2, default=(105.0, 68.0))
    ap.add_argument("--ball-missing", type=float, default=0.41, help="share of ALL frames without a ball")
    ap.add_argument("--ball-false", type=float, default=0.10, help="share of ball detections that are false")
    ap.add_argument("--player-false", type=float, default=0.10, help="share of player detections that are errors")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    L, W = a.pitch
    name = os.path.basename(os.path.normpath(a.src))
    stem = a.stem or "EVAL-" + name.replace("Sample_Game_", "G")

    f_home = [f for f in os.listdir(a.src) if f.endswith("Home_Team.csv")][0]
    f_away = [f for f in os.listdir(a.src) if f.endswith("Away_Team.csv")][0]
    f_ev = [f for f in os.listdir(a.src) if f.endswith("EventsData.csv")][0]
    period, frame, home, ball = read_tracking(os.path.join(a.src, f_home))
    _, frame2, away, _ = read_tracking(os.path.join(a.src, f_away))
    assert (frame == frame2).all()
    events = pd.read_csv(os.path.join(a.src, f_ev))
    T = len(frame)
    px = np.stack([v[:, 0] for v in list(home.values()) + list(away.values())], 1) * L
    py = np.stack([v[:, 1] for v in list(home.values()) + list(away.values())], 1) * W
    bx, by = ball[:, 0] * L, ball[:, 1] * W
    n_home = len(home)
    blocks = [[int(p), int(np.flatnonzero(period == p)[0]), int(np.flatnonzero(period == p)[-1])]
              for p in np.unique(period)]

    # truth possession and attack direction (home attacks towards +x if its players start on the left)
    gt = build_gt(events, period, frame, T)
    attack_home = np.ones(T)
    for p, s, e in blocks:
        seg = slice(s, min(e, s + 60 * FPS))
        attack_home[s:e + 1] = 1 if np.nanmean(px[seg, :n_home]) < np.nanmean(px[seg, n_home:]) else -1

    os.makedirs(os.path.join(a.out, "input"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "truth"), exist_ok=True)
    ti, tt = os.path.join(a.out, "input", stem), os.path.join(a.out, "truth", stem)

    # ---- truth
    write_atd(tt, list(home), list(away), px, py, bx, by, L, W)
    pd.DataFrame({"frame": np.arange(T), "x_m": np.round(bx, 3), "y_m": np.round(by, 3)}).to_csv(
        os.path.join(a.out, "truth", "ball.csv"), index=False)
    pd.DataFrame({"frame": np.arange(T), "label": gt}).to_csv(os.path.join(a.out, "truth", "gt.csv"), index=False)
    restarts, goals = restarts_from_events(events, frame, period)
    tags_true = tag_rules(bx, by, gt, restarts, goals, period, attack_home, L, W)
    write_tags(tt + "_smart_tagging.xml", tags_true)

    # ---- input
    ix, iy, team, pstate, psrc, n_home_in, pkinds, avg_tracked = make_input_players(
        px, py, n_home, bx, by, a.player_false, L, W, rng)
    ibx, iby, bstate, bkinds = make_input_ball(bx, by, px, py, a.ball_missing, a.ball_false, L, W, rng)
    names_in = [f"H{i}" for i in range(n_home_in)] + [f"A{i}" for i in range(ix.shape[1] - n_home_in)]
    write_atd(ti, names_in[:n_home_in], names_in[n_home_in:], ix, iy, ibx, iby, L, W)
    poss_in = naive_possession(ibx, iby, ix, iy, team, L, W)
    tags_in = tag_rules(ibx, iby, poss_in, restarts, goals, period, attack_home, L, W,
                        jitter={"miss": 0.05}, rng=rng)
    write_tags(ti + "_smart_tagging.xml", tags_in)
    np.savez_compressed(os.path.join(a.out, "truth", "errors.npz"),
                        ball_state=bstate, player_state=pstate, player_src=psrc, input_team=team,
                        true_team=np.array([0] * n_home + [1] * (px.shape[1] - n_home), np.int8),
                        true_bx=bx, true_by=by, input_bx=ibx, input_by=iby)

    inplay = np.isin(gt, ["H", "A"])
    stats = {"frames": T, "ball_missing_all": round(float(np.isnan(ibx).mean()), 3),
             "ball_missing_in_play": round(float(np.isnan(ibx[inplay]).mean()), 3),
             "ball_true_missing_all": round(float(np.isnan(bx).mean()), 3),
             "ball_false_share_of_detections": round(float((bstate == ERR_FALSE).sum() / max(1, np.isfinite(ibx).sum())), 3),
             "ball_false_kinds": bkinds, "player_error_kinds": pkinds,
             "player_error_share": round(float((np.isin(pstate, [ERR_FALSE, ERR_SWAP_IN])).sum()
                                               / max(1, np.isfinite(ix).sum())), 3),
             "players_tracked_mean": round(avg_tracked, 1),
             "tags_truth": len(tags_true), "tags_input": len(tags_in)}
    cfg = {"stem": stem, "n_frames": T, "blocks": blocks, "pitch": [L, W], "input_dir": "input",
           "gt": "truth/gt.csv", "source": "Metrica Sports sample-data " + name + " (please acknowledge)",
           "settings": {"ball_missing": a.ball_missing, "ball_false": a.ball_false,
                        "player_false": a.player_false, "seed": a.seed}, "stats": stats}
    with open(os.path.join(a.out, "game.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
