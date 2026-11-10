"""Top-down replay of D decisions: successes and failures side by side, to rethink the METHOD.

Run the pipeline with --dead-debug first, then:

    # evaluation game (has a truth): failures and successes are picked automatically
    python -m possession.v2.pipeline_v2 --repo ../more-than-a-hack-2027 --external external/EVAL-G1 \
        --out results/eval_g1.csv --dead-debug results/eval_g1_dead
    python tools/dead_view.py --repo ../more-than-a-hack-2027 --external external/EVAL-G1 \
        --frames results/eval_g1_dead_frames.csv --out results/deadview_g1

    # challenge game (no truth): short D, long D and play are picked from our own output
    python tools/dead_view.py --repo ../more-than-a-hack-2027 --frames results/dead_frames.csv --out results/deadview_real

Writes, per category, <out>_<category>.mp4 (clips one after another, each with a title card) and
<out>_<category>.png (one still per clip), plus <out>_index.txt (which clips, where, why).

Categories
    with a truth : false_d   we say D, the truth says play     (failure)
                   missed_d  we say play, the truth says D      (failure)
                   true_d    both D                             (success)
                   true_play both play, same team              (success)
    without      : short_d (< 3 s), long_d (>= 8 s), play

Each frame shows the pitch from above
    home blue, away red (grey = removed by the clean-up); hollow circles = TRUE players (truth only)
    ball yellow (trusted) / orange (interpolated) / magenta (rejected); white ring = TRUE ball
    arrows = player velocity (1 s)
    text: rules that fired for the pseudo label, the features that pushed P(dead), mean speed ...
    strips (+-10 s): truth, our label, pseudo label, P(dead)
Rules: on the challenge game this is for looking at what happens, never for labelling it.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from possession import data, preprocess  # noqa: E402
from possession.v2.ball_quality import INTERP, REJECTED, TRUSTED, assess_ball  # noqa: E402
from possession.v2.clean_players import clean_players  # noqa: E402
from possession.v2.config_v2 import Params2  # noqa: E402

FPS = 25
BLUE, RED, GREY = (230, 90, 20), (40, 40, 225), (150, 150, 150)
YELLOW, ORANGE, MAGENTA, WHITE = (0, 230, 255), (0, 140, 255), (200, 0, 200), (255, 255, 255)
GRASS, LINE = (60, 110, 50), (235, 235, 235)
LAB = {"H": BLUE, "A": RED, "D": (150, 150, 150), "X": (0, 0, 0), "": (40, 40, 40)}


def tstr(f):
    s = f / FPS
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def runs(mask):
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


# ------------------------------------------------------------------ loading
def load(args, P):
    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    import games  # noqa: E402
    truth = None
    if args.external:
        with open(os.path.join(args.external, "game.json")) as fh:
            cfg = json.load(fh)
        n, blocks = int(cfg["n_frames"]), [tuple(b) for b in cfg["blocks"]]
        P.pitch_length, P.pitch_width = cfg["pitch"]
        stem = os.path.join(args.external, cfg.get("input_dir", ""), cfg["stem"])
        tr = data.load_tracking(stem + "_atd.xml", stem + "_atd.txt", n, tuple(cfg["pitch"]))
        tdir = os.path.join(args.external, "truth")
        if os.path.isdir(tdir):
            ts = os.path.join(tdir, cfg["stem"])
            ttr = data.load_tracking(ts + "_atd.xml", ts + "_atd.txt", n, tuple(cfg["pitch"]))
            truth = {"px": ttr.px, "py": ttr.py, "bx": ttr.bx, "by": ttr.by}
    else:
        data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(repo, "data")
        paths = games.asset_paths(args.game, data_dir)
        n = games.source_frames("tactical", args.game)
        blocks = data.graded_blocks(repo, args.game)
        if args.game in games.GAMES:
            P.pitch_length, P.pitch_width = games.GAMES[args.game]["pitch"]
        os.makedirs(args.cache_dir, exist_ok=True)
        tr = data.load_tracking(paths["atd_meta"], paths["atd_raw"], n, (P.pitch_length, P.pitch_width),
                                cache=os.path.join(args.cache_dir, f"{args.game}_tracking.npz"))
    return tr, truth, blocks, n


# ------------------------------------------------------------------ clip selection
def pick_clips(fr, n_frames, per_cat, rng, min_s=1.0):
    lab = np.full(n_frames, "", dtype="<U1")
    lab[fr["frame"].values] = fr["label"].astype(str).values
    cats = {}
    if "gt" in fr:
        gt = np.full(n_frames, "", dtype="<U1")
        gt[fr["frame"].values] = fr["gt"].astype(str).values
        play_gt = np.isin(gt, ["H", "A"])
        masks = {"false_d": (lab == "D") & play_gt, "missed_d": np.isin(lab, ["H", "A"]) & (gt == "D"),
                 "true_d": (lab == "D") & (gt == "D"), "true_play": play_gt & (lab == gt)}
        for name, m in masks.items():
            rs = [(s, e) for s, e in runs(m) if e - s >= min_s * FPS]
            if name.startswith(("false", "missed")):
                rs = sorted(rs, key=lambda r: -(r[1] - r[0]))[:per_cat]           # biggest failures
            else:
                rs = [rs[i] for i in sorted(rng.choice(len(rs), size=min(per_cat, len(rs)), replace=False))] if rs else []
            cats[name] = rs
    else:
        d = [(s, e) for s, e in runs(lab == "D")]
        short = [r for r in d if r[1] - r[0] < 3 * FPS]
        long_ = [r for r in d if r[1] - r[0] >= 8 * FPS]
        play = [r for r in runs(np.isin(lab, ["H", "A"])) if r[1] - r[0] >= 5 * FPS]
        for name, rs in (("short_d", short), ("long_d", long_), ("play", play)):
            cats[name] = [rs[i] for i in sorted(rng.choice(len(rs), size=min(per_cat, len(rs)), replace=False))] if rs else []
    return cats, lab


# ------------------------------------------------------------------ drawing
class Pitch:
    def __init__(self, L, W, width=900):
        self.L, self.W = L, W
        self.m = 25
        self.s = (width - 2 * self.m) / L
        self.w, self.h = width, int(W * self.s + 2 * self.m)
        img = np.full((self.h, self.w, 3), GRASS, np.uint8)
        P = self.P
        cv2.rectangle(img, P(0, 0), P(L, W), LINE, 2)
        cv2.line(img, P(L / 2, 0), P(L / 2, W), LINE, 2)
        cv2.circle(img, P(L / 2, W / 2), int(9.15 * self.s), LINE, 2)
        for x0, sg in ((0, 1), (L, -1)):
            cv2.rectangle(img, P(x0, W / 2 - 20.16), P(x0 + sg * 16.5, W / 2 + 20.16), LINE, 2)
            cv2.rectangle(img, P(x0, W / 2 - 9.16), P(x0 + sg * 5.5, W / 2 + 9.16), LINE, 2)
        self.base = img

    def P(self, x, y):
        return int(self.m + x * self.s), int(self.m + y * self.s)


def strip(img, y0, label_arr, t, span, x0, x1, title, colour_of):
    cv2.putText(img, title, (8, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    xs = np.linspace(x0, x1, 2 * span + 2).astype(int)
    for k, f in enumerate(range(t - span, t + span + 1)):
        if 0 <= f < len(label_arr):
            cv2.rectangle(img, (xs[k], y0), (xs[k + 1], y0 + 16), colour_of(label_arr[f]), -1)
    cv2.line(img, (xs[span], y0 - 2), (xs[span], y0 + 18), WHITE, 2)


def render(t, pitch, st, row, extra_title=""):
    img = pitch.base.copy()
    P = pitch.P
    px, py, team, removed = st["px"], st["py"], st["team"], st["removed"]
    vx, vy = st["vx"], st["vy"]
    if st["truth"] is not None:
        tp = st["truth"]
        for s in range(tp["px"].shape[1]):
            if np.isfinite(tp["px"][t, s]):
                cv2.circle(img, P(tp["px"][t, s], tp["py"][t, s]), 9, (210, 210, 210), 1)
        if np.isfinite(tp["bx"][t]):
            cv2.circle(img, P(tp["bx"][t], tp["by"][t]), 11, WHITE, 2)
    for s in range(px.shape[1]):
        if not np.isfinite(st["raw_px"][t, s]):
            continue
        if removed[t, s]:
            cv2.circle(img, P(st["raw_px"][t, s], st["raw_py"][t, s]), 6, GREY, -1)
            continue
        c = BLUE if team[t, s] == 0 else RED
        p0 = P(px[t, s], py[t, s])
        cv2.circle(img, p0, 6, c, -1)
        if np.isfinite(vx[t, s]):
            cv2.arrowedLine(img, p0, P(px[t, s] + vx[t, s], py[t, s] + vy[t, s]), c, 1, tipLength=0.3)
    stt = st["ball_status"][t]
    if stt == REJECTED and np.isfinite(st["raw_bx"][t]):
        cv2.circle(img, P(st["raw_bx"][t], st["raw_by"][t]), 6, MAGENTA, -1)
    elif np.isfinite(st["bx"][t]):
        cv2.circle(img, P(st["bx"][t], st["by"][t]), 6, ORANGE if stt == INTERP else YELLOW, -1)
        cv2.circle(img, P(st["bx"][t], st["by"][t]), 6, (0, 0, 0), 1)

    # panel
    H = img.shape[0]
    panel = np.full((H + 150, 420, 3), 25, np.uint8)
    lines = [extra_title, f"{tstr(t)}  frame {t}",
             f"label {st['lab'][t] or '-'}   truth {st['gt'][t] if st['gt'] is not None else '-'}",
             f"P(dead) {row.get('p_dead', np.nan):.2f}   pseudo {row.get('pseudo_dead', '')} {row.get('pseudo_rule', '') or ''}",
             f"mean speed {row.get('speed_mean', np.nan)}  walking {row.get('frac_walking', np.nan)}",
             f"ball seen {row.get('ball_seen', np.nan)}  since {row.get('since_seen', np.nan)} s  "
             f"speed {row.get('ball_speed', np.nan)}",
             f"last ball out {row.get('last_ball_out', np.nan)}  to set piece {row.get('to_set_piece_s', np.nan)} s",
             "pushed by:"] + [f"  {row.get(f'top{k}', '')}" for k in range(1, 6)]
    for i, s in enumerate(lines):
        cv2.putText(panel, str(s)[:56], (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    YELLOW if i == 0 else WHITE, 1, cv2.LINE_AA)
    canvas = np.zeros((H + 150, pitch.w + 420, 3), np.uint8)
    canvas[:H, :pitch.w] = img
    canvas[:, pitch.w:] = panel
    span = 10 * FPS
    x0, x1 = 90, pitch.w - 10
    y = H + 10
    if st["gt"] is not None:
        strip(canvas, y, st["gt"], t, span, x0, x1, "truth", lambda v: LAB.get(v, (40, 40, 40)))
    strip(canvas, y + 26, st["lab"], t, span, x0, x1, "ours", lambda v: LAB.get(v, (40, 40, 40)))
    strip(canvas, y + 52, st["pseudo"], t, span, x0, x1, "pseudo",
          lambda v: {1: (150, 150, 150), 0: (60, 160, 60)}.get(int(v), (40, 40, 40)))
    strip(canvas, y + 78, st["pdead"], t, span, x0, x1, "P(dead)",
          lambda v: tuple(int(c) for c in (np.array([60, 160, 60]) * (1 - v) + np.array([150, 150, 150]) * v))
          if np.isfinite(v) else (40, 40, 40))
    cv2.putText(canvas, "-10s", (x0, y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)
    cv2.putText(canvas, "+10s", (x1 - 35, y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)
    return canvas


def title_card(shape, text1, text2):
    img = np.zeros(shape, np.uint8)
    cv2.putText(img, text1, (40, shape[0] // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, text2, (40, shape[0] // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ main
def main(argv=None):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--frames", required=True, help="<prefix>_frames.csv written by pipeline_v2 --dead-debug")
    ap.add_argument("--external", help="evaluation game folder (make_eval_data.py); omit for the challenge game")
    ap.add_argument("--out", required=True, help="output prefix, e.g. results/deadview_g1")
    ap.add_argument("--per-category", type=int, default=5)
    ap.add_argument("--pad-s", type=float, default=3.0, help="seconds shown before and after each stretch")
    ap.add_argument("--max-s", type=float, default=20.0, help="longest clip")
    ap.add_argument("--speed", type=int, default=1, help="2 = keep every 2nd frame (faster to watch)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--game", default="BAR-ATH")
    ap.add_argument("--data-dir")
    ap.add_argument("--cache-dir", default="cache")
    a = ap.parse_args(argv)

    P = Params2()
    tr, truth, blocks, n = load(a, P)
    attack = preprocess.attack_sign(tr, blocks, P.fps)
    cpx, cpy, team_mat, _ = clean_players(tr, attack, P)
    bq, _ = assess_ball(tr.bx, tr.by, cpx, cpy, P)
    hw = P.frames(P.velocity_half_window_s)
    vx, vy = preprocess.velocity(cpx, hw, P.fps), preprocess.velocity(cpy, hw, P.fps)

    fr = pd.read_csv(a.frames)
    rng = np.random.default_rng(a.seed)
    cats, lab = pick_clips(fr, n, a.per_category, rng)
    rows = {int(f): r for f, r in zip(fr["frame"].values, fr.to_dict("records"))}
    gt = None
    if "gt" in fr:
        gt = np.full(n, "", dtype="<U1")
        gt[fr["frame"].values] = fr["gt"].astype(str).values
    pseudo = np.full(n, -1)
    pseudo[fr["frame"].values] = fr["pseudo_dead"].values
    pdead = np.full(n, np.nan)
    pdead[fr["frame"].values] = fr["p_dead"].values
    st = {"px": cpx, "py": cpy, "raw_px": tr.px, "raw_py": tr.py, "team": team_mat,
          "removed": np.isfinite(tr.px) & np.isnan(cpx), "vx": vx, "vy": vy,
          "bx": bq["bx"], "by": bq["by"], "raw_bx": tr.bx, "raw_by": tr.by, "ball_status": bq["status"],
          "truth": truth, "lab": lab, "gt": gt, "pseudo": pseudo, "pdead": pdead}

    pitch = Pitch(P.pitch_length, P.pitch_width)
    index = [f"clips per category (video time). pad {a.pad_s} s, max {a.max_s} s"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    for cat, clips in cats.items():
        if not clips:
            continue
        sample = render(clips[0][0], pitch, st, rows.get(int(clips[0][0]), {}), cat)
        shape = sample.shape
        vw = cv2.VideoWriter(f"{a.out}_{cat}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (shape[1], shape[0]))
        stills = []
        index.append(f"\n[{cat}] {len(clips)} clips -> {a.out}_{cat}.mp4")
        for i, (s, e) in enumerate(clips):
            mid = (s + e) // 2
            r = rows.get(int(mid), {})
            index.append(f"  {i + 1}. {tstr(s)}-{tstr(e)}  ({(e - s) / FPS:.1f} s)  pseudo {r.get('pseudo_rule', '')}  "
                         f"P(dead) {r.get('p_dead', '')}  top {r.get('top1', '')}")
            card = title_card(shape, f"{cat}  #{i + 1}", f"{tstr(s)} - {tstr(e)}  ({(e - s) / FPS:.1f} s)")
            for _ in range(FPS):
                vw.write(card)
            a0 = max(0, int(s - a.pad_s * FPS))
            a1 = min(n, int(min(e + a.pad_s * FPS, a0 + a.max_s * FPS)))
            for t in range(a0, a1, a.speed):
                vw.write(render(t, pitch, st, rows.get(t, {}), f"{cat} #{i + 1}"))
            stills.append(render(mid, pitch, st, r, f"{cat} #{i + 1}  (middle)"))
        vw.release()
        sheet = np.vstack([cv2.resize(x, (x.shape[1] // 2, x.shape[0] // 2)) for x in stills])
        cv2.imwrite(f"{a.out}_{cat}.png", sheet)
        print(f"wrote {a.out}_{cat}.mp4 and .png ({len(clips)} clips)", flush=True)
    with open(f"{a.out}_index.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(index) + "\n")
    print(f"wrote {a.out}_index.txt")


if __name__ == "__main__":
    main()
