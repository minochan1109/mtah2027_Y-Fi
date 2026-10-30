"""Why did we say D (dead ball)?  Debug output for the dead-ball decision.

The D decision is made in three layers. This file shows each of them per frame and per D stretch:
  1. pseudo labels  (pseudo_labels.dead_labels): which rule made a training frame "dead" / "alive"
  2. dead model     (models.DeadModel, logistic regression): P(dead) and which features pushed it
                    up or down -- contribution = coefficient x standardised value (log-odds)
  3. state model    (HMM): the final H / A / D after the minimum-length and switching costs

Written by pipeline_v2 --dead-debug <prefix>:
  <prefix>_frames.csv     one row per frame: label, P(dead), pseudo label + the rule that fired,
                          key raw values, and the 5 features that pushed P(dead) most (+ up, - down)
  <prefix>_stretches.txt  every final D stretch with its time, length, mean P(dead), the rules and the
                          features behind it; for other games with a truth (gt) also whether the
                          stretch is a real stoppage, and a comparison of false D vs real D
Only algorithmic quantities are shown; nothing here labels the provided game by hand.
"""
import numpy as np
import pandas as pd

from ..preprocess import rolling, run_lengths


def label_reasons(fd, P, control):
    """The rule flags inside pseudo_labels.dead_labels, as separate arrays (same logic)."""
    T = len(fd)
    ms = rolling(fd["speed_mean"].values, P.fps, "centre")
    walking = ms < P.dead_walk_speed
    calm = np.zeros(T, bool)
    for s, e in run_lengths(walking):
        if e - s >= P.frames(P.dead_min_s):
            calm[s:e] = True
    since = np.nan_to_num(fd["since_seen"].values, nan=99.0)
    bspeed = np.nan_to_num(fd["ball_speed"].values, nan=0.0)
    return {
        "calm": calm,                                             # everybody walking >= dead_min_s
        "ball_out": fd["last_ball_out"].values > 0,                # last ball seen outside the pitch
        "before_setpiece": fd["to_set_piece_s"].values < P.set_piece_lead_s,
        "ball_unseen": since > 2.0,                                # no ball for > 2 s
        "ball_still": bspeed < 0.5,
        "alive_travel": (fd["ball_seen"].values > 0) & (bspeed > 3.0) & (fd["last_ball_out"].values == 0),
        "alive_run": ms > P.alive_run_speed,
        "alive_control": control >= 0,
    }


def contributions(dm, fd):
    """(T, F) log-odds contribution of each feature to P(dead), and the feature names."""
    X = fd[dm.cols].values
    Z = dm.model[:-1].transform(X)                               # imputer (+ missing flags) and scaler
    coef = dm.model[-1].coef_[0]
    names = dm.feature_names()
    return Z * coef[None, :], names, float(dm.model[-1].intercept_[0])


def _top(contrib, names, rows, k=3, sign=+1):
    m = (contrib[rows] * sign).mean(axis=0)
    order = np.argsort(-m)[:k]
    return ", ".join(f"{names[i]} {sign * m[i]:+.2f}" for i in order if m[i] > 0)


def explain(fd, dm, P, control, y_dead, p_dead, labels, blocks, prefix, gt=None, fps=25):
    contrib, names, b0 = contributions(dm, fd)
    reasons = label_reasons(fd, P, control)
    T = len(labels)

    # ---- per-frame CSV
    order = np.argsort(-np.abs(contrib), axis=1)[:, :5]
    rows = np.arange(T)
    t = np.arange(T) / fps
    df = pd.DataFrame({"frame": rows, "video_time": [f"{int(s // 60):02d}:{s % 60:05.2f}" for s in t],
                       "label": labels, "p_dead": np.round(p_dead, 3), "pseudo_dead": y_dead})
    if gt is not None:
        df["gt"] = gt
    why_dead = np.where(reasons["calm"] & reasons["ball_out"], "calm+ball_out",
               np.where(reasons["calm"] & reasons["before_setpiece"], "calm+before_setpiece",
               np.where(reasons["calm"] & reasons["ball_unseen"], "calm+ball_unseen",
               np.where(reasons["calm"] & reasons["ball_still"], "calm+ball_still", ""))))
    why_alive = np.where(reasons["alive_control"], "control",
                np.where(reasons["alive_travel"], "ball_travelling",
                np.where(reasons["alive_run"], "players_running", "")))
    df["pseudo_rule"] = np.where(y_dead == 1, why_dead, np.where(y_dead == 0, why_alive, ""))
    for c in ("speed_mean", "frac_walking", "ball_seen", "since_seen", "ball_speed", "last_ball_out",
              "to_set_piece_s", "ball_conf", "ball_flight"):
        if c in fd:
            df[c] = np.round(fd[c].values.astype(float), 2)
    for j in range(5):
        idx = order[:, j]
        df[f"top{j + 1}"] = [f"{names[i]}:{contrib[r, i]:+.2f}" for r, i in zip(rows, idx)]
    keep = np.zeros(T, bool)
    for _, a, b in blocks:
        keep[a:b + 1] = True
    df[keep].to_csv(prefix + "_frames.csv", index=False)

    # ---- per-stretch text
    L = []
    add = L.append
    add("WHY D?  every final D stretch (video time).  P(dead) = sigmoid(intercept + sum of contributions)")
    add(f"intercept {b0:+.2f}.  '+' features push towards D, '-' towards play.")
    add("pseudo = share of the stretch whose pseudo label is dead / alive / unknown")
    if gt is not None:
        add("truth = share of the stretch that is D in the truth of this OTHER game (false D if low)")
    add("")
    stretches = []
    for _, a, b in blocks:
        seg = labels[a:b + 1]
        for s, e in run_lengths(seg == "D"):
            stretches.append((a + s, a + e))
    false_rows, true_rows = [], []
    for s, e in stretches:
        r = np.arange(s, e)
        yd = y_dead[r]
        line = (f"{s:6d}-{e - 1:6d}  {df['video_time'][s]}  {(e - s) / fps:6.1f} s  P(dead) {p_dead[r].mean():.2f}  "
                f"pseudo dead/alive/unk {np.mean(yd == 1):.2f}/{np.mean(yd == 0):.2f}/{np.mean(yd < 0):.2f}  "
                f"ball seen {fd['ball_seen'].values[r].mean():.2f}  speed {np.nanmean(fd['speed_mean'].values[r]):.1f}")
        if gt is not None:
            tr = np.mean(gt[r] == "D")
            line += f"  truth D {tr:.2f}"
            (true_rows if tr >= 0.5 else false_rows).append(r)
        add(line)
        rules = [k for k in ("ball_out", "before_setpiece", "ball_unseen", "ball_still") if reasons[k][r].mean() > 0.5]
        add(f"      rules true >50%: {', '.join(rules) or '-'}   calm {reasons['calm'][r].mean():.2f}")
        add(f"      pushed to D by : {_top(contrib, names, r, 3, +1) or '-'}")
        add(f"      held back by   : {_top(contrib, names, r, 2, -1) or '-'}")
    add("")
    lens = np.array([(e - s) / fps for s, e in stretches]) if stretches else np.zeros(0)
    add(f"D stretches: {len(stretches)}  (<3 s: {(lens < 3).sum()}, 3-8 s: {((lens >= 3) & (lens < 8)).sum()}, "
        f">=8 s: {(lens >= 8).sum()})")
    if gt is not None and (false_rows or true_rows):
        fr = np.concatenate(false_rows) if false_rows else np.zeros(0, int)
        trr = np.concatenate(true_rows) if true_rows else np.zeros(0, int)
        add(f"false D stretches (truth D < 50%): {len(false_rows)}, {len(fr) / fps:.0f} s ; "
            f"real D stretches: {len(true_rows)}, {len(trr) / fps:.0f} s")
        live_as_d = (labels == "D") & np.isin(gt, ["H", "A"])
        add(f"frames of play called D: {live_as_d.sum()} ({live_as_d.mean():.1%} of all frames)")
        if len(fr) and len(trr):
            diff = contrib[fr].mean(axis=0) - contrib[trr].mean(axis=0)
            o = np.argsort(-diff)[:8]
            add("features pushing MORE towards D in false D than in real D (mean contribution, false - real):")
            for i in o:
                add(f"    {names[i]:<32} {diff[i]:+.2f}   (false {contrib[fr, i].mean():+.2f}, real {contrib[trr, i].mean():+.2f})")
            for k in ("ball_unseen", "ball_still", "ball_out", "before_setpiece", "calm"):
                add(f"    rule {k:<16} false D {reasons[k][fr].mean():.2f}  real D {reasons[k][trr].mean():.2f}")
            add(f"    pseudo label dead share   false D {np.mean(y_dead[fr] == 1):.2f}  real D {np.mean(y_dead[trr] == 1):.2f}")
    with open(prefix + "_stretches.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
