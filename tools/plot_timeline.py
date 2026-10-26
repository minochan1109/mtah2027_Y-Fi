"""Draw one stretch of the match as a picture, to compare with the video at the same time.

    python -m possession.pipeline ... --dump results/detail.csv
    python tools/plot_timeline.py --detail results/detail.csv --start 12:30 --seconds 60 --out results/12m30.png

Top: the estimated label (H blue, A orange, D grey, X white). Middle: P(dead) and P(home).
Bottom: when the ball is visible, and the pseudo labels if the run was in learned mode.
The x axis is VIDEO time (mm:ss), so you can open the tactical video at the same moment.

--gt draws a reference row too. Use it ONLY with another game (external/*/gt.csv) — never with
labels of the challenge game.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FPS = 25
COLORS = {"H": "#2b6cb0", "A": "#dd6b20", "D": "#a0aec0", "X": "#ffffff"}


def to_frame(text):
    if ":" in text:
        m, s = text.split(":")
        return int(round((int(m) * 60 + float(s)) * FPS))
    return int(text)


def strip(ax, labels, t, title):
    for lab, col in COLORS.items():
        ax.fill_between(t, 0, 1, where=(labels == lab), color=col, step="post", linewidth=0)
    ax.set_yticks([])
    ax.set_ylabel(title, rotation=0, ha="right", va="center")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", required=True, help="CSV written by the pipeline with --dump")
    ap.add_argument("--start", required=True, help="video time mm:ss (or a frame number)")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--gt", help="reference labels of ANOTHER game only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    d = pd.read_csv(a.detail)
    f0 = to_frame(a.start)
    f1 = min(len(d), f0 + int(a.seconds * FPS))
    d = d.iloc[f0:f1]
    t = d["frame"].to_numpy() / FPS

    rows = 4 + (a.gt is not None)
    fig, axes = plt.subplots(rows, 1, figsize=(14, 1.2 * rows + 1), sharex=True,
                             gridspec_kw={"height_ratios": [1] * (rows - 2) + [2, 1]})
    i = 0
    strip(axes[i], d["label"].to_numpy(), t, "estimate"); i += 1
    if a.gt:
        g = pd.read_csv(a.gt)["label"].to_numpy()[f0:f1]
        strip(axes[i], g, t, "reference"); i += 1
    if "pseudo_team" in d:
        pl = np.where(d["pseudo_dead"] == 1, "D", np.where(d["pseudo_team"] == 1, "H",
                      np.where(d["pseudo_team"] == 0, "A", "X")))
        strip(axes[i], pl, t, "pseudo label")
    else:
        axes[i].set_visible(False)
    i += 1
    ax = axes[i]; i += 1
    ax.plot(t, d["p_dead"], color="#4a5568", label="P(dead)")
    ax.plot(t, d["p_home"], color="#2b6cb0", label="P(home)")
    ax.set_ylim(0, 1); ax.legend(loc="upper right", fontsize=8)
    strip(axes[i], np.where(d["ball_seen"] == 1, "D", "X"), t, "ball seen")
    ticks = np.arange(np.ceil(t[0] / 5) * 5, t[-1] + 0.01, 5)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([f"{int(x // 60):02d}:{int(x % 60):02d}" for x in ticks])
    axes[-1].set_xlabel("video time (H blue / A orange / D grey)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
