"""[D] H/A team model of version 3, trained on OTHER games' answer keys (external data only).

Why: the v2 team model learns from pseudo labels of the game it labels. Those are ~10% wrong and late at
every hand-over (the "keep the last team for possession_hold_s" rule), and the model learned that lag.
Here the model learns from the true H/A of other games (Metrica public sample games turned into
external/EVAL-G1, -G2 by tools/make_eval_data.py). The provided game's own labels are never used.

Two steps:
  1. once per external game: python -m possession.v3.pipeline_v3 --repo ../more-than-a-hack-2027 \
         --external external/EVAL-G1 --out results/team_feat/EVAL-G1.csv --team-features-out results/team_feat/EVAL-G1.npz
     (writes that game's team features AND its true H/A)
  2. any game: python -m possession.v3.pipeline_v3 ... --team-train results/team_feat/EVAL-G1.npz results/team_feat/EVAL-G2.npz
     A file of the same game as the one being labelled is skipped (leave-one-game-out on the EVAL games).

Also here: the last-touch features (which team controlled the ball last, and how long ago).
"""
import os

import numpy as np
import pandas as pd

from .. import models
from ..preprocess import run_lengths
from ..v2 import features_v2


# --------------------------------------------------------------------------- last-touch features
def add_lasttouch(Xh, Xa, prep, team_mat, bq, P):
    """s_own_lasttouch (1 = own team controlled the ball last, 0 = the other team, NaN = nobody yet),
    s_own_lasttouch_age (seconds since that control ended, capped at lt_cap_s) and s_own_lasttouch_x
    (+-1 x how recent). Control: a player <= lt_dist_m from the ball, the nearest opponent >= lt_opp_m
    away, for >= lt_min_s, ball confidence >= ball_conf_min and not in flight."""
    T = prep["n_frames"]
    px, py = prep["px"], prep["py"]
    use = (bq["conf"] >= P.ball_conf_min) & ~bq["flight"] & np.isfinite(bq["bx"])
    d = np.hypot(px - bq["bx"][:, None], py - bq["by"][:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    best = [np.where(team_mat == k, d, np.inf).min(axis=1) for k in (0, 1)]
    ctrl = np.full(T, -1, np.int8)
    for k in (0, 1):
        ok = use & (best[k] < P.lt_dist_m) & (best[1 - k] > P.lt_opp_m)
        for s, e in run_lengths(ok):
            if e - s >= P.frames(P.lt_min_s):
                ctrl[s:e] = k
    cap = P.lt_cap_s
    last = np.full(T, -1.0)
    since = np.full(T, cap)
    cur, t0 = -1, 0
    for t in range(T):
        if ctrl[t] >= 0:
            cur, t0 = ctrl[t], t
        last[t] = cur
        since[t] = min(cap, (t - t0) / P.fps) if cur >= 0 else cap
    for view, own in ((Xh, 0), (Xa, 1)):
        view["s_own_lasttouch"] = np.where(last < 0, np.nan, (last == own).astype(float))
        view["s_own_lasttouch_age"] = since
        view["s_own_lasttouch_x"] = np.where(last < 0, 0.0, np.where(last == own, 1.0, -1.0) * (cap - since) / cap)
    return Xh, Xa


# --------------------------------------------------------------------------- features of an external game
def save_team_features(path, name, Xh, Xa, conf, gt_labels):
    """Team features of one EXTERNAL game and its true H/A (1 = home, 0 = away, -1 = D / X)."""
    y = np.where(gt_labels == "H", 1, np.where(gt_labels == "A", 0, -1)).astype(np.int8)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, name=np.array(name), cols=np.array(list(Xh.columns)), Xh=Xh.values.astype(np.float32),
                        Xa=Xa.values.astype(np.float32), conf=np.asarray(conf, np.float32), y=y)


def _load(path):
    z = np.load(path, allow_pickle=False)
    cols = [str(c) for c in z["cols"]]
    return str(z["name"]), pd.DataFrame(z["Xh"], columns=cols), pd.DataFrame(z["Xa"], columns=cols), z["y"]


def fit_predict(Xh, Xa, P, train_files, this_game, log=print):
    """P(home) of this game from a model trained on other games' true H/A only."""
    parts = []
    for f in train_files:
        name, th, ta, y = _load(f)
        if name == this_game:
            log(f"[D] team model: skip {os.path.basename(f)} (same game as the one being labelled)")
            continue
        missing = [c for c in Xh.columns if c not in th.columns]
        if missing:
            raise SystemExit(f"{f} lacks team features {missing[:5]}... -- rebuild it with the same settings")
        parts.append((name, th[Xh.columns], ta[Xh.columns], y))
    if not parts:
        raise SystemExit("--team-train: no other game left to train on")
    TXh = pd.concat([p[1] for p in parts], ignore_index=True)
    TXa = pd.concat([p[2] for p in parts], ignore_index=True)
    Ty = np.concatenate([p[3] for p in parts])
    cols = features_v2.all_columns(Xh) if P.team_model_external == "single" else features_v2.base_columns(Xh)
    P_ext = _with_lgbm(P)
    m = models.TeamModel(P_ext, cols).fit(TXh, TXa, Ty)
    log(f"[D] team model trained on other games' answer keys: {[p[0] for p in parts]}, "
        f"{int((Ty >= 0).sum())} labelled frames, {len(cols)} features")
    log("team model top features: " + ", ".join(n for n, _ in m.importance(8)))
    return m.predict(Xh, Xa)


def _with_lgbm(P):
    import copy
    q = copy.copy(P)
    q.lgbm = dict(P.lgbm_external)
    return q
