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
    if getattr(P, "team_nexttouch", False):
        # the same control, looking forward: which team controls the ball NEXT, and how soon
        nxt = np.full(T, -1.0)
        until = np.full(T, cap)
        cur, t0 = -1, T
        for t in range(T - 1, -1, -1):
            if ctrl[t] >= 0:
                cur, t0 = ctrl[t], t
            nxt[t] = cur
            until[t] = min(cap, (t0 - t) / P.fps) if cur >= 0 else cap
        for view, own in ((Xh, 0), (Xa, 1)):
            view["s_own_nexttouch"] = np.where(nxt < 0, np.nan, (nxt == own).astype(float))
            view["s_own_nexttouch_age"] = until
            view["s_own_nexttouch_x"] = np.where(nxt < 0, 0.0, np.where(nxt == own, 1.0, -1.0) * (cap - until) / cap)
    return Xh, Xa


# --------------------------------------------------------------------------- features of an external game
def save_team_features(path, name, Xh, Xa, conf, gt_labels):
    """Team features of one EXTERNAL game and its true H/A (1 = home, 0 = away, -1 = D / X)."""
    y = np.where(gt_labels == "H", 1, np.where(gt_labels == "A", 0, -1)).astype(np.int8)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, name=np.array(name), cols=np.array(list(Xh.columns)), Xh=Xh.values.astype(np.float32),
                        Xa=Xa.values.astype(np.float32), conf=np.asarray(conf, np.float32), y=y)


def dead_path(team_npz):
    """The D-features file that goes with a team-features file (EVAL-G1.npz -> EVAL-G1_dead.npz)."""
    root, ext = os.path.splitext(team_npz)
    return root + "_dead" + (ext or ".npz")


def save_dead_features(path, name, fd, gt_labels, true_ball_seen=None):
    """D-model features of one EXTERNAL game and its true D (1 = D, 0 = H / A, -1 = X / unknown),
    plus whether the answer key's own ball is known in each frame (for dead_train_unseen_s)."""
    g = np.asarray(gt_labels).astype(str)
    y = np.where(g == "D", 1, np.where(np.isin(g, ["H", "A"]), 0, -1)).astype(np.int8)
    seen = np.ones(len(y), bool) if true_ball_seen is None else np.asarray(true_ball_seen, bool)[:len(y)]
    if len(seen) < len(y):
        seen = np.r_[seen, np.ones(len(y) - len(seen), bool)]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, name=np.array(name), cols=np.array(list(fd.columns)),
                        X=fd.values[:len(y)].astype(np.float32), y=y, true_ball_seen=seen)


def _fix_unseen(y, seen, min_s, fps):
    """The external answer keys come from event data: a stoppage without its event is labelled play.
    Their ball is blanked while play is stopped, so 'play' with the answer's ball unknown for >= min_s
    is taken as D (training labels of the OTHER games only)."""
    y = y.copy()
    for s, e in run_lengths(~seen):
        if e - s >= min_s * fps:
            seg = y[s:e]
            seg[seg == 0] = 1
    return y


def fit_dead(fd, P, train_files, this_game, log=print):
    """The D model (same logistic regression and features as before) fitted on OTHER games' true D."""
    parts = []
    for f in train_files:
        dp = dead_path(f)
        if not os.path.isfile(dp):
            raise SystemExit(f"{dp} missing -- rebuild the external files with --team-features-out")
        z = np.load(dp, allow_pickle=False)
        if str(z["name"]) == this_game:
            continue
        cols = [str(c) for c in z["cols"]]
        missing = [c for c in fd.columns if c not in cols]
        if missing:
            raise SystemExit(f"{dp} lacks D features {missing[:5]}... -- rebuild it with the same settings")
        y = z["y"]
        fix_s = getattr(P, "dead_train_unseen_s", 0.0)
        if fix_s > 0 and "true_ball_seen" in z.files:
            y2 = _fix_unseen(y, z["true_ball_seen"], fix_s, P.fps)
            log(f"[D] {z['name']}: {int(((y2 == 1) & (y == 0)).sum())} 'play' frames with the answer's ball "
                f"unknown >= {fix_s} s taken as D for training")
            y = y2
        parts.append((str(z["name"]), pd.DataFrame(z["X"], columns=cols)[list(fd.columns)], y))
    if not parts:
        raise SystemExit("--team-train: no other game left to train the D model on")
    X = pd.concat([p[1] for p in parts], ignore_index=True)
    y = np.concatenate([p[2] for p in parts])
    dm = models.DeadModel(P).fit(X, y)
    log(f"[D] D model trained on other games' true D: {[p[0] for p in parts]}, "
        f"dead={int((y == 1).sum())} alive={int((y == 0).sum())}")
    return dm


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
    n_seeds = int(getattr(P, "team_seeds", 1))
    if n_seeds <= 1:
        return m.predict(Xh, Xa)
    # average several random seeds: one seed's P(home) moves spellF1 by about 0.01
    preds = [m.predict(Xh, Xa)]
    for seed in range(1, n_seeds):
        P_s = _with_lgbm(P)
        P_s.lgbm["random_state"] = seed
        preds.append(models.TeamModel(P_s, cols).fit(TXh, TXa, Ty).predict(Xh, Xa))
    log(f"[D] team model P(home) averaged over {n_seeds} random seeds")
    return np.mean(preds, axis=0)


def self_train(Xh, Xa, P, train_files, this_game, p_first, log=print):
    """Second pass: other games' true H/A PLUS this game's frames where the first model is confident
    (no answer key of this game is used), weighted by team_self_train_weight."""
    conf = P.team_self_train_conf
    parts = []
    for f in train_files:
        name, th, ta, y = _load(f)
        if name != this_game:
            parts.append((th[Xh.columns], ta[Xh.columns], y, np.ones(len(y))))
    sure = np.where(p_first >= conf, 1, np.where(p_first <= 1 - conf, 0, -1)).astype(np.int8)
    parts.append((Xh, Xa, sure, np.full(len(sure), P.team_self_train_weight)))
    TXh = pd.concat([p[0] for p in parts], ignore_index=True)
    TXa = pd.concat([p[1] for p in parts], ignore_index=True)
    Ty = np.concatenate([p[2] for p in parts]); Tw = np.concatenate([p[3] for p in parts])
    cols = features_v2.all_columns(Xh) if P.team_model_external == "single" else features_v2.base_columns(Xh)
    import lightgbm as lgb
    idx = np.flatnonzero(Ty >= 0)
    X = np.vstack([TXh.iloc[idx][cols].values, TXa.iloc[idx][cols].values])
    t = np.r_[(Ty[idx] == 1).astype(int), (Ty[idx] == 0).astype(int)]
    w = np.r_[Tw[idx], Tw[idx]]
    preds = []
    for seed in range(max(1, int(getattr(P, "team_seeds", 1)))):
        prm = dict(P.lgbm_external); prm["random_state"] = seed
        m = lgb.LGBMClassifier(**prm).fit(X, t, sample_weight=w)
        ph = m.predict_proba(Xh[cols].values)[:, 1]; pa = m.predict_proba(Xa[cols].values)[:, 1]
        preds.append(0.5 * (ph + (1 - pa)))
    log(f"[D] team model self-trained: + {int((sure >= 0).sum())} confident frames of this game "
        f"(P >= {conf}, weight {P.team_self_train_weight})")
    return np.mean(preds, axis=0)


def fit_predict_switch(Xh, P, train_files, this_game, log=print):
    """P(an H<->A switch within +-switch_model_win_s) of this game, learned from OTHER games' true H/A only.
    The hand-made recovery signal (hmm_v2.switch_signal) ranks switch moments no better than chance on EVAL
    (AUC 0.49 / 0.50); this model reaches AUC 0.85 / 0.86 (leave-one-game-out)."""
    import lightgbm as lgb
    fps = P.fps
    win = int(round(P.switch_model_win_s * fps))
    X, Y = [], []
    for f in train_files:
        name, th, _, y = _load(f)
        if name == this_game:
            continue
        lab = np.zeros(len(y), bool)
        idx = np.flatnonzero(np.isin(y, [0, 1]))
        for a, b in zip(idx[:-1], idx[1:]):
            if y[a] != y[b] and b - a <= fps:       # direct switch, or across a gap shorter than 1 s
                lab[max(0, b - win):b + win] = True
        keep = np.isin(y, [0, 1])
        X.append(th[Xh.columns][keep]); Y.append(lab[keep])
    if not X:
        raise SystemExit("--team-train: no other game left to train on")
    X = pd.concat(X, ignore_index=True); Y = np.concatenate(Y)
    preds = []
    for seed in range(max(1, int(getattr(P, "team_seeds", 1)))):
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                               random_state=seed, deterministic=True, force_row_wise=True)
        m.fit(X, Y)
        preds.append(m.predict_proba(Xh)[:, 1])
    p = np.mean(preds, axis=0)
    log(f"[E] switch model trained on other games: {len(Y)} frames, switch share {Y.mean():.3f}, "
        f"{len(preds)} seeds; P(switch) mean {p.mean():.3f}")
    return p


def _with_lgbm(P):
    import copy
    q = copy.copy(P)
    q.lgbm = dict(P.lgbm_external)
    return q
