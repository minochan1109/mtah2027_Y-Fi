"""End to end: provided data -> frame,label CSV.

    python -m possession.pipeline --repo ../more-than-a-hack-2027 --out results/team.csv
    python -m possession.pipeline --repo ... --mode rules        # no learning at all
    python -m possession.pipeline --repo ... --check             # also score on the public 5 min

--check prints the public-window score ONLY to confirm the pipeline is wired correctly.
Do not compare settings with it and do not tune on it (RULES.md).
"""
import argparse
import warnings
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import data, features, hmm, models, postprocess, preprocess, pseudo_labels
from .config import Params


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- rule mode
def rule_p_dead(fd, P):
    ms = preprocess.rolling(fd["speed_mean"].values, P.fps, "centre")
    z = (2.0 * (P.dead_walk_speed - ms) + 2.0 * fd["last_ball_out"].values * (fd["since_seen"].values > 1)
         + 1.0 * (fd["to_set_piece_s"].values < P.set_piece_lead_s) - 1.0)
    return sigmoid(np.nan_to_num(z, nan=0.0))


def rule_p_home(Xh, prep):
    """Ball visible: compare who reaches the ball first. Ball not visible: no evidence (0.5)."""
    diff = Xh["b_diff_ttb"].values               # own(home) - opp(away); negative = home first
    p = sigmoid(-3.0 * np.nan_to_num(diff, nan=0.0))
    return np.where(prep["ball_ok"], p, np.nan)


# --------------------------------------------------------------------------- learned mode
def learned_probs(fd, Xh, Xa, prep, P, verbose=True):
    control = pseudo_labels.control_events(prep, P)
    y_dead = pseudo_labels.dead_labels(fd, P, control)
    y_team = pseudo_labels.team_labels(control, y_dead, P)
    pseudo = {"pseudo_dead": y_dead, "pseudo_team": y_team}
    log(f"pseudo labels: dead={int((y_dead == 1).sum())} alive={int((y_dead == 0).sum())} "
        f"home={int((y_team == 1).sum())} away={int((y_team == 0).sum())}")

    dm = models.DeadModel(P).fit(fd, y_dead)
    p_dead = dm.predict(fd)
    if verbose:
        log("module 1 top coefficients: " + ", ".join(f"{n}={c:+.2f}" for n, c in dm.coefficients()[:6]))

    ball_rows = np.flatnonzero(prep["ball_ok"])
    noball_rows = np.flatnonzero(~prep["ball_ok"])
    bcols = features.ball_columns(Xh)
    scols = features.shape_columns(Xh)

    m2, y2 = models.self_train(models.TeamModel, P, bcols, Xh, Xa, y_team, ball_rows)
    m3, _ = models.self_train(models.TeamModel, P, scols, Xh, Xa, y2, np.arange(len(Xh)))
    p_home = np.full(len(Xh), np.nan)
    p_home[ball_rows] = m2.predict(Xh, Xa, ball_rows)
    p_home[noball_rows] = m3.predict(Xh, Xa, noball_rows)
    if verbose:
        log("module 2 top features: " + ", ".join(n for n, _ in m2.importance(6)))
        log("module 3 top features: " + ", ".join(n for n, _ in m3.importance(6)))
    return p_dead, p_home, pseudo


# --------------------------------------------------------------------------- main
def run(args):
    P = Params()
    if args.params:
        with open(args.params) as fh:
            for k, v in json.load(fh).items():
                setattr(P, k, v)
    sys.path.insert(0, args.repo)
    import games  # noqa: E402

    if args.external:
        # another game converted to the same file formats (see tools/README_external.md)
        with open(os.path.join(args.external, "game.json")) as fh:
            cfg = json.load(fh)
        name = cfg["stem"]
        stem = os.path.join(args.external, cfg["stem"])
        paths = {"atd_meta": stem + "_atd.xml", "atd_raw": stem + "_atd.txt",
                 "homography": stem + "_homography.csv", "smart_tagging": stem + "_smart_tagging.xml"}
        n = int(cfg["n_frames"])
        blocks = [tuple(b) for b in cfg["blocks"]]
        P.pitch_length, P.pitch_width = cfg.get("pitch", (P.pitch_length, P.pitch_width))
    else:
        name = args.game
        data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(args.repo, "data")
        paths = games.asset_paths(args.game, data_dir)
        n = args.n_frames or games.source_frames("tactical", args.game)
        blocks = json.loads(args.blocks) if args.blocks else data.graded_blocks(args.repo, args.game)
        P.pitch_length, P.pitch_width = games.GAMES[args.game]["pitch"] if args.game in games.GAMES else (P.pitch_length, P.pitch_width)
    os.makedirs(args.cache_dir, exist_ok=True)

    log("loading tracking")
    tr = data.load_tracking(paths["atd_meta"], paths["atd_raw"], n, (P.pitch_length, P.pitch_width),
                            cache=os.path.join(args.cache_dir, f"{name}_tracking.npz"))
    if os.path.isfile(paths["homography"]):
        H, _, valid = data.load_homography(paths["homography"], n)
    else:
        log("no homography file: camera position features are left empty")
        H, valid = np.full((n, 3, 3), np.nan), np.zeros(n, bool)
    tags = data.load_smart_tagging(paths["smart_tagging"], P.fps) if os.path.isfile(paths["smart_tagging"]) else []
    side_of = lambda name: data.tag_team_side(name, tr.team_names)  # noqa: E731
    mapping = {t: side_of(t) for t in sorted({e["tags"].get("Team") for e in tags} - {None})}
    log(f"teams {tr.team_names}; Smart Tagging team mapping {mapping} (check it once)")

    log("preprocessing")
    prep = preprocess.prepare(tr, H, valid, blocks, P)
    log(f"ball visible {prep['ball_seen'].mean():.1%}, after short-gap interpolation {prep['ball_ok'].mean():.1%}")

    log("features")
    fd = features.dead_features(prep, tags, P)
    Xh, Xa = features.team_features(prep, P)

    if args.mode == "rules":
        p_dead, p_home, pseudo = rule_p_dead(fd, P), rule_p_home(Xh, prep), {}
    else:
        p_dead, p_home, pseudo = learned_probs(fd, Xh, Xa, prep, P)

    log("state model")
    prior = hmm.tagging_prior(tags, side_of, n, P)
    p_home = hmm.combine(p_home, prior, P)
    logE = hmm.emissions(p_dead, p_home, P)
    states = [hmm.viterbi(logE[a:b + 1], P) for _, a, b in blocks]
    labels = postprocess.to_labels(states, n, blocks)
    labels = postprocess.shift_transitions(labels, P.transition_shift_frames)

    data.write_submission(args.out, labels)
    log(f"wrote {args.out}")
    print(json.dumps(postprocess.summary(labels, blocks, P.fps), indent=2))
    np.savez_compressed(os.path.join(args.cache_dir, f"{name}_probs.npz"), p_dead=p_dead, p_home=p_home)

    if args.dump:
        frame = np.arange(n)
        sec = frame / P.fps
        table = pd.DataFrame({"frame": frame, "video_time": [f"{int(t // 60):02d}:{t % 60:05.2f}" for t in sec],
                              "label": labels, "p_dead": np.round(p_dead, 3), "p_home": np.round(p_home, 3),
                              "ball_seen": prep["ball_seen"].astype(int)})
        for k, v in pseudo.items():
            table[k] = v
        table.to_csv(args.dump, index=False)
        log(f"wrote per-frame details to {args.dump}")

    if args.external and os.path.isfile(os.path.join(args.external, "gt.csv")):
        sys.path.insert(0, os.path.join(args.repo, "evaluation"))
        import possession_metrics as pm  # noqa: E402
        gt = pm.label_codes(pm.read_label_csv(os.path.join(args.external, "gt.csv")))
        res = pm.evaluate(gt, pm.label_codes(labels))
        log("EXTERNAL GAME score (allowed for tuning): macroF1={:.4f} spellF1={:.4f} possessions={}".format(
            res["macro_f1"], res["spell_f1_mean"], res["n_pred_spells"]))

    if args.check:
        sys.path.insert(0, os.path.join(args.repo, "evaluation"))
        import possession_metrics as pm  # noqa: E402
        gt_path = args.gt or os.path.join(args.repo, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
        gt = pm.label_codes(pm.read_label_csv(gt_path))
        res = pm.evaluate(gt, pm.label_codes(labels))
        log("PIPELINE CHECK ONLY (not for tuning): macroF1={:.4f} spellF1={:.4f} possessions={}".format(
            res["macro_f1"], res["spell_f1_mean"], res["n_pred_spells"]))


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*fragmented.*")
    warnings.filterwarnings("ignore", message=".*Skipping features without any observed values.*")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, help="path to the more-than-a-hack-2027 clone")
    ap.add_argument("--data-dir", help="data bundle (default: $MTAH_DATA_DIR or <repo>/data)")
    ap.add_argument("--game", default="BAR-ATH")
    ap.add_argument("--out", default="results/submission.csv")
    ap.add_argument("--mode", choices=["learned", "rules"], default="learned")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--params", help="JSON file overriding config.Params fields")
    ap.add_argument("--check", action="store_true", help="score on the public 5 minutes (sanity only)")
    ap.add_argument("--gt", help="ground truth CSV for --check (default: public)")
    ap.add_argument("--dump", help="also write a per-frame CSV (time, label, probabilities, pseudo labels)")
    ap.add_argument("--external", help="folder with another game converted to the same formats + game.json (+ gt.csv)")
    ap.add_argument("--n-frames", type=int, help=argparse.SUPPRESS)   # synthetic tests only
    ap.add_argument("--blocks", help=argparse.SUPPRESS)               # synthetic tests only
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
