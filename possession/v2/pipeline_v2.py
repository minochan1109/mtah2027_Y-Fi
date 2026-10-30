"""Version 2 end to end.

    python -m possession.v2.pipeline_v2 --repo ../more-than-a-hack-2027 --out results/v2.csv \
        --report results/v2_report.txt --dump results/v2_detail.csv
    python -m possession.v2.pipeline_v2 --repo ../more-than-a-hack-2027 --external external/MET-G1 --out results/v2_g1.csv

Same options as version 1 (--external, --dump, --check, --params). --report writes what every
module did (players removed, ball rejected by reason, tags used, top features, summary).
--check is a PIPELINE CHECK ONLY: never compare versions or settings with it.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from .. import data, features, hmm, models, postprocess, preprocess, pseudo_labels
from . import ball_quality, clean_players, features_v2, hmm_v2, model_v2, pseudo_v2, tag_policy
from .ball_quality import TRUSTED
from .config_v2 import Params2

REPORT = []


def _external_gt(folder):
    """Truth labels of an external game: game.json "gt" (make_eval_data.py) or gt.csv (convert_metrica.py)."""
    try:
        with open(os.path.join(folder, "game.json")) as fh:
            return os.path.join(folder, json.load(fh).get("gt", "gt.csv"))
    except OSError:
        return os.path.join(folder, "gt.csv")


def log(msg):
    line = time.strftime("%H:%M:%S") + " " + str(msg)
    print(line, flush=True)
    REPORT.append(line)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _paths(args, P):
    args.repo = os.path.abspath(args.repo)
    if not os.path.isfile(os.path.join(args.repo, "games.py")):
        raise SystemExit(f"--repo does not point at more-than-a-hack-2027 (games.py not found): {args.repo}")
    sys.path.insert(0, args.repo)
    import games  # noqa: E402
    if args.external:
        with open(os.path.join(args.external, "game.json")) as fh:
            cfg = json.load(fh)
        stem = os.path.join(args.external, cfg.get("input_dir", ""), cfg["stem"])
        paths = {"atd_meta": stem + "_atd.xml", "atd_raw": stem + "_atd.txt",
                 "homography": stem + "_homography.csv", "smart_tagging": stem + "_smart_tagging.xml"}
        P.pitch_length, P.pitch_width = cfg.get("pitch", (P.pitch_length, P.pitch_width))
        return cfg["stem"], paths, int(cfg["n_frames"]), [tuple(b) for b in cfg["blocks"]]
    data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(args.repo, "data")
    paths = games.asset_paths(args.game, data_dir)
    n = args.n_frames or games.source_frames("tactical", args.game)
    blocks = json.loads(args.blocks) if args.blocks else data.graded_blocks(args.repo, args.game)
    if args.game in games.GAMES:
        P.pitch_length, P.pitch_width = games.GAMES[args.game]["pitch"]
    return args.game, paths, n, blocks


def run(args):
    P = Params2()
    if args.params:
        with open(args.params) as fh:
            for k, v in json.load(fh).items():
                setattr(P, k, v)
    name, paths, n, blocks = _paths(args, P)
    os.makedirs(args.cache_dir, exist_ok=True)

    log("loading tracking")
    tr = data.load_tracking(paths["atd_meta"], paths["atd_raw"], n, (P.pitch_length, P.pitch_width),
                            cache=os.path.join(args.cache_dir, f"{name}_tracking.npz"))
    if os.path.isfile(paths["homography"]):
        H, _, valid = data.load_homography(paths["homography"], n)
    else:
        H, valid = np.full((n, 3, 3), np.nan), np.zeros(n, bool)
    tags = tag_policy.load_tags(paths["smart_tagging"], P.fps) if os.path.isfile(paths["smart_tagging"]) else []
    side_of = lambda nm: data.tag_team_side(nm, tr.team_names)  # noqa: E731

    # [A] players
    attack = preprocess.attack_sign(tr, blocks, P.fps)
    px, py, team_mat, rep_a = clean_players.clean_players(tr, attack, P)
    log(f"[A] clean_players: {rep_a}")

    # [B] ball
    bq, rep_b = ball_quality.assess_ball(tr.bx, tr.by, px, py, P)
    log(f"[B] ball_quality: {rep_b}")

    # shared preprocessing on the cleaned data (version 1 code, cleaned inputs)
    tr2 = data.Tracking(px.astype(np.float32), py.astype(np.float32), tr.pspeed, tr.team,
                        bq["bx"].astype(np.float32), bq["by"].astype(np.float32), tr.team_names)
    prep = preprocess.prepare(tr2, H, valid, blocks, P)
    prep["ball_seen"] = bq["status"] == TRUSTED

    # [C] tags
    tag_logit, tag_dead, rep_c = tag_policy.tag_evidence(tags, side_of, n, P)
    log(f"[C] tag_policy: {rep_c}")

    # [D] features, pseudo labels, models
    log("features")
    fd = features.dead_features(prep, tags, P)
    fd = pd.concat([fd, features_v2.dead_extra(bq, tag_dead, P)], axis=1)
    Xh, Xa = features_v2.team_features_v2(prep, team_mat, bq, P)

    control = pseudo_v2.control_events_v2(prep, team_mat, bq, P)
    y_dead = pseudo_labels.dead_labels(fd, P, control)
    y_team = pseudo_v2.team_labels_v2(control, y_dead, P)
    log(f"[D] pseudo labels: dead={int((y_dead == 1).sum())} alive={int((y_dead == 0).sum())} "
        f"home={int((y_team == 1).sum())} away={int((y_team == 0).sum())}")
    dm = models.DeadModel(P).fit(fd, y_dead)
    p_dead = dm.predict(fd)
    log("dead model top coefficients: " + ", ".join(f"{a}={c:+.2f}" for a, c in dm.coefficients()[:6]))
    p_model, _ = model_v2.fit_predict(Xh, Xa, y_team, bq["conf"], P, log=log)
    p_home = 1 / (1 + np.exp(-(_logit(p_model) + tag_logit)))

    # [E] state model
    log("state model")
    sig = hmm_v2.switch_signal(Xh, P)
    switch_cost = P.cost_switch_team * (1.0 - P.switch_signal_drop * sig)
    logE = hmm.emissions(p_dead, p_home, P)
    states = [hmm_v2.viterbi_tv(logE[a:b + 1], P, switch_cost[a:b + 1]) for _, a, b in blocks]
    labels = postprocess.to_labels(states, n, blocks)
    data.write_submission(args.out, labels)
    log(f"wrote {args.out}")
    summ = postprocess.summary(labels, blocks, P.fps)
    log("summary: " + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in summ.items()}))

    if args.dump:
        t = np.arange(n) / P.fps
        pd.DataFrame({"frame": np.arange(n),
                      "video_time": [f"{int(s // 60):02d}:{s % 60:05.2f}" for s in t],
                      "label": labels, "p_dead": np.round(p_dead, 3), "p_home": np.round(p_home, 3),
                      "p_model": np.round(p_model, 3), "tag_logit": np.round(tag_logit, 2),
                      "ball_status": bq["status"], "ball_conf": np.round(bq["conf"], 2),
                      "ball_flight": bq["flight"].astype(int), "switch_signal": np.round(sig, 2),
                      "ball_seen": (bq["status"] == TRUSTED).astype(int),
                      "pseudo_dead": y_dead, "pseudo_team": y_team}).to_csv(args.dump, index=False)
        log(f"wrote {args.dump}")

    sys.path.insert(0, os.path.join(args.repo, "evaluation"))
    if args.external and os.path.isfile(_external_gt(args.external)):
        import possession_metrics as pm  # noqa: E402
        gt = pm.label_codes(pm.read_label_csv(_external_gt(args.external)))
        res = pm.evaluate(gt, pm.label_codes(labels))
        log("EXTERNAL GAME score (allowed for tuning): macroF1={:.4f} spellF1={:.4f} possessions={}".format(
            res["macro_f1"], res["spell_f1_mean"], res["n_pred_spells"]))
    if args.check:
        import possession_metrics as pm  # noqa: E402
        gt_path = args.gt or os.path.join(args.repo, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
        res = pm.evaluate(pm.label_codes(pm.read_label_csv(gt_path)), pm.label_codes(labels))
        log("PIPELINE CHECK ONLY (not for tuning): macroF1={:.4f} spellF1={:.4f}".format(
            res["macro_f1"], res["spell_f1_mean"]))
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("\n".join(REPORT) + "\n")
        print(f"wrote {args.report}")


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*fragmented.*")
    warnings.filterwarnings("ignore", message=".*Skipping features without any observed values.*")
    warnings.filterwarnings("ignore", message=".*All-NaN.*")
    REPORT.clear()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data-dir")
    ap.add_argument("--game", default="BAR-ATH")
    ap.add_argument("--out", default="results/v2.csv")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--params", help="JSON overriding Params2 fields")
    ap.add_argument("--report", help="text file with what every module did")
    ap.add_argument("--dump", help="per-frame CSV for the debug tools")
    ap.add_argument("--external", help="another game converted with tools/convert_metrica.py")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gt")
    ap.add_argument("--n-frames", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--blocks", help=argparse.SUPPRESS)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
