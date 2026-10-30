"""Score the v2 clean-up (players [A] and ball [B]) against the truth of an evaluation game.

    python tools/eval_cleaning.py --games external/EVAL-G1 external/EVAL-G2
    python tools/eval_cleaning.py --games external/EVAL-G1 external/EVAL-G2 --params params/my_try.json

Needs games made with tools/make_eval_data.py (they keep truth/errors.npz). Nothing here uses the
challenge game, so these numbers may be used to choose settings.

BALL (a position counts as correct when it is within --tol metres of the true ball)
    recall       correct positions / frames where the true ball is tracked      <- MT-10 target
    precision    correct positions / positions we give to the model
    false kept   false detections still used after the clean-up (share of all false detections)
    real lost    correct detections thrown away by the clean-up (share of all correct detections)
PLAYERS
    false removed      false objects removed (share of false-object detections)
    real removed       real players removed by mistake (share of real detections)
    swaps fixed        detections in the wrong team that now have the right team (share)
    team correct       real detections with the right team after the clean-up (share)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from possession import data, preprocess  # noqa: E402
from possession.v2.ball_quality import assess_ball  # noqa: E402
from possession.v2.clean_players import clean_players  # noqa: E402
from possession.v2.config_v2 import Params2  # noqa: E402

ERR_NONE, ERR_REMOVED, ERR_FALSE, ERR_SWAP_IN, ERR_SWAP_OUT = 0, 1, 2, 3, 4


def evaluate(folder, params, tol):
    with open(os.path.join(folder, "game.json")) as fh:
        cfg = json.load(fh)
    P = Params2()
    for k, v in params.items():
        setattr(P, k, v)
    P.pitch_length, P.pitch_width = cfg["pitch"]
    n = int(cfg["n_frames"])
    stem = os.path.join(folder, cfg.get("input_dir", ""), cfg["stem"])
    tr = data.load_tracking(stem + "_atd.xml", stem + "_atd.txt", n, tuple(cfg["pitch"]))
    err = np.load(os.path.join(folder, "truth", "errors.npz"))
    blocks = [tuple(b) for b in cfg["blocks"]]
    attack = preprocess.attack_sign(tr, blocks, P.fps)

    cpx, cpy, team_mat, _ = clean_players(tr, attack, P)
    bq, _ = assess_ball(tr.bx, tr.by, cpx, cpy, P)

    tbx, tby = err["true_bx"], err["true_by"]
    true_ok = np.isfinite(tbx)
    out_ok = np.isfinite(bq["bx"]) & (bq["conf"] >= P.ball_conf_min)
    d_out = np.hypot(bq["bx"] - tbx, bq["by"] - tby)
    correct = out_ok & true_ok & (d_out <= tol)
    d_in = np.hypot(tr.bx - tbx, tr.by - tby)
    in_correct = np.isfinite(tr.bx) & true_ok & (d_in <= tol)
    bstate = err["ball_state"]
    false_det = bstate == ERR_FALSE
    ball = {
        "recall_input": in_correct.sum() / max(1, true_ok.sum()),
        "recall": correct.sum() / max(1, true_ok.sum()),
        "precision": correct.sum() / max(1, out_ok.sum()),
        "false_kept": (false_det & out_ok & ~correct).sum() / max(1, false_det.sum()),
        "real_lost": (in_correct & ~correct).sum() / max(1, in_correct.sum()),
    }

    state, src = err["player_state"], err["player_src"]
    true_team = err["true_team"]
    present = np.isfinite(tr.px)
    kept = np.isfinite(cpx)
    false_obj = present & (state == ERR_FALSE)
    real = present & (src >= 0)
    swap = present & (state == ERR_SWAP_IN)
    tt = np.where(src >= 0, true_team[np.clip(src, 0, None)], -1)
    players = {
        "false_removed": (false_obj & ~kept).sum() / max(1, false_obj.sum()),
        "real_removed": (real & ~kept).sum() / max(1, real.sum()),
        "swaps_fixed": (swap & kept & (team_mat == tt)).sum() / max(1, swap.sum()),
        "team_correct": (real & kept & (team_mat == tt)).sum() / max(1, (real & kept).sum()),
    }
    return ball, players


def main(argv=None):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--params", help="JSON overriding Params2 fields (same as the pipelines)")
    ap.add_argument("--tol", type=float, default=2.0, help="metres for a ball position to count as correct")
    a = ap.parse_args(argv)
    params = {}
    if a.params:
        with open(a.params) as fh:
            params = json.load(fh)
    rows = []
    for g in a.games:
        ball, players = evaluate(g, params, a.tol)
        rows.append((os.path.basename(os.path.normpath(g)), ball, players))
    keys_b = ["recall_input", "recall", "precision", "false_kept", "real_lost"]
    keys_p = ["false_removed", "real_removed", "swaps_fixed", "team_correct"]
    print(f"settings: {a.params or 'default'}   ball tolerance {a.tol} m")
    print("BALL     " + "".join(f"{k:>14}" for k in keys_b))
    for name, b, _ in rows:
        print(f"{name:<9}" + "".join(f"{b[k]:14.3f}" for k in keys_b))
    print("PLAYERS  " + "".join(f"{k:>14}" for k in keys_p))
    for name, _, p in rows:
        print(f"{name:<9}" + "".join(f"{p[k]:14.3f}" for k in keys_p))
    print("higher is better except false_kept, real_lost, real_removed")


if __name__ == "__main__":
    main()
