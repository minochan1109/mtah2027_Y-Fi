"""Do the MT-26 signals separate what they should on the evaluation games? (check only, no tuning)"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from possession import data, preprocess
from possession.v2 import ball_quality, clean_players, tag_policy, clue_features
from possession.v2.ball_quality import TRUSTED
from possession.v2.config_v2 import Params2

def load(g):
    P = Params2()
    folder = f"external/EVAL-{g}"
    cfg = json.load(open(os.path.join(folder, "game.json")))
    stem = os.path.join(folder, cfg["input_dir"], cfg["stem"])
    n = int(cfg["n_frames"]); blocks = [tuple(b) for b in cfg["blocks"]]
    P.pitch_length, P.pitch_width = cfg["pitch"]
    tr = data.load_tracking(stem + "_atd.xml", stem + "_atd.txt", n, (P.pitch_length, P.pitch_width), cache=f"cache/{cfg['stem']}_tracking.npz")
    H, valid = np.full((n, 3, 3), np.nan), np.zeros(n, bool)
    attack = preprocess.attack_sign(tr, blocks, P.fps)
    px, py, team_mat, _ = clean_players.clean_players(tr, attack, P)
    bq, _ = ball_quality.assess_ball(tr.bx, tr.by, px, py, P)
    tr2 = data.Tracking(px.astype(np.float32), py.astype(np.float32), tr.pspeed, tr.team, bq["bx"].astype(np.float32), bq["by"].astype(np.float32), tr.team_names)
    prep = preprocess.prepare(tr2, H, valid, blocks, P)
    tags = tag_policy.load_tags(stem + "_smart_tagging.xml", P.fps)
    gt = pd.read_csv(os.path.join(folder, cfg["gt"]))["label"].astype(str).to_numpy()
    return P, prep, team_mat, bq, tags, tr, gt

for g in ["G1", "G2"]:
    P, prep, team_mat, bq, tags, tr, gt = load(g)
    print("=====", g)
    fa = clue_features.align_features(prep, team_mat, P)
    for lab in ["H", "A", "D"]:
        m = gt == lab
        print(f"align_frac  gt={lab}: mean {np.nanmean(fa['align_frac'][m]):.3f}  median {np.nanmedian(fa['align_frac'][m]):.3f}")
    # around D starts
    st = np.flatnonzero((gt[1:] == "D") & np.isin(gt[:-1], ["H", "A"])) + 1
    for off in (-75, -25, 0, 25, 75, 125):
        idx = np.clip(st + off, 0, len(gt) - 1)
        print(f"  D start {off/25:+.0f}s: align_frac {np.nanmean(fa['align_frac'].values[idx]):.3f}, align_drop {np.nanmean(fa['align_drop'].values[idx]):+.3f}")
    sig = clue_features.crowd_switch_signal(prep, team_mat, bq, P)
    sw = np.flatnonzero(((gt[1:] == "H") & (gt[:-1] == "A")) | ((gt[1:] == "A") & (gt[:-1] == "H"))) + 1
    near = np.zeros(len(gt), bool)
    for s in sw:
        near[max(0, s - 25):s + 26] = True
    play = np.isin(gt, ["H", "A"])
    print(f"crowd switch signal: near true switches (+-1 s) {sig[near].mean():.3f} / elsewhere in play {sig[play & ~near].mean():.3f}; switches {len(sw)}")
    lead, minute, ng = clue_features.score_state(tags, lambda nm: data.tag_team_side(nm, tr.team_names), len(gt), P.fps)
    print(f"goals from tags {ng}; lead values {np.unique(lead)}")
