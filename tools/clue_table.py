"""Table of the MT-26 clue experiments (EVAL-G1 / EVAL-G2): spellF1 and the D / H-A details.

    python tools/clue_table.py --repo ../more-than-a-hack-2027 --runs base d_align ... --dir results/x
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def spells(lab, v):
    m = np.r_[False, lab == v, False].astype(np.int8)
    return int((np.diff(m) == 1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dir", default="results/x")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--games", nargs="+", default=["G1", "G2"])
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.repo, "evaluation"))
    import possession_metrics as pm
    rows = []
    for r in a.runs:
        row = {"run": r}
        f1s = []
        for g in a.games:
            gt_path = f"external/EVAL-{g}/truth/gt.csv"
            pr_path = os.path.join(a.dir, f"{r}_{g}.csv")
            if not os.path.isfile(pr_path):
                continue
            gt = pd.read_csv(gt_path)["label"].astype(str).to_numpy()
            pr = pd.read_csv(pr_path)["label"].astype(str).to_numpy()
            res = pm.evaluate(pm.label_codes(pm.read_label_csv(gt_path)), pm.label_codes(pm.read_label_csv(pr_path)))
            f1s.append(res["spell_f1_mean"])
            graded = gt != "X"
            play = graded & np.isin(gt, ["H", "A"])
            both = play & np.isin(pr, ["H", "A"])
            row[f"F1_{g}"] = round(res["spell_f1_mean"], 4)
            row[f"play->D_{g}"] = round(float(((pr == "D") & play).sum() / graded.sum()), 4)
            row[f"D->play_{g}"] = round(float(((pr != "D") & (gt == "D")).sum() / graded.sum()), 4)
            row[f"nD_{g}"] = spells(pr, "D")
            row[f"HA_{g}"] = round(float((pr[both] == gt[both]).mean()), 4)
            row[f"poss_{g}"] = spells(pr, "H") + spells(pr, "A")
            row["poss_gt_" + g] = spells(gt, "H") + spells(gt, "A")
            row["nD_gt_" + g] = spells(gt, "D")
        row["F1_mean"] = round(float(np.mean(f1s)), 4) if f1s else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index("run")
    pd.set_option("display.width", 250)
    print(df.to_string())
    df.to_csv(os.path.join(a.dir, "clue_table.csv"))


if __name__ == "__main__":
    main()
