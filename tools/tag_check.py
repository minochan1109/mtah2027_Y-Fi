"""How well does each kind of Smart Tagging tag agree with OUR algorithmic pseudo labels?

    python -m possession.v2.pipeline_v2 --repo ../more-than-a-hack-2027 --out results/v2.csv --dump results/v2_detail.csv
    python tools/tag_check.py --repo ../more-than-a-hack-2027 --detail results/v2_detail.csv --out results/tag_check.txt

For every code (and every Type text of SET PIECES) it prints
    n            number of tags
    covered      how many of them overlap frames that have a pseudo team label
    agree        share of those pseudo-labelled frames whose team matches what the tag suggests
                 (attack-type codes: the tagged team; defend-type codes: the other team)
    dead_before  share of the 3 s before the tag start that our estimate calls D (restarts)

Use it to fill the tables in possession/v2/tag_policy.py: high agreement -> keep / raise the
weight, around 0.5 -> weight 0 ("ignore"). Decide per code, never per single tag.
Pseudo labels are made by an algorithm, so using them here is allowed; no hand labels involved.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from possession import data  # noqa: E402
from possession.v2 import tag_policy  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--detail", required=True, help="per-frame CSV from pipeline_v2 --dump")
    ap.add_argument("--game", default="BAR-ATH")
    ap.add_argument("--data-dir")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    repo = os.path.abspath(a.repo)
    sys.path.insert(0, repo)
    import games  # noqa: E402
    data_dir = a.data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(repo, "data")
    paths = games.asset_paths(a.game, data_dir)
    _, names = data._team_of_each_track(paths["atd_meta"])
    tags = tag_policy.load_tags(paths["smart_tagging"])
    d = pd.read_csv(a.detail)
    y = d["pseudo_team"].to_numpy()          # 1 home, 0 away, -1 unknown
    lab = d["label"].astype(str).to_numpy()
    T = len(y)

    stats = defaultdict(lambda: {"n": 0, "covered": 0, "agree": [], "dead_before": []})
    for g in tags:
        types = [t for gr, t in g["labels"] if gr.lower() == "type"]
        key = g["code"] + (f" / {types[0]}" if g["code"] == "SET PIECES" and types else "")
        role, _ = tag_policy.CODE_POLICY.get(g["code"], ("ignore", 0.0))
        st = stats[key]
        st["n"] += 1
        st["role"] = role
        s, e = max(0, g["start"]), min(T, g["end"])
        b0 = max(0, s - 75)
        if s > b0:
            st["dead_before"].append(float((lab[b0:s] == "D").mean()))
        side = data.tag_team_side(g["tags"].get("Team"), names)
        if side is None or e <= s:
            continue
        seg = y[s:e]
        seg = seg[seg >= 0]
        if len(seg) == 0:
            continue
        st["covered"] += 1
        home_share = float((seg == 1).mean())
        suggested_home = (side == 0) if role != "defend" else (side == 1)
        st["agree"].append(home_share if suggested_home else 1 - home_share)

    lines = [f"team mapping: {names[0]} = home(0), {names[1]} = away(1)",
             f"{'code / type':<45}{'role':>9}{'n':>6}{'covered':>9}{'agree':>8}{'dead_before':>13}"]
    for key in sorted(stats, key=lambda k: -stats[k]["n"]):
        st = stats[key]
        ag = f"{np.mean(st['agree']):.2f}" if st["agree"] else "-"
        db = f"{np.mean(st['dead_before']):.2f}" if st["dead_before"] else "-"
        lines.append(f"{key[:44]:<45}{st.get('role', '-'):>9}{st['n']:6d}{st['covered']:9d}{ag:>8}{db:>13}")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
