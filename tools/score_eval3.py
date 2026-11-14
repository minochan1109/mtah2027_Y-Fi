"""v3 を評価用の 3 試合（EVAL-G1 / G2 / G3）で動かして spellF1 を出す（3 試合を同時に実行）。
チームモデルと D モデルは results/team_feat/ の G1・G2 だけで学習する。G1・G2 は自分の試合を除いて学習し、
G3 は学習に一切使わない（点数を測るだけ）。
Run v3 on EVAL-G1/G2/G3 in parallel and print spellF1. Models learn from the G1/G2 files in
results/team_feat/ only (leave-one-game-out for G1/G2; G3 is never used for training).

    python tools/score_eval3.py --repo ../more-than-a-hack-2027
    python tools/score_eval3.py --repo ../more-than-a-hack-2027 --params params/my_try.json --games 3

出力: results/eval3/EVAL-G<n>.csv（提出形式）と .log（最後に spellF1 が出る）
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join("..", "more-than-a-hack-2027"))
    ap.add_argument("--external", default="external", help="EVAL-G1 / G2 / G3 のあるフォルダ")
    ap.add_argument("--games", nargs="+", default=["1", "2", "3"])
    ap.add_argument("--params", default=None, help="設定を上書きする JSON（任意）")
    ap.add_argument("--out", default=os.path.join("results", "eval3"))
    a = ap.parse_args()

    train = [os.path.join("results", "team_feat", f"EVAL-G{g}.npz") for g in (1, 2)]
    for f in train:
        if not os.path.isfile(os.path.join(ROOT, f)):
            sys.exit(f"missing {f} (先に python tools/build_team_feat.py --repo {a.repo} を実行)")
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1")
    procs = []
    for g in a.games:
        ext = os.path.join(a.external, f"EVAL-G{g}")
        if not os.path.isdir(os.path.join(ROOT, ext)):
            sys.exit(f"missing {ext} (README の「評価用データを作る」を参照)")
        cmd = [sys.executable, "-m", "possession.v3.pipeline_v3", "--repo", a.repo, "--external", ext,
               "--out", os.path.join(a.out, f"EVAL-G{g}.csv"), "--team-train", *train]
        if a.params:
            cmd += ["--params", a.params]
        log = open(os.path.join(ROOT, a.out, f"EVAL-G{g}.log"), "w", encoding="utf-8")
        procs.append((g, subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    for g, p, log in procs:
        p.wait()
        log.close()
        text = open(os.path.join(ROOT, a.out, f"EVAL-G{g}.log"), encoding="utf-8").read()
        m = re.search(r"spellF1=([\d.]+)", text)
        print(f"EVAL-G{g}: spellF1 = {m.group(1) if m else 'ERROR (see ' + a.out + f'/EVAL-G{g}.log)'}")


if __name__ == "__main__":
    main()
