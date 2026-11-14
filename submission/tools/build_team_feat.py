"""v3 の学習用ファイル（他の試合の特徴量と正解）を、Metrica の公開サンプル試合から作り直す。
これらのファイルはリポジトリに入れない（.gitignore 済み）。毎回このスクリプトで同じものができる。
Rebuild the v3 training files (other games' features + answer keys) from Metrica's public sample games.
They are not committed; this script regenerates exactly the same files every time.

    git clone https://github.com/metrica-sports/sample-data.git ../../sample-data
    python tools/build_team_feat.py

作るもの / writes:
    external/EVAL-G1, EVAL-G2, EVAL-G3       tools/make_eval_data.py（seed 0 / 1 / 2）
    results/team_feat/EVAL-G1.npz, _dead.npz  pipeline_v3 --team-features-out（G1・G2 だけ。G3 は学習に使わない）
    results/team_feat/EVAL-G2.npz, _dead.npz

配布された試合（BAR-ATH）のデータ・正解はいっさい使わない。
No data or labels of the provided game are used.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (EVAL name, sample game folder, seed, extra make_eval_data args) -- the same as the README commands
GAMES = [("EVAL-G1", "Sample_Game_1", 0, []),
         ("EVAL-G2", "Sample_Game_2", 1, []),
         ("EVAL-G3", "Sample_Game_3_csv", 2, ["--stem", "EVAL-G3"])]
TRAIN = ("EVAL-G1", "EVAL-G2")


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=dict(os.environ, PYTHONUTF8="1"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join("..", "..", "more-than-a-hack-2027"),
                    help="公式リポジトリ（評価スクリプトを使う） / the official challenge repository")
    ap.add_argument("--sample-data", default=os.path.join("..", "..", "sample-data", "data"),
                    help="metrica-sports/sample-data の data フォルダ")
    ap.add_argument("--external", default="external", help="EVAL-G1/G2/G3 を作るフォルダ")
    ap.add_argument("--out", default=os.path.join("results", "team_feat"), help="npz を書くフォルダ")
    ap.add_argument("--force", action="store_true", help="すでにあっても作り直す / rebuild existing files")
    a = ap.parse_args()

    sd = a.sample_data if os.path.isabs(a.sample_data) else os.path.join(ROOT, a.sample_data)
    for g in ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3"):
        if not os.path.isdir(os.path.join(sd, g)):
            sys.exit(f"missing {os.path.join(sd, g)}: git clone https://github.com/metrica-sports/sample-data.git")

    # Sample_Game_3 is FIFA EPTS: convert it to the CSV layout of games 1 / 2 first
    g3csv = os.path.join(sd, "Sample_Game_3_csv")
    if a.force or not os.path.isdir(g3csv):
        run([sys.executable, os.path.join("tools", "epts_to_metrica_csv.py"), os.path.join(sd, "Sample_Game_3"), g3csv])

    for name, src, seed, extra in GAMES:
        dst = os.path.join(a.external, name)
        if a.force or not os.path.isfile(os.path.join(ROOT, dst, "game.json")):
            run([sys.executable, os.path.join("tools", "make_eval_data.py"), "--src", os.path.join(sd, src),
                 "--out", dst, "--seed", str(seed), *extra])

    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    for name in TRAIN:
        npz = os.path.join(a.out, f"{name}.npz")
        dead = os.path.join(a.out, f"{name}_dead.npz")
        if a.force or not (os.path.isfile(os.path.join(ROOT, npz)) and os.path.isfile(os.path.join(ROOT, dead))):
            run([sys.executable, "-m", "possession.v3.pipeline_v3", "--repo", a.repo,
                 "--external", os.path.join(a.external, name), "--out", os.path.join(a.out, f"{name}.csv"),
                 "--team-features-out", npz])
    print("done:", ", ".join(os.path.join(a.out, f"{n}{s}.npz") for n in TRAIN for s in ("", "_dead")))


if __name__ == "__main__":
    main()
