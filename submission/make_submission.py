"""提出ファイルを1つのコマンドで作る / Make the submission file with one command.

    python make_submission.py

1. 学習用ファイルを作る（Metrica の公開サンプル試合から。すでにあれば飛ばす）
   tools/build_team_feat.py -> external/EVAL-G1..G3, results/team_feat/EVAL-G1/G2(.npz, _dead.npz)
2. v3 で配布された試合（BAR-ATH）の全フレームにラベルを付ける -> results/submission.csv
3. 公式の validate_submission.py で形式を確認する

フォルダの位置（初期値）/ default locations:
    ../../more-than-a-hack-2027        公式リポジトリ（data/ に配布データ）
    ../../sample-data/data             metrica-sports/sample-data
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print("\n+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=dict(os.environ, PYTHONUTF8="1"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.path.join(ROOT, "..", "..", "more-than-a-hack-2027"),
                    help="公式リポジトリ more-than-a-hack-2027 の場所")
    ap.add_argument("--sample-data", default=os.path.join(ROOT, "..", "..", "sample-data", "data"),
                    help="metrica-sports/sample-data の data フォルダ")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "submission.csv"), help="提出ファイル")
    ap.add_argument("--report", default=os.path.join(ROOT, "results", "report.txt"),
                    help="各段階で何をしたかの記録")
    a = ap.parse_args()
    repo, sd = os.path.abspath(a.repo), os.path.abspath(a.sample_data)
    out, report = os.path.abspath(a.out), os.path.abspath(a.report)
    if not os.path.isfile(os.path.join(repo, "evaluation", "validate_submission.py")):
        sys.exit(f"公式リポジトリが見つからない: {repo}  （--repo で場所を指定）")

    feat = [os.path.join("results", "team_feat", f"EVAL-G{g}.npz") for g in (1, 2)]
    run([sys.executable, os.path.join("tools", "build_team_feat.py"), "--repo", repo, "--sample-data", sd])
    run([sys.executable, "-m", "possession.v3.pipeline_v3", "--repo", repo, "--out", out, "--report", report,
         "--team-train", *feat])
    run([sys.executable, os.path.join(repo, "evaluation", "validate_submission.py"), out])
    print(f"\n提出ファイル: {out}")


if __name__ == "__main__":
    main()
