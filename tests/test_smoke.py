"""Smoke test on a small synthetic match (no real data needed).

    MTAH_REPO=/path/to/more-than-a-hack-2027 python -m pytest tests -q
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import synthetic  # noqa: E402
from possession import pipeline  # noqa: E402


def test_pipeline_runs(tmp_path):
    repo = os.environ.get("MTAH_REPO")
    assert repo, "set MTAH_REPO to the more-than-a-hack-2027 clone"
    repo = os.path.abspath(repo)
    assert os.path.isfile(os.path.join(repo, "games.py")), (
        f"MTAH_REPO={repo} に games.py がありません。more-than-a-hack-2027 フォルダを指定してください")
    out = str(tmp_path)
    synthetic.write_bundle(out, T=12000, blocks=((1, 200, 5800), (2, 6200, 11800)))
    for mode in ("rules", "learned"):
        csv = os.path.join(out, f"{mode}.csv")
        pipeline.main(["--repo", repo, "--data-dir", os.path.join(out, "data"), "--mode", mode,
                       "--out", csv, "--cache-dir", os.path.join(out, "cache"),
                       "--n-frames", "12000", "--blocks", "[[1,200,5800],[2,6200,11800]]"])
        sub = pd.read_csv(csv)
        assert len(sub) == 12000
        assert list(sub["frame"]) == list(range(12000))
        assert set(sub["label"]) <= {"H", "A", "D", "X"}
        assert (sub["label"].iloc[200:5801] != "X").all()


def test_pipeline_v2_runs(tmp_path):
    from possession.v2 import pipeline_v2
    repo = os.path.abspath(os.environ.get("MTAH_REPO", ""))
    assert os.path.isfile(os.path.join(repo, "games.py")), "set MTAH_REPO to the more-than-a-hack-2027 clone"
    out = str(tmp_path)
    synthetic.write_bundle(out, T=12000, blocks=((1, 200, 5800), (2, 6200, 11800)))
    csv = os.path.join(out, "v2.csv")
    pipeline_v2.main(["--repo", repo, "--data-dir", os.path.join(out, "data"), "--out", csv,
                      "--cache-dir", os.path.join(out, "cache"), "--n-frames", "12000",
                      "--blocks", "[[1,200,5800],[2,6200,11800]]"])
    sub = pd.read_csv(csv)
    assert len(sub) == 12000 and set(sub["label"]) <= {"H", "A", "D", "X"}
