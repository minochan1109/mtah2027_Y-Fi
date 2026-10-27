"""Score version 1 and version 2 (or two settings) on other games, side by side.

    python tools/compare_external.py --repo ../more-than-a-hack-2027 --games external/MET-G1 external/MET-G2 \
        external/MET-G1F external/MET-G2F
    python tools/compare_external.py --repo ../more-than-a-hack-2027 --games external/MET-G1F \
        --versions v2 v2:params/ball_strict.json

--versions entries: "v1", "v2", or "v2:<params.json>" / "v1:<params.json>".
Games ending in F were converted with --fake (false balls, spikes, goal-post object, keeper swaps).
This is the place to decide whether a change is better: it uses only OTHER games.
"""
import argparse
import os
import re
import subprocess
import sys

MODULES = {"v1": "possession.pipeline", "v2": "possession.v2.pipeline_v2"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--versions", nargs="+", default=["v1", "v2"])
    ap.add_argument("--out-dir", default="results/compare")
    ap.add_argument("--cache-dir", default="cache")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for ver in a.versions:
        base, _, params = ver.partition(":")
        scores = []
        for g in a.games:
            tag = f"{ver.replace(':', '_').replace('/', '_').replace(os.sep, '_')}_{os.path.basename(os.path.normpath(g))}"
            cmd = [sys.executable, "-m", MODULES[base], "--repo", a.repo, "--external", g,
                   "--out", os.path.join(a.out_dir, tag + ".csv"), "--cache-dir", a.cache_dir]
            if params:
                cmd += ["--params", params]
            print("running", " ".join(cmd), flush=True)
            res = subprocess.run(cmd, capture_output=True, text=True)
            with open(os.path.join(a.out_dir, tag + ".log"), "w", encoding="utf-8") as fh:
                fh.write(res.stdout + res.stderr)
            m = re.search(r"spellF1=([0-9.]+)", res.stdout)
            scores.append(float(m.group(1)) if m else float("nan"))
            print(f"   spellF1 = {scores[-1]:.4f}" if m else "   failed, see the log", flush=True)
        rows.append((ver, scores))
    names = [os.path.basename(os.path.normpath(g)) for g in a.games]
    print("\nspellF1 on other games (higher is better)")
    print(f"{'version':<30}" + "".join(f"{n:>12}" for n in names) + f"{'mean':>10}")
    for ver, sc in rows:
        ok = [s for s in sc if s == s]
        print(f"{ver:<30}" + "".join(f"{s:12.4f}" for s in sc) + f"{(sum(ok) / len(ok) if ok else float('nan')):10.4f}")


if __name__ == "__main__":
    main()
