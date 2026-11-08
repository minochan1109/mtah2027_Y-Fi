"""MT-24 step 1: how well are H and A decided now (with the D decision fixed)?

    python tools/ha_analysis.py --repo ../more-than-a-hack-2027 \
        --states results/ds/g1_state.pkl results/ds/g2_state.pkl --config params/dead_hybrid.txt \
        --out results/ha/report

States come from pipeline_v2 --external ... --save-state (evaluation games with a truth). D is taken from
the hybrid D detector (params/dead_hybrid.txt) and kept FIXED; only H / A inside the play stretches is
decided, in several ways:
    argmax     every frame: the team with the higher P(home) / P(away)          (no time model)
    hmm_fixed  2-state Viterbi with a constant switch cost (cost_switch_team)
    hmm_tv     2-state Viterbi with the current time-varying switch cost (v2)
    current    the full v2 3-state Viterbi fed with the hybrid D (what dead_situations.py writes)
plus two "what if" rows: the truth D with hmm_tv (how much D still costs), and our D with the truth H/A
inside play (how much H/A costs).

Writes <out>_ja.txt (Japanese). Only OTHER games with a truth are used.
"""
import argparse
import os
import pickle
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from possession.v2.dead_situation import detect_hybrid, read_config  # noqa: E402
from possession.v2.expr import FPS, _runs  # noqa: E402
from possession.v2.situation_features import situation_features  # noqa: E402
from possession.v2.config_v2 import Params2  # noqa: E402


def viterbi2(lp_h, lp_a, switch_cost):
    """Two states (0 = H, 1 = A). lp_*: log probabilities; switch_cost (T,) cost of a switch INTO t."""
    T = len(lp_h)
    back = np.zeros((T, 2), np.int8)
    c = np.array([-lp_h[0], -lp_a[0]])
    for t in range(1, T):
        sc = switch_cost[t]
        stay0, sw0 = c[0], c[1] + sc          # into H
        stay1, sw1 = c[1], c[0] + sc          # into A
        b0 = 0 if stay0 <= sw0 else 1
        b1 = 1 if stay1 <= sw1 else 0
        c = np.array([min(stay0, sw0) - lp_h[t], min(stay1, sw1) - lp_a[t]])
        back[t] = (b0, b1)
    s = int(np.argmin(c))
    path = np.empty(T, np.int8)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    return path


def decode(st, dead, method, P):
    lab = np.full(st["n"], "", dtype="<U1")
    graded = np.zeros(st["n"], bool)
    for _, a, b in st["blocks"]:
        graded[a:b + 1] = True
    lab[graded & dead] = "D"
    ph = np.clip(np.nan_to_num(st["p_home"], nan=0.5), P.prob_floor, 1 - P.prob_floor)
    for s, e in _runs(graded & ~dead):
        if method == "argmax":
            lab[s:e] = np.where(ph[s:e] >= 0.5, "H", "A")
            continue
        sc = np.full(e - s, P.cost_switch_team) if method == "hmm_fixed" else st["switch_cost"][s:e]
        path = viterbi2(np.log(ph[s:e]), np.log(1 - ph[s:e]), sc)
        lab[s:e] = np.where(path == 0, "H", "A")
    return lab


def spells(lab, graded):
    """[(start, end, team)] of H / A runs (graded frames)."""
    out = []
    for k in ("H", "A"):
        for s, e in _runs((lab == k) & graded):
            out.append((s, e, k))
    return sorted(out)


def analyse(st, lab, graded):
    gt = st["gt"]
    play_both = graded & np.isin(gt, ["H", "A"]) & np.isin(lab, ["H", "A"])
    acc = (lab[play_both] == gt[play_both]).mean()
    ours, truth = spells(lab, graded), spells(gt, graded)
    lens_o = np.array([(e - s) / FPS for s, e, _ in ours])
    lens_t = np.array([(e - s) / FPS for s, e, _ in truth])
    # truth switches H<->A with no D in between
    sw_t = [e for (s, e, k), (s2, e2, k2) in zip(truth, truth[1:]) if e == s2 and k != k2]
    sw_o = np.array([e for (s, e, k), (s2, e2, k2) in zip(ours, ours[1:]) if e == s2 and k != k2])
    delays, missed = [], 0
    for t in sw_t:
        if len(sw_o) == 0:
            missed += 1
            continue
        d = sw_o - t
        j = np.argmin(np.abs(d))
        if abs(d[j]) <= 5 * FPS:
            delays.append(d[j] / FPS)
        else:
            missed += 1
    extra = sum(1 for o in sw_o if not any(abs(o - t) <= 5 * FPS for t in sw_t))
    # first team after a stoppage (our play stretches that follow our D)
    restart_ok = restart_n = 0
    for s, e in _runs(graded & np.isin(lab, ["H", "A"])):
        if s == 0 or lab[s - 1] != "D":
            continue
        w = slice(s, min(e, s + 3 * FPS))
        g = gt[w]
        g = g[np.isin(g, ["H", "A"])]
        if len(g) < FPS:
            continue
        restart_n += 1
        restart_ok += (lab[s] == ("H" if (g == "H").mean() >= 0.5 else "A"))
    return {"acc": acc, "n_ours": len(ours), "n_truth": len(truth), "len_ours": np.percentile(lens_o, [10, 50, 90]),
            "len_truth": np.percentile(lens_t, [10, 50, 90]), "short_ours": int((lens_o < 2).sum()),
            "short_truth": int((lens_t < 2).sum()), "sw_truth": len(sw_t), "sw_found": len(delays), "sw_missed": missed,
            "sw_extra": extra, "delay": np.percentile(delays, [25, 50, 75]) if delays else [np.nan] * 3,
            "restart_ok": restart_ok, "restart_n": restart_n}


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(HERE), "params", "dead_hybrid.txt"))
    ap.add_argument("--out", default="results/ha/report")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(os.path.abspath(a.repo), "evaluation"))
    import possession_metrics as pm  # noqa: E402
    from dead_situations import relabel  # noqa: E402
    cfg = read_config(a.config)
    rows = {}
    sig_rows = []
    for p in a.states:
        with open(p, "rb") as fh:
            st = pickle.load(fh)
        if "gt" not in st:
            raise SystemExit(f"{p}: no truth (use evaluation games)")
        P = Params2()
        for k, v in st["params"].items():
            setattr(P, k, v)
        cols = situation_features(st["px"], st["py"], st["team_mat"], st["attack"], st["tags"], st["pitch"], FPS,
                                  st.get("bx"), st.get("by"), st.get("H"), st.get("H_valid"))
        cols["p_dead"] = np.asarray(st["p_dead"], float)
        cols["v2_D"] = (st["labels"] == "D").astype(float)
        dead, _, _, _ = detect_hybrid(cols, st["blocks"], cfg, st.get("tags"))
        graded = np.zeros(st["n"], bool)
        for _, x, y in st["blocks"]:
            graded[x:y + 1] = True
        gt = st["gt"]
        variants = {
            "argmax": decode(st, dead, "argmax", P),
            "hmm_fixed": decode(st, dead, "hmm_fixed", P),
            "hmm_tv": decode(st, dead, "hmm_tv", P),
            "current": relabel(st, dead),
            "truthD+hmm_tv": decode(st, gt == "D", "hmm_tv", P),
        }
        oracle = np.where(dead, "D", np.where(np.isin(gt, ["H", "A"]), gt, "H"))
        oracle[~graded] = ""
        variants["ourD+truthHA"] = oracle
        for name, lab in variants.items():
            res = pm.evaluate(pm.label_codes(gt), pm.label_codes(lab))
            r = analyse(st, lab, graded)
            r["spell"] = res["spell_f1_mean"]
            rows.setdefault(name, []).append(r)
        # the switch signal around the truth switches
        sig = np.clip((1 - st["switch_cost"] / P.cost_switch_team) / max(P.switch_signal_drop, 1e-9), 0, 1)
        truth = spells(gt, graded)
        sw_t = [e for (s, e, k), (s2, e2, k2) in zip(truth, truth[1:]) if e == s2 and k != k2]
        near = np.zeros(st["n"], bool)
        for t in sw_t:
            near[max(0, t - 2 * FPS):t + 2 * FPS] = True
        play = graded & np.isin(gt, ["H", "A"])
        sig_rows.append((st["name"], float(np.mean(sig[near & play])), float(np.mean(sig[~near & play])),
                         float(np.mean(sig[play] > 0.5))))

    L = []
    add = L.append
    add("H と A の判定の今の状態（MT-24 ステップ1）  D は組み合わせ版で固定")
    add("評価用試合: " + ", ".join(os.path.basename(p) for p in a.states))
    add("")
    names = {"argmax": "フレームごとに確率の高い方（時間のモデルなし）", "hmm_fixed": "HMM（切り替えコスト一定）",
             "hmm_tv": "HMM（時間で変わるコスト＝今の v2）", "current": "今のパイプライン（3状態の HMM）",
             "truthD+hmm_tv": "もし D が正解どおりなら（HMM は今の v2）", "ourD+truthHA": "もし H/A が正解どおりなら（D は組み合わせ版）"}
    add(f"{'方法':<40}{'spellF1':>9}{'H/A一致':>9}{'保持数':>12}{'切替見逃し':>11}{'余分な切替':>11}{'再開チーム':>11}")
    for k, rs in rows.items():
        sp = np.mean([r["spell"] for r in rs])
        acc = np.mean([r["acc"] for r in rs])
        n_o = sum(r["n_ours"] for r in rs)
        n_t = sum(r["n_truth"] for r in rs)
        swt = sum(r["sw_truth"] for r in rs)
        miss = sum(r["sw_missed"] for r in rs)
        extra = sum(r["sw_extra"] for r in rs)
        rok, rn = sum(r["restart_ok"] for r in rs), sum(r["restart_n"] for r in rs)
        add(f"{names[k]:<40}{sp:>9.4f}{acc:>9.1%}{n_o:>6}/{n_t:<5}{miss:>5}/{swt:<5}{extra:>11}{rok:>5}/{rn:<5}")
    add("")
    add("見方: 保持数 = 推定/正解（2試合合計）、切替見逃し = 正解の H↔A の切り替わり（間に D なし）のうち")
    add("      ±5秒以内に推定の切り替わりがないもの、余分な切替 = 正解に対応しない推定の切り替わり、")
    add("      再開チーム = 停止のあとのプレーの最初のチームが正しかった数")
    add("")
    r = rows["hmm_tv"]
    add("保持の長さ（秒, p10 / 中央値 / p90）と 2秒未満の保持の数（今の v2 の HMM）")
    for st_r, p in zip(r, a.states):
        add(f"  {os.path.basename(p)}  推定 {np.round(st_r['len_ours'], 1)} 短い {st_r['short_ours']}   "
            f"正解 {np.round(st_r['len_truth'], 1)} 短い {st_r['short_truth']}")
    add("切り替わりの遅れ（推定 − 正解, 秒, 25% / 中央値 / 75%）")
    for st_r, p in zip(r, a.states):
        add(f"  {os.path.basename(p)}  {np.round(st_r['delay'], 1)}")
    add("")
    add("切り替わりの兆候の信号（0〜1）の平均: 正解の切り替わりの ±2秒 / それ以外のプレー中 / 0.5 を超える割合")
    for name, a1, a2, a3 in sig_rows:
        add(f"  {name}  {a1:.3f} / {a2:.3f} / {a3:.1%}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out + "_ja.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
