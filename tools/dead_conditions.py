"""Which conditions happen ONLY when play is stopped (D)?  And which combination of D algorithms works best?

Step 1 - save the state of each evaluation game (games with a truth, from tools/make_eval_data.py):
    python -m possession.v2.pipeline_v2 --repo ../more-than-a-hack-2027 --external external/EVAL-G1 \
        --out results/dc/g1.csv --save-state results/dc/g1_state.pkl
    python -m possession.v2.pipeline_v2 --repo ../more-than-a-hack-2027 --external external/EVAL-G2 \
        --out results/dc/g2.csv --save-state results/dc/g2_state.pkl

Step 2 - analyse (writes an English and a Japanese report with the same numbers):
    python tools/dead_conditions.py --repo ../more-than-a-hack-2027 \
        --states results/dc/g1_state.pkl results/dc/g2_state.pkl \
        --conditions params/dead_conditions.txt --algorithms params/dead_algorithms.txt --out results/dc/report

    -> results/dc/report_en.txt, results/dc/report_ja.txt

What it does
  A. conditions: every candidate condition is measured on the truth
       D cover      share of real stoppage frames where the condition holds        (higher = catches more D)
       play overlap share of play frames where it ALSO holds                       (lower = safer)
       precision    share of frames with the condition that are really D
     candidates = automatic ones (every feature at several thresholds, both directions) + yours
     (--conditions). Lists "D-only" conditions (play overlap <= 0 / 1 / 2 / 5 %) and the best union of
     them, and the mirror image: "play-only" conditions.
  B. algorithms (--algorithms): each algorithm turns conditions into D stretches (start / keep / end /
     minimum length). All combinations (union of their D) are measured on the D decision, and the best
     ones are run through the state model and scored with spellF1.

Expressions (conditions file and algorithms file) use the column names listed at the end of the report,
numbers, and / or / not, < <= > >= == !=, + - * /, abs(x), isnan(x), and time helpers:
    lasting(x, s)  x has been true for the last s seconds        recent(x, s)  x was true within the last s seconds
    ahead(x, s)    x becomes true within the next s seconds      MODEL         the current v2 D decision
Only OTHER games with a truth are used here; nothing labels the challenge game.
"""
import argparse
import itertools
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

FPS = 25
LIMITS = (0.0, 0.01, 0.02, 0.05)

# ------------------------------------------------------------------ feature descriptions (en, ja)
DESC = {
    "speed_mean": ("mean player speed (m/s)", "全選手の平均速度（m/s）"),
    "speed_std": ("spread of player speeds", "選手の速さのばらつき"),
    "speed_mean_home": ("mean speed, home team", "ホームの平均速度"),
    "speed_mean_away": ("mean speed, away team", "アウェイの平均速度"),
    "frac_walking": ("share of players walking", "歩いている選手の割合"),
    "n_sprint": ("number of players sprinting", "全力で走っている選手の数"),
    "n_tracked": ("number of players tracked", "追跡できている選手の数"),
    "ball_seen": ("ball detected (1/0)", "ボールが検出されている（1/0）"),
    "ball_speed": ("ball speed (m/s)", "ボールの速さ（m/s）"),
    "since_seen": ("seconds since the ball was last seen", "ボールが最後に見えてからの秒数"),
    "until_seen": ("seconds until the ball is seen again", "次にボールが見えるまでの秒数"),
    "last_ball_out_dist": ("how far outside the pitch the last ball was (m, >0 = outside)",
                           "最後のボールがピッチの外にどれだけ出ていたか（m、正なら外）"),
    "last_ball_out": ("last ball seen outside the pitch (1/0)", "最後に見えたボールがピッチの外（1/0）"),
    "last_ball_line_dist": ("distance of the last ball to the nearest line (m)", "最後のボールから一番近いラインまでの距離（m）"),
    "cam_speed": ("camera movement speed (m/s on the pitch)", "カメラの動く速さ（ピッチ上の m/s）"),
    "cam_off_centre": ("camera aim distance from the centre spot (m)", "カメラの向き先とセンターの距離（m）"),
    "frac_own_half": ("share of players in their own half", "自陣にいる選手の割合"),
    "centroid_gap": ("distance between the two team centroids (m, along the pitch)", "両チームの重心の距離（m、縦方向）"),
    "to_set_piece_s": ("seconds until the next SET PIECES tag", "次のセットプレーのタグまでの秒数"),
    "in_set_piece_tag": ("inside a SET PIECES tag (1/0)", "セットプレーのタグの中（1/0）"),
    "ball_conf": ("ball confidence 0-1 (v2)", "ボールの信頼度 0〜1（v2）"),
    "ball_flight": ("ball probably in the air (1/0)", "ボールが空中らしい（1/0）"),
    "tag_to_restart_s": ("seconds until the next restart tag (tag policy)", "次の再開タグまでの秒数（タグの使い方の表）"),
    "tag_since_restart_s": ("seconds since the last restart tag", "直前の再開タグからの秒数"),
    "tag_since_goal_s": ("seconds since the last goal tag", "直前のゴールのタグからの秒数"),
    "ball_conf_p1": ("ball confidence, mean of the past 1 s", "ボールの信頼度、過去1秒の平均"),
    "ball_conf_f1": ("ball confidence, mean of the next 1 s", "ボールの信頼度、未来1秒の平均"),
    "p_dead": ("current P(dead) of the v2 dead model", "今の v2 の D のモデルの P(dead)"),
    "model_D": ("current v2 final label is D (1/0)", "今の v2 の最終ラベルが D（1/0）"),
}
VIEW = {"r_own_": ("home ", "ホームの"), "r_opp_": ("away ", "アウェイの"),
        "k_own_": ("home ", "ホームの"), "k_opp_": ("away ", "アウェイの")}
VDESC = {"n_recover": ("players running back fast to their own goal", "自陣ゴールへ速く戻る選手の数"),
         "recover_onset": ("players who just started running back", "戻り始めたばかりの選手の数"),
         "cvx_change": ("change of forward speed of the team centroid over 1 s", "重心の前進速度の1秒間の変化"),
         "frac_opp_inside": ("share of opponents inside the team box", "チームの枠の中にいる相手の割合"),
         "box_area": ("area of the team box (m2)", "チームの枠の面積（m²）"),
         "box_length": ("length of the team box (m)", "チームの枠の縦の長さ（m）"),
         "box_width": ("width of the team box (m)", "チームの枠の横の幅（m）")}


def describe(col):
    m = re.match(r"(.+)_([pfd])(\d+(?:\.\d+)?)$", col)
    win = None
    base = col
    if m and m.group(1) in DESC:
        base, kind, w = m.group(1), m.group(2), m.group(3)
        win = {"p": (f"mean of the past {w} s of ", f"過去{w}秒の平均："),
               "f": (f"mean of the next {w} s of ", f"未来{w}秒の平均："),
               "d": (f"next {w} s minus past {w} s of ", f"未来{w}秒−過去{w}秒の差：")}[kind]
    if base in DESC:
        en, ja = DESC[base]
    else:
        en, ja = col, col
        for pre, (pe, pj) in VIEW.items():
            if col.startswith(pre) and col[len(pre):] in VDESC:
                ve, vj = VDESC[col[len(pre):]]
                en, ja = pe + ve, pj + vj
    if win:
        en, ja = win[0] + en, win[1] + ja
    return en, ja


# ------------------------------------------------------------------ time helpers (per game)
def _runs(mask):
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def lasting(x, s):
    x = np.asarray(x, bool)
    k = int(round(s * FPS))
    out = np.zeros(len(x), bool)
    for a, b in _runs(x):
        out[min(b, a + k):b] = True
    return out


def recent(x, s):
    x = np.asarray(x, bool).astype(int)
    k = int(round(s * FPS))
    c = np.r_[0, np.cumsum(x)]
    t = np.arange(len(x))
    return (c[t + 1] - c[np.clip(t - k, 0, None)]) > 0


def ahead(x, s):
    x = np.asarray(x, bool).astype(int)
    k = int(round(s * FPS))
    c = np.r_[0, np.cumsum(x)]
    t = np.arange(len(x))
    return (c[np.clip(t + k + 1, None, len(x))] - c[t]) > 0


def evaluate_expr(expr, cols):
    """Evaluate one expression on one game's columns (dict of numpy arrays). Only the small language
    described at the top is allowed (no other Python)."""
    import ast
    import operator as op
    funcs = {"lasting": lasting, "recent": recent, "ahead": ahead, "abs": np.abs, "isnan": np.isnan}
    cmp = {ast.Lt: op.lt, ast.LtE: op.le, ast.Gt: op.gt, ast.GtE: op.ge, ast.Eq: op.eq, ast.NotEq: op.ne}
    arith = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.BoolOp):
            vals = [np.asarray(ev(v), bool) for v in n.values]
            return np.logical_and.reduce(vals) if isinstance(n.op, ast.And) else np.logical_or.reduce(vals)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return ~np.asarray(ev(n.operand), bool)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.Compare):
            left, res = ev(n.left), None
            for o, c in zip(n.ops, n.comparators):
                right = ev(c)
                with np.errstate(invalid="ignore"):
                    r = cmp[type(o)](left, right)
                res = r if res is None else (res & r)
                left = right
            return res
        if isinstance(n, ast.BinOp) and type(n.op) in arith:
            return arith[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs:
            return funcs[n.func.id](*[ev(x) for x in n.args])
        if isinstance(n, ast.Name):
            if n.id not in cols:
                raise ValueError(f"unknown column '{n.id}'")
            return cols[n.id]
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        raise ValueError(f"not allowed here: {ast.dump(n)[:60]}")

    v = np.asarray(ev(ast.parse(expr, mode="eval")))
    if v.dtype != bool:
        v = np.nan_to_num(v.astype(float)) != 0
    return v


# ------------------------------------------------------------------ data
def safe(c):
    """Column name usable in expressions: 'speed_mean_p0.5' -> 'speed_mean_p0_5'."""
    return c.replace(".", "_")


class Games:
    def __init__(self, paths, exclude_suspect_s):
        self.games = []
        for p in paths:
            with open(p, "rb") as fh:
                st = pickle.load(fh)
            if "gt" not in st:
                raise SystemExit(f"{p}: no truth inside. Save states of evaluation games (--external).")
            fd = st["fd"].copy()
            for c in st["home_view"].columns:
                fd[c] = st["home_view"][c].values
            fd["p_dead"] = st["p_dead"]
            fd["model_D"] = (st["labels"] == "D").astype(float)
            st["fd"] = fd
            graded = np.zeros(st["n"], bool)
            for _, a, b in st["blocks"]:
                graded[a:b + 1] = True
            gt = st["gt"]
            dead, play = (gt == "D"), np.isin(gt, ["H", "A"])
            use = graded & (dead | play)
            suspect = np.zeros(st["n"], bool)
            if exclude_suspect_s > 0 and "true_ball_seen" in st:
                for a, b in _runs(play & ~st["true_ball_seen"]):
                    if (b - a) / FPS >= exclude_suspect_s:
                        suspect[a:b] = True
            st.update({"cols": {safe(c): fd[c].to_numpy(dtype=float) for c in fd.columns},
                       "use": use & ~suspect, "dead": dead, "play": play, "suspect": suspect, "graded": graded})
            self.games.append(st)
        self.columns = list(self.games[0]["cols"].keys())
        self.orig = {safe(c): c for c in self.games[0]["fd"].columns}
        for c in self.games[0]["home_view"].columns:
            self.orig[safe(c)] = c

    def mask(self, expr):
        if expr.strip() == "MODEL":
            return [g["labels"] == "D" for g in self.games]
        return [evaluate_expr(expr, g["cols"]) for g in self.games]

    def stack(self, masks, key):
        return np.concatenate([m[g[key] if key != "all" else slice(None)] for m, g in zip(masks, self.games)])


def cond_metrics(G, masks):
    dc = pl = pr = 0.0
    nd = npl = nm = tp = 0
    touched = total_d = false_runs = 0
    hours = 0.0
    for m, g in zip(masks, G.games):
        u = g["use"]
        d, p = g["dead"] & u, g["play"] & u
        nd += d.sum(); npl += p.sum(); tp += (m & d).sum(); nm += (m & u).sum()
        dc += (m & d).sum(); pl += (m & p).sum()
        for a, b in _runs(d):
            total_d += 1
            touched += m[a:b].any()
        for a, b in _runs(m & u):
            false_runs += p[a:b].mean() > 0.5
        hours += u.sum() / FPS / 3600
    return {"d_cover": dc / max(nd, 1), "play_overlap": pl / max(npl, 1), "precision": tp / max(nm, 1),
            "d_stretch_touched": touched / max(total_d, 1), "false_runs_per_h": false_runs / max(hours, 1e-9)}


def auto_conditions(G):
    out = []
    qs = (5, 10, 25, 50, 75, 90, 95)
    for c in G.columns:
        vals = np.concatenate([g["cols"][c][g["use"]] for g in G.games])
        fin = vals[np.isfinite(vals)]
        if len(fin) < 100 or np.nanstd(fin) == 0:
            continue
        uniq = np.unique(fin)
        if len(uniq) <= 2:
            for v in uniq:
                out.append((f"{c} == {v:g}", c, "==", v))
        else:
            seen = set()
            for q in qs:
                v = float(np.percentile(fin, q))
                v = float(f"{v:.3g}")
                if v in seen:
                    continue
                seen.add(v)
                out.append((f"{c} <= {v:g}", c, "<=", v))
                out.append((f"{c} >= {v:g}", c, ">=", v))
        if np.isnan(vals).mean() > 0.01:
            out.append((f"isnan({c})", c, "nan", None))
    return out


def auto_label(c, op, v, lang, orig=None):
    en, ja = describe((orig or {}).get(c, c))
    if op == "nan":
        return f"{en} is unknown" if lang == "en" else f"{ja}が不明"
    if op == "==":
        return f"{en} = {v:g}" if lang == "en" else f"{ja} = {v:g}"
    word = {"<=": ("at most", "以下"), ">=": ("at least", "以上")}[op]
    return f"{en} {word[0]} {v:g}" if lang == "en" else f"{ja}が {v:g} {word[1]}"


# ------------------------------------------------------------------ files
def read_conditions(path):
    """Lines:  dead | 日本語の名前 | English name | expression      (or 'play' instead of 'dead')"""
    out = []
    if not path:
        return out
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 4 or parts[0] not in ("dead", "play"):
                raise SystemExit(f"{path}:{ln}: write  dead|play | 日本語名 | English name | expression")
            out.append({"kind": parts[0], "ja": parts[1], "en": parts[2], "expr": parts[3]})
    return out


def read_algorithms(path):
    """Blocks:
        [日本語の名前 | English name]
        start = expression          (D begins when this is true)
        keep  = expression          (optional: D continues while this is true; default = start)
        end   = expression          (optional: D ends when this is true)
        min_s = 2                   (optional: drop D stretches shorter than this)
    """
    algs = []
    if not path:
        return algs
    cur = None
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                names = [x.strip() for x in line[1:-1].split("|")]
                cur = {"ja": names[0], "en": names[-1], "start": None, "keep": None, "end": None, "min_s": 0.0}
                algs.append(cur)
                continue
            if cur is None or "=" not in line:
                raise SystemExit(f"{path}:{ln}: expected [name] or key = value")
            k, v = [x.strip() for x in line.split("=", 1)]
            if k not in ("start", "keep", "end", "min_s"):
                raise SystemExit(f"{path}:{ln}: unknown key {k}")
            cur[k] = float(v) if k == "min_s" else v
    for a in algs:
        if not a["start"]:
            raise SystemExit(f"{path}: algorithm {a['en']} has no start")
    return algs


def run_algorithm(G, alg):
    if alg["start"].strip() == "MODEL":
        return G.mask("MODEL")
    starts = G.mask(alg["start"])
    keeps = G.mask(alg["keep"]) if alg["keep"] else starts
    ends = G.mask(alg["end"]) if alg["end"] else [np.zeros_like(s) for s in starts]
    out = []
    for s, k, e in zip(starts, keeps, ends):
        m = np.zeros(len(s), bool)
        on = False
        for t in range(len(s)):
            if not on:
                on = bool(s[t])
            elif e[t] or not k[t]:
                on = False
            m[t] = on
        if alg["min_s"] > 0:
            for a, b in _runs(m):
                if (b - a) < alg["min_s"] * FPS:
                    m[a:b] = False
        out.append(m)
    return out


def d_metrics(G, masks):
    nd = npl = tp = fp = 0
    stretches = short = real = hit = 0
    for m, g in zip(masks, G.games):
        u = g["use"]
        d, p = g["dead"] & u, g["play"] & u
        nd += d.sum(); npl += p.sum(); tp += (m & d).sum(); fp += (m & p).sum()
        for _, a, b in g["blocks"]:
            for s, e in _runs(m[a:b + 1]):
                stretches += 1
                short += (e - s) < 3 * FPS
            for s, e in _runs(g["dead"][a:b + 1]):
                real += 1
                hit += m[a + s:a + e].mean() >= 0.5
    rec = tp / max(nd, 1)
    prec = tp / max(tp + fp, 1)
    return {"d_recall": rec, "play_to_d": fp / max(npl, 1), "d_precision": prec,
            "d_f1": 2 * rec * prec / max(rec + prec, 1e-9), "d_stretches": stretches, "short": short,
            "real_stretches": real, "real_hit": hit / max(real, 1)}


def spell_f1(G, masks, repo):
    sys.path.insert(0, os.path.join(repo, "evaluation"))
    import possession_metrics as pm  # noqa: E402
    from possession import hmm, postprocess  # noqa: E402
    from possession.v2 import hmm_v2  # noqa: E402
    from possession.v2.config_v2 import Params2  # noqa: E402
    scores = []
    for m, g in zip(masks, G.games):
        P = Params2()
        for k, v in g["params"].items():
            setattr(P, k, v)
        p_dead = np.where(m, 0.999, 0.001)
        logE = hmm.emissions(p_dead, g["p_home"], P)
        states = [hmm_v2.viterbi_tv(logE[a:b + 1], P, g["switch_cost"][a:b + 1]) for _, a, b in g["blocks"]]
        labels = postprocess.to_labels(states, g["n"], g["blocks"])
        res = pm.evaluate(pm.label_codes(g["gt"]), pm.label_codes(labels))
        scores.append(res["spell_f1_mean"])
    return scores


# ------------------------------------------------------------------ report
class Report:
    def __init__(self):
        self.en, self.ja = [], []

    def add(self, en, ja=None):
        self.en.append(en)
        self.ja.append(en if ja is None else ja)

    def write(self, prefix):
        os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".", exist_ok=True)
        for lang, lines in (("en", self.en), ("ja", self.ja)):
            with open(f"{prefix}_{lang}.txt", "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")


def pct(x):
    return f"{100 * x:5.1f}%"


def main(argv=None):
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--states", nargs="+", required=True, help="state pickles from pipeline_v2 --save-state")
    ap.add_argument("--conditions", help="your conditions file (see params/dead_conditions_example.txt)")
    ap.add_argument("--algorithms", help="your algorithms file (see params/dead_algorithms_example.txt)")
    ap.add_argument("--out", required=True, help="output prefix -> <out>_en.txt and <out>_ja.txt")
    ap.add_argument("--no-auto", action="store_true", help="skip the automatic conditions")
    ap.add_argument("--exclude-suspect-s", type=float, default=10.0,
                    help="leave out truth 'play' stretches with no true ball for this long (likely truth errors). 0 = keep all")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--spell-top", type=int, default=5, help="how many algorithm combinations get a spellF1 run")
    a = ap.parse_args(argv)

    G = Games(a.states, a.exclude_suspect_s)
    R = Report()
    n_use = sum(g["use"].sum() for g in G.games)
    n_sus = sum(g["suspect"].sum() for g in G.games)
    R.add("DEAD-BALL CONDITIONS AND ALGORITHMS", "プレー停止（D）の条件とアルゴリズムの分析")
    R.add(f"games: {', '.join(g['name'] for g in G.games)}   frames used: {n_use}   "
          f"left out as likely truth errors: {n_sus} ({n_sus / FPS:.0f} s)",
          f"試合: {', '.join(g['name'] for g in G.games)}   使ったフレーム: {n_use}   "
          f"正解の誤りの疑いで除いたフレーム: {n_sus}（{n_sus / FPS:.0f} 秒）")
    R.add("D cover = share of real stoppage frames where the condition holds. play overlap = share of play frames where it also holds.",
          "停止の拾い = 本当の停止のフレームのうち条件が成り立つ割合。プレー重なり = プレー中のフレームのうち条件が成り立ってしまう割合。")
    R.add("precision = share of condition frames that are really D. D touched = share of real stoppages with at least one frame. false runs/h = condition stretches that are mostly play, per hour.",
          "適合 = 条件が成り立つフレームのうち本当に停止の割合。停止に触れる = 本当の停止のうち1フレーム以上で成り立つ割合。誤りの塊/時 = 大半がプレー中の、条件の連続区間の1時間あたりの数。")
    R.add("")

    # ---------------- A. conditions
    cands = []
    if not a.no_auto:
        for expr, c, op, v in auto_conditions(G):
            cands.append({"src": "auto", "expr": expr, "en": auto_label(c, op, v, "en", G.orig), "ja": auto_label(c, op, v, "ja", G.orig)})
    for c in read_conditions(a.conditions):
        cands.append({"src": f"yours ({c['kind']})", "expr": c["expr"], "en": c["en"], "ja": c["ja"]})
    masks = {}
    for c in cands:
        try:
            masks[c["expr"]] = G.mask(c["expr"])
        except Exception as e:  # noqa: BLE001
            R.add(f"!! cannot evaluate '{c['expr']}': {e}", f"!! 計算できない条件 '{c['expr']}'：{e}")
            continue
        c.update(cond_metrics(G, masks[c["expr"]]))
    cands = [c for c in cands if "d_cover" in c]
    R.add(f"A. CONDITIONS: {len(cands)} candidates (automatic + yours)", f"A. 条件：候補 {len(cands)} 個（自動 + あなたの条件）")
    head_en = f"   {'D cover':>8} {'play ovl':>8} {'precision':>9} {'D touch':>8} {'false/h':>8}  condition"
    head_ja = f"   {'停止の拾い':>6} {'プレー重なり':>6} {'適合':>7} {'停止に触れる':>6} {'誤りの塊/時':>6}  条件"

    def row(c, lang):
        return (f"   {pct(c['d_cover']):>8} {pct(c['play_overlap']):>8} {pct(c['precision']):>9} "
                f"{pct(c['d_stretch_touched']):>8} {c['false_runs_per_h']:8.1f}  {c[lang]}   [{c['expr']}]")

    for lim in LIMITS:
        ok = sorted([c for c in cands if c["play_overlap"] <= lim], key=lambda c: -c["d_cover"])
        R.add("")
        R.add(f"A-{lim:.0%} D-only conditions: play overlap <= {lim:.0%}  ({len(ok)} found, top {a.top} by D cover)",
              f"A-{lim:.0%} 停止にしか起きない条件：プレー重なり {lim:.0%} 以下（{len(ok)} 個、停止の拾いの上位 {a.top}）")
        R.en.append(head_en); R.ja.append(head_ja)
        for c in ok[:a.top]:
            R.add(row(c, "en"), row(c, "ja"))
        # greedy union under the same overlap limit
        chosen, union = [], None
        for _ in range(8):
            best, best_gain = None, 0.0
            for c in ok[:200]:
                if c in chosen:
                    continue
                m = masks[c["expr"]] if union is None else [u | x for u, x in zip(union, masks[c["expr"]])]
                met = cond_metrics(G, m)
                if met["play_overlap"] > lim:
                    continue
                gain = met["d_cover"] - (cond_metrics(G, union)["d_cover"] if union is not None else 0.0)
                if gain > best_gain + 1e-4:
                    best, best_gain, best_m, best_met = c, gain, m, met
            if best is None:
                break
            chosen.append(best)
            union = best_m
            R.add(f"      + {best['en']}  -> union: D cover {pct(best_met['d_cover'])}, play overlap {pct(best_met['play_overlap'])}, "
                  f"precision {pct(best_met['precision'])}",
                  f"      + {best['ja']}  → 合わせて：停止の拾い {pct(best_met['d_cover'])}、プレー重なり {pct(best_met['play_overlap'])}、"
                  f"適合 {pct(best_met['precision'])}")
    # mirror: play-only conditions
    R.add("")
    R.add("A-play. PLAY-only conditions: hold in play but almost never in a stoppage (D overlap <= 1%), top by play cover",
          "A-play. プレー中にしか起きない条件：停止中にはほとんど成り立たない（停止との重なり 1% 以下）、プレーの拾いの上位")
    for c in cands:
        ms = masks[c["expr"]]
        nd = npl = cd = cp = 0
        for m, g in zip(ms, G.games):
            d, p = g["dead"] & g["use"], g["play"] & g["use"]
            nd += d.sum(); npl += p.sum(); cd += (m & d).sum(); cp += (m & p).sum()
        c["play_cover"], c["d_overlap"] = cp / max(npl, 1), cd / max(nd, 1)
    okp = sorted([c for c in cands if c["d_overlap"] <= 0.01], key=lambda c: -c["play_cover"])
    R.add(f"   {'play cover':>10} {'D overlap':>9}  condition", f"   {'プレーの拾い':>6} {'停止との重なり':>6}  条件")
    for c in okp[:a.top]:
        R.add(f"   {pct(c['play_cover']):>10} {pct(c['d_overlap']):>9}  {c['en']}   [{c['expr']}]",
              f"   {pct(c['play_cover']):>10} {pct(c['d_overlap']):>9}  {c['ja']}   [{c['expr']}]")

    # ---------------- B. algorithms
    algs = read_algorithms(a.algorithms)
    if not any(x["start"].strip() == "MODEL" for x in algs):
        algs.insert(0, {"ja": "今の v2（基準）", "en": "current v2 (baseline)", "start": "MODEL", "keep": None, "end": None, "min_s": 0})
    R.add("")
    R.add(f"B. ALGORITHMS: {len(algs)} (each one alone, then combinations = union of their D)",
          f"B. アルゴリズム：{len(algs)} 個（1つずつ、次に組み合わせ＝それぞれの D を合わせたもの）")
    amasks = []
    for alg in algs:
        try:
            amasks.append(run_algorithm(G, alg))
        except Exception as e:  # noqa: BLE001
            R.add(f"!! cannot run algorithm {alg['en']}: {e}", f"!! アルゴリズム {alg['ja']} を実行できない：{e}")
            amasks.append(None)
    idx = [i for i, m in enumerate(amasks) if m is not None]
    combos = []
    if len(idx) <= 10:
        for r in range(1, len(idx) + 1):
            combos += list(itertools.combinations(idx, r))
    else:
        combos = [(i,) for i in idx]
    results = []
    for cb in combos:
        m = [np.logical_or.reduce([amasks[i][k] for i in cb]) for k in range(len(G.games))]
        results.append((cb, m, d_metrics(G, m)))
    results.sort(key=lambda r: -r[2]["d_f1"])
    real = results[0][2]["real_stretches"] if results else 0
    R.add(f"   real stoppages in the truth: {real}", f"   正解の本当の停止の数：{real}")
    R.add(f"   {'D F1':>6} {'D recall':>8} {'play->D':>8} {'#D':>5} {'<3s':>5} {'hit':>6}  combination",
          f"   {'D F1':>6} {'停止の拾い':>6} {'プレー→D':>6} {'D回数':>4} {'3秒未満':>4} {'停止を当てた':>5}  組み合わせ")
    for cb, _, mt in results[:30]:
        en = " + ".join(algs[i]["en"] for i in cb)
        ja = " + ".join(algs[i]["ja"] for i in cb)
        line = (f"   {mt['d_f1']:6.3f} {pct(mt['d_recall']):>8} {pct(mt['play_to_d']):>8} {mt['d_stretches']:5d} "
                f"{mt['short']:5d} {pct(mt['real_hit']):>6}  ")
        R.add(line + en, line + ja)

    # spellF1 for the best combinations and every single algorithm
    R.add("")
    R.add("B-spell. spellF1 (the ranking score) when the D of the combination replaces the model's D",
          "B-spell. 組み合わせの D をモデルの D の代わりに使ったときの spellF1（順位を決める点数）")
    todo = [r for r in results[:a.spell_top]] + [r for r in results if len(r[0]) == 1 and r not in results[:a.spell_top]]
    for cb, m, mt in todo:
        sc = spell_f1(G, m, os.path.abspath(a.repo))
        en = " + ".join(algs[i]["en"] for i in cb)
        ja = " + ".join(algs[i]["ja"] for i in cb)
        s = "  ".join(f"{g['name']} {x:.4f}" for g, x in zip(G.games, sc))
        R.add(f"   mean {np.mean(sc):.4f}  ({s})  {en}", f"   平均 {np.mean(sc):.4f}  （{s}）  {ja}")

    # ---------------- columns
    R.add("")
    R.add("C. COLUMNS you can use in expressions", "C. 条件の式で使える列の名前")
    for c in G.columns:
        en, ja = describe(G.orig.get(c, c))
        R.add(f"   {c:<28} {en}", f"   {c:<28} {ja}")
    R.write(a.out)
    print(f"wrote {a.out}_en.txt and {a.out}_ja.txt")


if __name__ == "__main__":
    main()
