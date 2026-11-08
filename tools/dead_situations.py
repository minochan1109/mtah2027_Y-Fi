"""Run the situation-based D detector (no ball) and report how well it works.

Step 1 - save the state of each game (needs the new --save-state, which also stores the players):
    python -m possession.v2.pipeline_v2 --repo ../more-than-a-hack-2027 --external external/EVAL-G1 \
        --out results/ds/g1.csv --save-state results/ds/g1_state.pkl
    (same for EVAL-G2; for the challenge game leave out --external -> no truth, only the frames CSV)

Step 2 - detect and report:
    python tools/dead_situations.py --repo ../more-than-a-hack-2027 \
        --states results/ds/g1_state.pkl results/ds/g2_state.pkl \
        --config params/dead_situations.txt --out results/ds/report

    -> results/ds/report_en.txt, results/ds/report_ja.txt   (same numbers, English / Japanese)
       results/ds/report_<game>_frames.csv                   (for tools/debug_view.py --frames)
       results/ds/report_<game>_stretches.csv                (every D stretch: time, situation, why)

The flow is explained at the top of possession/v2/dead_situation.py. Only players and Smart Tagging
are used (no ball). Truth (for scores) is only read for OTHER games (made with tools/make_eval_data.py).
"""
import argparse
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from possession import hmm, postprocess  # noqa: E402
from possession.v2 import hmm_v2, tag_policy  # noqa: E402
from possession.v2.config_v2 import Params2  # noqa: E402
from possession.v2.dead_situation import detect, detect_hybrid, read_config  # noqa: E402
from possession.v2.expr import FPS, _runs  # noqa: E402
from possession.v2.situation_features import situation_features  # noqa: E402

SIT_JA_FALLBACK = {"none": "タグなし（ボールアウト等）", "goal": "ゴール後", "throw-in": "スローイン",
                   "corner kick": "コーナーキック", "free kick": "FK", "goal kick": "ゴールキック",
                   "kick off (after goal)": "キックオフ（ゴール後）", "kick off (start)": "キックオフ（開始）",
                   "penalty": "PK", "penalty kick": "PK"}


def tstr(f):
    s = f / FPS
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def truth_types(st):
    """Type of every truth D stretch from the truth tags of the evaluation game (per frame)."""
    T = st["n"]
    typ = np.full(T, "", dtype=object)
    ext = st.get("external")
    tt = []
    if ext:
        with open(os.path.join(ext, "game.json")) as fh:
            stem = json.load(fh)["stem"]
        p = os.path.join(ext, "truth", f"{stem}_smart_tagging.xml")
        if os.path.isfile(p):
            tt = tag_policy.load_tags(p)
    for a, b in _runs(st["gt"] == "D"):
        ty = "none"
        for x in tt:
            if x["code"] == "SET PIECES" and a - FPS <= x["start"] <= b + FPS:
                ty = (x["tags"].get("Type") or "none").lower()
        if ty == "none" and any(x["code"] == "GOALS" and a - 5 * FPS <= x["start"] <= b for x in tt):
            ty = "goal"
        typ[a:b] = ty
    return typ


def relabel(st, mask):
    P = Params2()
    for k, v in st["params"].items():
        setattr(P, k, v)
    p_dead = np.where(mask, 0.999, 0.001)
    logE = hmm.emissions(p_dead, st["p_home"], P)
    states = [hmm_v2.viterbi_tv(logE[a:b + 1], P, st["switch_cost"][a:b + 1]) for _, a, b in st["blocks"]]
    return postprocess.to_labels(states, st["n"], st["blocks"])


def d_metrics(sts, masks):
    nd = npl = tp = fp = 0
    stretches = short = real = hit = 0
    for m, g in zip(masks, sts):
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
    rec, prec = tp / max(nd, 1), tp / max(tp + fp, 1)
    return {"d_recall": rec, "play_to_d": fp / max(npl, 1), "d_precision": prec,
            "d_f1": 2 * rec * prec / max(rec + prec, 1e-9), "d_stretches": stretches, "short": short,
            "real_stretches": real, "real_hit": hit / max(real, 1)}


class Report:
    def __init__(self):
        self.en, self.ja = [], []

    def add(self, en, ja=None):
        self.en.append(en)
        self.ja.append(en if ja is None else ja)

    def write(self, prefix):
        for lang, lines in (("en", self.en), ("ja", self.ja)):
            with open(f"{prefix}_{lang}.txt", "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")


def pct(x):
    return f"{100 * x:5.1f}%"


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(HERE), "params", "dead_situations.txt"))
    ap.add_argument("--repo", help="more-than-a-hack-2027 (for spellF1)")
    ap.add_argument("--out", default="results/ds/report")
    ap.add_argument("--exclude-suspect-s", type=float, default=10.0,
                    help="leave out 'play' stretches of the truth with no true ball for at least this long "
                         "(probably stoppages the event data missed); 0 = use every frame")
    a = ap.parse_args(argv)
    cfg = read_config(a.config)
    hybrid = bool(cfg["core"])
    if hybrid:          # what the frames CSV / report call "start signs": core signs + start event + back_to
        none = {"ja": "-", "en": "-", "expr": "0"}
        cfg["start"] = cfg["core"] + [cfg["event"] or none, cfg["back_to"] or none]
    sits = cfg["situations"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    pm = None
    if a.repo:
        sys.path.insert(0, os.path.join(os.path.abspath(a.repo), "evaluation"))
        import possession_metrics as pm  # noqa: E402

    sts = []
    for p in a.states:
        with open(p, "rb") as fh:
            st = pickle.load(fh)
        if "px" not in st:
            raise SystemExit(f"{p}: old state without players. Run pipeline_v2 --save-state again (new version).")
        t0 = time.time()
        cols = situation_features(st["px"], st["py"], st["team_mat"], st["attack"], st["tags"], st["pitch"], FPS,
                                  st.get("bx"), st.get("by"), st.get("H"), st.get("H_valid"))
        if hybrid and "H" not in st:
            print(f"{p}: no camera model in this state (old --save-state): 'in the picture' is unknown", flush=True)
        cols["p_dead"] = np.asarray(st["p_dead"], float)          # the current D model (core)
        cols["v2_D"] = (st["labels"] == "D").astype(float)        # the current final D decision
        if hybrid and "bx" not in st:
            print(f"{p}: no ball position in this state (old --save-state): ball_* columns are missing", flush=True)
        if hybrid:
            mask, stretches, dropped, info = detect_hybrid(cols, st["blocks"], cfg, st.get("tags"))
        else:
            mask, stretches, dropped, info = detect(cols, st["blocks"], cfg)
        print(f"{st['name']}: features + detection {time.time() - t0:.0f} s, {len(stretches)} D stretches", flush=True)
        graded = np.zeros(st["n"], bool)
        for _, x, y in st["blocks"]:
            graded[x:y + 1] = True
        st.update({"cols": cols, "mask": mask, "stretches": stretches, "dropped": dropped, "info": info,
                   "graded": graded, "has_gt": "gt" in st})
        st["labels_new"] = relabel(st, mask)
        if st["has_gt"]:
            gt = st["gt"]
            st["dead"], st["play"] = gt == "D", np.isin(gt, ["H", "A"])
            suspect = np.zeros(st["n"], bool)
            if a.exclude_suspect_s > 0 and "true_ball_seen" in st:
                for x, y in _runs(st["play"] & ~st["true_ball_seen"]):
                    if (y - x) / FPS >= a.exclude_suspect_s:
                        suspect[x:y] = True
            st["suspect"] = suspect
            st["use"] = graded & (st["dead"] | st["play"]) & ~suspect
            st["ttype"] = truth_types(st)
        sts.append(st)

    R = Report()
    if hybrid:
        R.add("HYBRID D DETECTOR: core from the current D model, start / end from the situation signs, 3-second rule",
              "組み合わせの D 判定: 芯＝現在の D モデル、始まりと終わり＝状況のサイン、3秒ルール")
    else:
        R.add("SITUATION-BASED D DETECTOR (players + tags, NO ball)", "状況別 D 判定（選手とタグのみ・ボールなし）")
    R.add(f"config: {a.config}   need {int(cfg['settings']['need'])} of each list of signs",
          f"設定: {a.config}   各サインのリストから {int(cfg['settings']['need'])} つ以上")
    R.add("")

    # ---------------------------------------------------------------- 1. overall
    with_gt = [s for s in sts if s["has_gt"]]
    if with_gt:
        R.add("1. OVERALL (truth of the evaluation games)", "1. 全体（評価用試合の正解で採点）")
        sus = sum((s["suspect"] & s["graded"]).sum() for s in with_gt) / max(1, sum(s["graded"].sum() for s in with_gt))
        if a.exclude_suspect_s > 0:
            R.add(f"  D metrics leave out {pct(sus)} of the frames: truth says play but the true ball is missing for "
                  f">= {a.exclude_suspect_s:.0f} s (probably a stoppage the truth missed). spellF1 uses every frame.",
                  f"  D の指標はフレームの {pct(sus)} を除外: 正解はプレー中だが本当のボールが {a.exclude_suspect_s:.0f} 秒以上ない"
                  "（正解側の見落としの可能性が高い）。spellF1 は全フレームで計算。")
        new = d_metrics(with_gt, [s["mask"] for s in with_gt])
        old = d_metrics(with_gt, [s["labels"] == "D" for s in with_gt])
        rows = [("d_recall", "D recall (real D frames found)", "D の再現率（本当の D を見つけた割合）"),
                ("play_to_d", "play called D (share of play frames)", "プレー中を D と誤判定した割合"),
                ("d_precision", "D precision", "D の適合率"),
                ("d_f1", "D F1", "D の F1"),
                ("real_hit", "real D stretches found (>=50% covered)", "本当の D 区間を見つけた割合（50%以上）"),
                ("d_stretches", "number of D stretches", "D 区間の数"),
                ("short", "D stretches shorter than 3 s", "3 秒未満の D 区間")]
        R.add(f"{'':44}{'NEW (hybrid)' if hybrid else 'NEW (situations)':>18}{'v2 now':>12}", f"{'':40}{'新（組み合わせ）' if hybrid else '新（状況別）':>16}{'v2 現在':>12}")
        for k, en, ja in rows:
            fmt = (lambda v: pct(v)) if isinstance(new[k], float) else (lambda v: f"{v:6d}")
            R.add(f"  {en:<42}{fmt(new[k]):>18}{fmt(old[k]):>12}", f"  {ja:<38}{fmt(new[k]):>16}{fmt(old[k]):>12}")
        R.add(f"  {'real D stretches in the truth':<42}{new['real_stretches']:>18}",
              f"  {'正解の D 区間の数':<38}{new['real_stretches']:>16}")
        if pm is not None:
            sn, so = [], []
            for s in with_gt:
                sn.append(pm.evaluate(pm.label_codes(s["gt"]), pm.label_codes(s["labels_new"]))["spell_f1_mean"])
                so.append(pm.evaluate(pm.label_codes(s["gt"]), pm.label_codes(s["labels"]))["spell_f1_mean"])
            per = "  ".join(f"{s['name']} {x:.4f}/{y:.4f}" for s, x, y in zip(with_gt, sn, so))
            R.add(f"  spellF1 (new / v2 now): mean {np.mean(sn):.4f} / {np.mean(so):.4f}   ({per})",
                  f"  spellF1（新 / v2 現在）: 平均 {np.mean(sn):.4f} / {np.mean(so):.4f}   ({per})")
            R.add("  (spellF1: only the D decision changes; H/A comes from the v2 model as before)",
                  "  （spellF1 は D の判定だけを入れ替えたもの。H/A は今までの v2 モデルのまま）")
        R.add("")

    # ---------------------------------------------------------------- 2. per detected situation
    R.add("2. D STRETCHES BY DETECTED SITUATION", "2. 判定した状況ごとの D 区間")
    other_en, other_ja = ("core end", "芯の終わり") if hybrid else ("by max", "最大長")
    hdr_en = f"  {'situation':<28}{'count':>6}{'median s':>9}{'ended by end signs':>20}{other_en:>9}"
    hdr_ja = f"  {'状況':<24}{'数':>6}{'中央値秒':>9}{'終了サインで終了':>18}{other_ja:>8}"
    if with_gt:
        hdr_en += f"{'truly D':>9}{'false':>7}"
        hdr_ja += f"{'本当にD':>9}{'誤り':>7}"
    R.add(hdr_en, hdr_ja)
    for k, S in enumerate(sits):
        lens, by_end, by_max, prec, false = [], 0, 0, [], 0
        for s in sts:
            for x in s["stretches"]:
                if x["situation"] != k:
                    continue
                lens.append((x["end"] - x["start"]) / FPS)
                by_end += x["ended_by"] == "end signs"
                by_max += x["ended_by"] != "end signs"
                if s["has_gt"]:
                    pr = s["dead"][x["start"]:x["end"]].mean()
                    prec.append(pr)
                    false += pr < 0.5
        if not lens:
            R.add(f"  {S['en']:<28}{0:>6}", f"  {S['ja']:<24}{0:>6}")
            continue
        line = f"{len(lens):>6}{np.median(lens):>9.1f}{by_end:>20}{by_max:>8}"
        line_ja = f"{len(lens):>6}{np.median(lens):>9.1f}{by_end:>18}{by_max:>8}"
        if with_gt:
            line += f"{pct(np.mean(prec)):>9}{false:>7}"
            line_ja += f"{pct(np.mean(prec)):>9}{false:>7}"
        R.add(f"  {S['en']:<28}" + line, f"  {S['ja']:<24}" + line_ja)
    if hybrid:
        from collections import Counter
        cs = Counter(x["start_by"] for s in sts for x in s["stretches"])
        ce = Counter(x["ended_by"] for s in sts for x in s["stretches"] if not x.get("special"))
        nsp = sum(1 for s in sts for x in s["stretches"] if x.get("special"))
        nsk = [x for s in sts for x in s["info"].get("skipped_special", [])]
        R.add(f"  special stretches from kick-off / goal tags: {nsp} (goal stretches not made: {len(nsk)})",
              f"  キックオフ/ゴールのタグによる特別処理の停止: {nsp}（条件に合わず作らなかったゴール後の停止: {len(nsk)}）")
        for x in nsk:
            R.add(f"    {tstr(x['start'])}-{tstr(x['end'])}  {x['why']}")
        nfill = sum(len(_runs(s["info"]["changed"] == 1)) for s in sts)
        ndrop = sum(len(_runs(s["info"]["changed"] == -1)) for s in sts)
        ncore = sum(len(_runs(s["info"]["core"])) for s in sts)
        R.add(f"  cores: {ncore}   start found by: ball/event {cs['event']}, after last clear play {cs['back_to']}, "
              f"core start {cs['core']}, joined {cs['joined']}   end found by: end signs {ce['end signs']}, core end {ce['core end']}, "
              f"ball moved twice {ce['ball moved twice']}",
              f"  芯: {ncore}   始まり: イベント {cs['event']}、最後のはっきりしたプレーの直後 {cs['back_to']}、芯の始まり {cs['core']}、"
              f"前の停止とつないだ {cs['joined']}"
              f"   終わり: 終了サイン {ce['end signs']}、芯の終わり {ce['core end']}、ボールが2回目に動いた {ce['ball moved twice']}")
        nre = sum(x.get("first_choice", x["situation"]) != x["situation"] for s in sts for x in s["stretches"])
        ndr = sum(len(s["dropped"]) for s in sts)
        R.add(f"  minimum length / require / must per situation: {nre} stretches given another situation, "
              f"{ndr} dropped (play)",
              f"  状況ごとの最低時間・条件: 別の状況にした区間 {nre}、停止にしなかった候補 {ndr}")
        from collections import Counter as _C
        reasons = _C(w.split("：")[0] if "：" in w else w for s in sts for x in s["dropped"] + s["stretches"]
                     for w in (x.get("why") or "").split("、") if w)
        for w, c in reasons.most_common(12):
            R.add(f"    {c:4d}  {w}")
        if with_gt:
            lost = sum(s["dead"][x["start"]:x["end"]].mean() >= 0.5 for s in with_gt for x in s["dropped"])
            R.add(f"    of the dropped ones really D: {lost}", f"    停止にしなかった候補のうち本当は D だったもの: {lost}")
        nveto = sum(int(s["info"]["vetoed"].sum()) for s in sts) / FPS
        R.add(f"  play signs ([play] block) took {nveto:.0f} s out of D stretches",
              f"  プレー中のサイン（[play]）で停止から外した時間: {nveto:.0f} 秒")
        R.add(f"  3-second rule: {ndrop} short D turned into play, {nfill} short play joined into D",
              f"  3秒ルール: 短い停止をプレー中に戻した {ndrop} 回、短いプレー中を停止につないだ {nfill} 回")
    if not hybrid:
        nd = sum(len(s["dropped"]) for s in sts)
        R.add(f"  candidates dropped as too short (end signs before min_s): {nd}",
              f"  短すぎて捨てた候補（min_s より前に終了サイン）: {nd}")
        if with_gt:
            lost = sum(s["dead"][x["start"]:x["end"]].mean() >= 0.5 for s in with_gt for x in s["dropped"])
            R.add(f"    of which really D (lost by the length rule): {lost}",
                  f"    そのうち本当は D だったもの（長さの条件で失った）: {lost}")
        R.add("  'truly D' = mean share of the stretch that is D in the truth; 'false' = stretches mostly play",
              "  「本当にD」= 区間のうち正解で D の割合（平均）、「誤り」= 大半がプレー中だった区間の数")
    R.add("")

    # ---------------------------------------------------------------- 3. per truth type
    if with_gt:
        R.add("3. REAL STOPPAGES BY TYPE (truth tags of the evaluation games)",
              "3. 正解の停止を種類別に（評価用試合の正解タグ）")
        R.add(f"  {'truth type':<24}{'count':>6}{'length s p10/median/p90':>26}{'frames found':>14}{'stretches found':>17}"
              "   detected as",
              f"  {'正解の種類':<22}{'数':>6}{'長さ秒 p10/中央値/p90':>24}{'見つけたフレーム':>16}{'見つけた区間':>14}"
              "   判定された状況")
        types = sorted({t for s in with_gt for t in s["ttype"] if t})
        for ty in types:
            n = 0
            lens, found_f, tot_f, found_s = [], 0, 0, 0
            got = {}
            for s in with_gt:
                for x, y in _runs(s["ttype"] == ty):
                    if not s["graded"][x:y].any():
                        continue
                    n += 1
                    lens.append((y - x) / FPS)
                    cov = s["mask"][x:y]
                    found_f += cov.sum(); tot_f += y - x
                    found_s += cov.mean() >= 0.5
                    so = s["info"]["sit_of"][x:y]
                    so = so[so >= 0]
                    key = sits[int(np.bincount(so).argmax())]["en"] if len(so) else "-"
                    got[key] = got.get(key, 0) + 1
            if not n:
                continue
            q = np.percentile(lens, [10, 50, 90])
            mix = ", ".join(f"{k} {v}" for k, v in sorted(got.items(), key=lambda kv: -kv[1])[:3])
            R.add(f"  {ty:<24}{n:>6}{q[0]:>10.1f}/{q[1]:.1f}/{q[2]:.1f}{pct(found_f / max(tot_f, 1)):>14}"
                  f"{found_s:>11}/{n:<5}   {mix}",
                  f"  {SIT_JA_FALLBACK.get(ty, ty):<22}{n:>6}{q[0]:>10.1f}/{q[1]:.1f}/{q[2]:.1f}{pct(found_f / max(tot_f, 1)):>14}"
                  f"{found_s:>9}/{n:<5}   {mix}")
        R.add("  '-' = not detected at all.  Use the lengths to set min_s / max_s of each situation.",
              "  「-」= まったく検出できなかった。長さの分布は各状況の min_s / max_s を決める参考に。")
        R.add("")

        # ------------------------------------------------------------ 4. signs
        R.add("4. HOW OFTEN EACH SIGN IS TRUE", "4. 各サインが成り立つ割合")
        R.add("  start signs: in the 3 s after a real D starts  vs  during play (lower = fewer false starts)",
              "  開始サイン: 本当の D が始まってから3秒間  vs  プレー中（低いほど誤検出が少ない）")
        for i, sg in enumerate(cfg["start"]):
            dd, pp = [], []
            for s in with_gt:
                m = s["info"]["start_masks"][i]
                idx = np.concatenate([np.arange(x, min(x + 3 * FPS, y)) for x, y in _runs(s["dead"])] or [np.zeros(0, int)])
                dd.append(m[idx]); pp.append(m[s["play"] & s["use"]])
            R.add(f"    {sg['en']:<46} D start {pct(np.concatenate(dd).mean())}   play {pct(np.concatenate(pp).mean())}",
                  f"    {sg['ja']:<36} D開始 {pct(np.concatenate(dd).mean())}   プレー {pct(np.concatenate(pp).mean())}")
        R.add("  end signs: in the 2 s after a real restart  vs  inside real D (lower = fewer early ends)",
              "  終了サイン: 本当の再開から2秒間  vs  本当の D の途中（低いほど早すぎる終了が少ない）")
        for k, S in enumerate(sits):
            R.add(f"   [{S['en']}]", f"   [{S['ja']}]")
            for i, sg in enumerate(S["end"]):
                aa, dd = [], []
                for s in with_gt:
                    m = s["info"]["end_masks"][k][i]
                    idx = np.concatenate([np.arange(y, min(y + 2 * FPS, s["n"])) for x, y in _runs(s["dead"])] or [np.zeros(0, int)])
                    aa.append(m[idx]); dd.append(m[s["dead"] & s["use"]])
                R.add(f"    {sg['en']:<46} restart {pct(np.concatenate(aa).mean())}   inside D {pct(np.concatenate(dd).mean())}",
                      f"    {sg['ja']:<36} 再開 {pct(np.concatenate(aa).mean())}   Dの途中 {pct(np.concatenate(dd).mean())}")
        R.add("  confirm signs: inside real D of the matching truth type vs inside other real D",
              "  確認サイン: その種類の本当の D の中  vs  それ以外の本当の D の中")
        match = {"throw-in": ["throw-in"], "corner kick": ["corner kick"], "goal kick": ["goal kick"],
                 "free kick near goal": ["free kick"], "free kick far from goal": ["free kick"],
                 "kick-off": ["kick off (after goal)", "kick off (start)", "goal"], "penalty kick": ["penalty", "penalty kick"]}
        for k, S in enumerate(sits):
            if not S["confirm"]:
                continue
            want = match.get(S["en"], [])
            R.add(f"   [{S['en']}]  (truth types: {', '.join(want) or '-'})", f"   [{S['ja']}]  （正解の種類: {', '.join(want) or '-'}）")
            for i, sg in enumerate(S["confirm"]):
                mm, oo = [], []
                for s in with_gt:
                    m = s["info"]["conf_masks"][k][i]
                    isw = np.isin(s["ttype"], want) & s["dead"]
                    mm.append(m[isw]); oo.append(m[s["dead"] & ~isw])
                mv = np.concatenate(mm)
                R.add(f"    {sg['en']:<46} this type {pct(mv.mean()) if len(mv) else '   n/a'}   other D {pct(np.concatenate(oo).mean())}",
                      f"    {sg['ja']:<36} この種類 {pct(mv.mean()) if len(mv) else '   n/a'}   他の D {pct(np.concatenate(oo).mean())}")
        R.add("")

    # ---------------------------------------------------------------- 5. columns
    R.add("5. COLUMNS YOU CAN USE (players + tags; see possession/v2/situation_features.py)",
          "5. 使える列（選手とタグ。説明は possession/v2/situation_features.py）")
    names = sorted(sts[0]["cols"].keys())
    for i in range(0, len(names), 6):
        R.add("    " + "  ".join(names[i:i + 6]))
    R.write(a.out)

    # ---------------------------------------------------------------- files per game
    for s in sts:
        info, T = s["info"], s["n"]
        rows = np.flatnonzero(s["graded"])
        df = pd.DataFrame({"frame": rows, "video_time": [tstr(f) for f in rows],
                           "label": s["labels_new"][rows], "p_dead": s["mask"][rows].astype(float),
                           "pseudo_dead": s["mask"][rows].astype(int)})
        if s["has_gt"]:
            df["gt"] = s["gt"][rows]
            df["truth_type"] = s["ttype"][rows]
        so = info["sit_of"][rows]
        df["pseudo_rule"] = [sits[k]["en"] if k >= 0 else "" for k in so]
        df["speed_mean"] = np.round(s["cols"]["sp_mean"][rows], 2)
        df["frac_walking"] = np.round(s["cols"]["frac_walk"][rows], 2)
        df["top1"] = [f"start {int(v)}/{len(cfg['start'])}" for v in info["n_start"][rows]]
        k_end = np.where(so >= 0, so, [i for i, S in enumerate(sits) if S["en"] == "unknown"][0])
        df["top2"] = [f"end {int(info['n_end'][k][r])}/{len(sits[k]['end'])}" for k, r in zip(k_end, rows)]
        short = [sg["en"][:22] for sg in cfg["start"]]
        on = np.array(info["start_masks"])[:, rows]
        df["top3"] = [", ".join(short[i] for i in np.flatnonzero(on[:, j])) for j in range(len(rows))]
        # Japanese explanation columns (tools/compare_dead_view.py shows them next to the video)
        df["situation_ja"] = [sits[k]["ja"] if k >= 0 else "" for k in so]
        df["start_n"] = info["n_start"][rows].astype(int)
        df["start_signs_ja"] = [" / ".join(cfg["start"][i]["ja"] for i in np.flatnonzero(on[:, j]))
                                for j in range(len(rows))]
        df["end_n"] = [int(info["n_end"][k][r]) for k, r in zip(k_end, rows)]
        em = {k: np.array(info["end_masks"][k]) for k in set(k_end.tolist())}
        df["end_signs_ja"] = [" / ".join(sits[k]["end"][i]["ja"] for i in np.flatnonzero(em[k][:, r]))
                              for k, r in zip(k_end, rows)]
        cm = {k: np.array(info["conf_masks"][k]) for k in range(len(sits)) if sits[k]["confirm"]}
        df["confirm_signs_ja"] = [" / ".join(sits[k]["confirm"][i]["ja"] for i in np.flatnonzero(cm[k][:, r]))
                                  if k in cm else "" for k, r in zip(so, rows)]
        note = np.full(T, "", dtype=object)
        for x in s["stretches"]:
            note[x["start"]:x["end"]] = (f"D {tstr(x['start'])}-{tstr(x['end'])} ({(x['end'] - x['start']) / FPS:.0f}秒) "
                                         f"状況スコア {x['score']:.2f} 終了理由: "
                                         + ("終了サイン" if x["ended_by"] == "end signs" else "最大の長さ"))
        for x in s["dropped"]:
            note[x["start"]:x["end"]] = (f"候補 {tstr(x['start'])} は {sits[x['situation']]['ja']} の最短 "
                                         f"{sits[x['situation']]['min_s']:.0f}秒 より前に終了サイン → 短すぎて捨てた")
        if hybrid:
            how_s = {"event": "始まり＝" + (cfg["event"] or {"ja": "-"})["ja"],
                     "back_to": "始まり＝" + (cfg["back_to"] or {"ja": "-"})["ja"] + " の直後", "core": "始まり＝芯の始まり",
                     "joined": "始まり＝前の停止から（間にはっきりしたプレーがないので1つの停止）",
                     "half start": "前後半の始まり〜キックオフ（タグ）", "goal tag": "ゴール〜キックオフ（タグ）"}
            for x in s["stretches"]:
                cs, ce = x["core"]
                if x.get("special"):
                    note[x["start"]:x["end"]] = f"D {tstr(x['start'])}-{tstr(x['end'])} 特別処理：{how_s[x['start_by']]}"
                    continue
                note[x["start"]:x["end"]] = (f"D {tstr(x['start'])}-{tstr(x['end'])}（芯 {tstr(cs)}-{tstr(ce)}） "
                                             f"{how_s[x['start_by']]}、終わり＝"
                                             + {"end signs": "終了サイン", "core end": "芯の終わり",
                                                "ball moved twice": "ボールが2回目に動いた時点"}.get(x["ended_by"], x["ended_by"]))
            for x in s["stretches"]:
                if x.get("why"):
                    note[x["start"]:x["end"]] = (note[x["start"]] + f"（{x['why']} → {sits[x['situation']]['ja']} とした）")
            for x in s["dropped"]:
                note[x["start"]:x["end"]] = (f"停止の候補 {tstr(x['start'])}-{tstr(x['end'])}（{(x['end'] - x['start']) / FPS:.0f}秒）"
                                             f"は停止にしない：{x.get('why', '')}")
            if cfg["veto"]:
                vt = " / ".join(v["ja"] for v in cfg["veto"])
                note[info["vetoed"]] = f"プレー中のサイン（{vt}）があるので停止にしない"
                df["play_sign"] = info["veto"][rows].astype(int)
            for x in info.get("skipped_special", []):
                note[x["start"]:x["end"]] = np.where(note[x["start"]:x["end"]] == "",
                                                     f"ゴール〜キックオフの特別処理はしない：{x['why']}", note[x["start"]:x["end"]])
            ch = info["changed"]
            note[ch == -1] = "3秒未満の停止なのでプレー中に戻した（3秒ルール）"
            note[ch == 1] = "3秒未満のプレー中なので前後の停止とつないだ（3秒ルール）"
            df["core"] = info["core"][rows].astype(int)
        df["mode"] = "hybrid" if hybrid else "situations"
        df["note_ja"] = note[rows]
        df.to_csv(f"{a.out}_{s['name']}_frames.csv", index=False)
        st_rows = []
        for x in s["stretches"]:
            r = {"start_frame": x["start"], "end_frame": x["end"] - 1, "start": tstr(x["start"]), "end": tstr(x["end"]),
                 "seconds": round((x["end"] - x["start"]) / FPS, 1), "situation": sits[x["situation"]]["en"],
                 "situation_ja": sits[x["situation"]]["ja"], "score": round(x["score"], 2), "ended_by": x["ended_by"],
                 "started_by": x.get("start_by", ""),
                 "scores": " ".join(f"{sits[i]['en']}:{v:.2f}" for i, v in enumerate(x["scores"]) if v > 0)}
            if s["has_gt"]:
                r["truth_D_share"] = round(float(s["dead"][x["start"]:x["end"]].mean()), 2)
                tt = [t for t in s["ttype"][x["start"]:x["end"]] if t]
                r["truth_type"] = max(set(tt), key=tt.count) if tt else ""
            st_rows.append(r)
        pd.DataFrame(st_rows).to_csv(f"{a.out}_{s['name']}_stretches.csv", index=False)
    print(f"wrote {a.out}_en.txt, {a.out}_ja.txt and per-game _frames.csv / _stretches.csv")


if __name__ == "__main__":
    main()
