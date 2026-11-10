"""Side-by-side debug video: play or stopped?  LEFT = current method (v2), RIGHT = new method (situations,
or the hybrid: core -> start / end -> 3-second rule, when the right CSV was made with params/dead_hybrid.txt).

Only "play" vs "stopped (D)" is shown (H and A are not separated). Both halves show the SAME video
frame; the outer frame of each half is coloured by that method's decision:
    green = play            orange = stopped (D)
Under each half, the reason for the decision is written in Japanese:
    LEFT  (v2)          P(dead), the pseudo-label rule, the features that pushed P(dead) up / down
    RIGHT (situations)  the situation, how many start / end signs hold (k / 5) and which ones,
                        the confirm signs of the situation, and why the D stretch started / ended
At the bottom, +-10 s strips: current / new (/ truth when the CSV has a gt column).

Step 1 - the current method, with its reasons per frame:
    python -m possession.v3.pipeline_v3 --repo ../more-than-a-hack-2027 --dead-method v2 --out results/cmp/v2.csv \
        --dead-debug results/cmp/v2 --save-state results/cmp/state.pkl
Step 2 - the new method on the same state:
    python tools/dead_situations.py --states results/cmp/state.pkl --out results/cmp/new
Step 3 - the video (clips where the two methods disagree, plus some where they agree):
    python tools/compare_dead_view.py --repo ../more-than-a-hack-2027 \
        --left results/cmp/v2_frames.csv --right results/cmp/new_BAR-ATH_frames.csv \
        --auto --out-video results/cmp/compare.mp4
    -> compare_left_only_d.mp4   current says stopped, new says play
       compare_right_only_d.mp4  new says stopped, current says play
       compare_both_d.mp4 / compare_both_play.mp4   both agree (random samples)
       compare_index.txt         every clip with its time and both reasons + whole-match numbers
    or one scene:  --start 23:10 --seconds 40 --out-video results/cmp/scene.mp4

--left-drop-short 3 shows the current method with D stretches shorter than 3 s turned into play
(the "v2 minus short D" idea), without running anything again.
Only the video and our own outputs are used; nothing here labels the provided game.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from debug_view import runs, to_frame, tstr  # noqa: E402

FPS = 25
PLAY, DEAD, NONE = (60, 190, 60), (0, 140, 255), (30, 30, 30)        # BGR: green, orange, dark
FONTS = ["C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/msgothic.ttc",
         "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
         "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]
RULE_JA = {"calm+ball_out": "全員ゆっくり＋ボールが外に出た", "calm+before_setpiece": "全員ゆっくり＋セットプレーの前",
           "calm+ball_unseen": "全員ゆっくり＋ボールが見えない", "calm+ball_still": "全員ゆっくり＋ボールが止まっている",
           "control": "誰かがボールを持っている", "ball_travelling": "ボールが動いている",
           "players_running": "選手が走っている"}


# ---------------------------------------------------------------- text (Japanese with PIL when a font exists)
class Text:
    def __init__(self, size):
        self.font = None
        try:
            from PIL import ImageFont
            for p in FONTS:
                if os.path.isfile(p):
                    self.font = ImageFont.truetype(p, size)
                    break
        except ImportError:
            pass
        self.size = size
        if self.font is None:
            print("no Japanese font found: text is drawn with OpenCV (Japanese shows as ???)", flush=True)

    def draw(self, img, lines, x=8, y=6, color=(255, 255, 255)):
        """lines: list of str or (str, BGR)."""
        if self.font is None:
            for i, ln in enumerate(lines):
                s, c = ln if isinstance(ln, tuple) else (ln, color)
                cv2.putText(img, s.encode("ascii", "replace").decode(), (x, y + 16 + i * (self.size + 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, self.size / 30, c, 1, cv2.LINE_AA)
            return img
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for i, ln in enumerate(lines):
            s, c = ln if isinstance(ln, tuple) else (ln, color)
            d.text((x, y + i * (self.size + 6)), s, font=self.font, fill=(c[2], c[1], c[0]))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def wrap(s, n):
    s = str(s)
    return [s[i:i + n] for i in range(0, len(s), n)] or [""]


# ---------------------------------------------------------------- results
def feature_ja(token):
    """'missing_ball_speed:+1.23' -> ('ボールの速さ が欠損', '+1.23')."""
    try:
        from dead_conditions import describe
    except Exception:  # noqa: BLE001
        describe = lambda c: (c, c)  # noqa: E731
    name, _, val = str(token).rpartition(":")
    if not name:
        return str(token), ""
    if name.startswith("missing_"):
        return describe(name[len("missing_"):])[1] + " が欠損（無い）", val
    return describe(name)[1], val


def load(path, n, drop_short_s):
    fr = pd.read_csv(path, keep_default_na=False, low_memory=False)
    lab = np.full(n, "", dtype="<U1")
    f = fr["frame"].values.astype(int)
    f_ok = f < n
    lab[f[f_ok]] = fr["label"].astype(str).values[f_ok]
    dead = lab == "D"
    play = np.isin(lab, ["H", "A"])
    if drop_short_s > 0:
        for s, e in runs(dead):
            if (e + 1 - s) < drop_short_s * FPS:
                dead[s:e + 1] = False
                play[s:e + 1] = True
    rows = {int(k): r for k, r in zip(f, fr.to_dict("records"))}
    gt = None
    if "gt" in fr.columns:
        gt = np.full(n, "", dtype="<U1")
        gt[f[f_ok]] = fr["gt"].astype(str).values[f_ok]
    hybrid = "mode" in fr.columns and (fr["mode"].astype(str) == "hybrid").any()
    return {"dead": dead, "play": play, "rows": rows, "gt": gt, "hybrid": hybrid}


def left_lines(r, dead, play, drop_short_s):
    if not r:
        return [("（この時間は採点対象外）", (180, 180, 180))]
    dec = "停止（D）" if dead else ("プレー中" if play else "-")
    lines = [(f"判定: {dec}", DEAD if dead else PLAY),
             f"P(dead) = {r.get('p_dead', '')}   ※ P(dead) が高い状態が続くと停止（HMM で整える）"]
    if drop_short_s > 0 and r.get("label") == "D" and not dead:
        lines.append(("元は D だが、{:.0f}秒未満の D なのでプレー中に戻した".format(drop_short_s), (0, 200, 255)))
    rule = str(r.get("pseudo_rule", ""))
    pd_ = r.get("pseudo_dead", "")
    lines.append(f"疑似ラベル: {'停止' if str(pd_) == '1' else ('プレー' if str(pd_) == '0' else 'なし')}"
                 + (f"（{RULE_JA.get(rule, rule)}）" if rule else ""))
    lines.append(f"平均速度 {r.get('speed_mean', '')} m/s   歩いている割合 {r.get('frac_walking', '')}   "
                 f"ボールが見えている {r.get('ball_seen', '')}")
    ups, downs = [], []
    for k in range(1, 6):
        tok = r.get(f"top{k}", "")
        if not tok:
            continue
        name, val = feature_ja(tok)
        (ups if val.startswith("+") else downs).append(f"{name} {val}")
    lines.append(("停止へ押した: " + ("、".join(ups[:3]) or "-"), (200, 200, 255)))
    lines.append(("プレーへ押した: " + ("、".join(downs[:3]) or "-"), (200, 255, 200)))
    return lines


def right_lines(r, dead, play):
    if not r:
        return [("（この時間は採点対象外）", (180, 180, 180))]
    dec = "停止（D）" if dead else ("プレー中" if play else "-")
    sit = r.get("situation_ja", "") or ""
    lines = [(f"判定: {dec}" + (f"   状況: {sit}" if sit and dead else ""), DEAD if dead else PLAY)]
    if r.get("mode") == "hybrid":
        lines.append(f"芯（止まっていると決まった区間）: {'あり' if str(r.get('core', '')) == '1' else 'なし'}"
                     + ("   プレー中のサイン: あり（キーパーがボールを持っている等）" if str(r.get("play_sign", "")) == "1" else ""))
        lines.append(f"成り立っているサイン: {r.get('start_signs_ja', '') or '-'}")
        lines.append(f"終了サイン {r.get('end_n', '')}/5（2つ以上で再開）: {r.get('end_signs_ja', '') or '-'}")
        if dead and sit:
            lines.append(f"確認サイン（{sit}）: {r.get('confirm_signs_ja', '') or '-'}")
        if r.get("note_ja", ""):
            lines.append((str(r["note_ja"]), (0, 200, 255)))
        lines.append(f"平均速度 {r.get('speed_mean', '')} m/s   歩いている割合 {r.get('frac_walking', '')}   （ボールは位置だけ）")
        return lines
    lines.append(f"開始サイン {r.get('start_n', '')}/5（2つ以上で停止の候補）: {r.get('start_signs_ja', '') or '-'}")
    lines.append(f"終了サイン {r.get('end_n', '')}/5（2つ以上で再開）: {r.get('end_signs_ja', '') or '-'}")
    if dead and sit:
        lines.append(f"確認サイン（{sit}）: {r.get('confirm_signs_ja', '') or '-'}")
    note = r.get("note_ja", "")
    if note:
        lines.append((str(note), (0, 200, 255)))
    lines.append(f"平均速度 {r.get('speed_mean', '')} m/s   歩いている割合 {r.get('frac_walking', '')}   （ボールは使わない）")
    return lines


# ---------------------------------------------------------------- clips
def pick(L, R, n_frames, per_cat, rng, min_s=1.0, merge_s=1.0):
    graded = L["play"] | L["dead"] | R["play"] | R["dead"]
    masks = {"left_only_d": L["dead"] & R["play"], "right_only_d": R["dead"] & L["play"],
             "both_d": L["dead"] & R["dead"], "both_play": L["play"] & R["play"]}
    cats = {}
    for name, m in masks.items():
        m = m & graded
        rs = []
        for s, e in runs(m):
            if rs and s - rs[-1][1] <= merge_s * FPS:
                rs[-1] = (rs[-1][0], e)
            else:
                rs.append((s, e))
        rs = [(s, e + 1) for s, e in rs if e + 1 - s >= min_s * FPS]
        if name.endswith("only_d"):
            rs = sorted(rs, key=lambda r: -(r[1] - r[0]))[:per_cat]                  # biggest differences
            rs = sorted(rs)
        else:
            rs = [rs[i] for i in sorted(rng.choice(len(rs), size=min(per_cat, len(rs)), replace=False))] if rs else []
        cats[name] = rs
    return cats


def summary(L, R, drop):
    g = L["play"] | L["dead"]
    def stretches(m):  # noqa: E306
        rr = [(e + 1 - s) / FPS for s, e in runs(m & g)]
        return len(rr), sum(x < 3 for x in rr)
    nl, sl = stretches(L["dead"])
    nr, sr = stretches(R["dead"])
    agree = ((L["dead"] == R["dead"]) & g).sum() / max(1, g.sum())
    out = ["試合全体（採点対象の時間）",
           f"  停止の割合        現在 {L['dead'][g].mean():.1%}   新 {R['dead'][g].mean():.1%}",
           f"  停止の区間の数    現在 {nl}（3秒未満 {sl}）   新 {nr}（3秒未満 {sr}）",
           f"  2つの判定が同じ   {agree:.1%} のフレーム",
           f"  現在だけ停止      {(L['dead'] & R['play'] & g).sum() / FPS:.0f} 秒   "
           f"新だけ停止 {(R['dead'] & L['play'] & g).sum() / FPS:.0f} 秒"]
    if drop > 0:
        out.append(f"  （現在の手法は {drop:.0f} 秒未満の D をプレー中に戻して表示）")
    if L["gt"] is not None:
        gt = L["gt"]
        for nm, X in (("現在", L), ("新", R)):
            d, p = gt == "D", np.isin(gt, ["H", "A"])
            out.append(f"  正解あり: {nm}  D の再現率 {(X['dead'] & d).sum() / max(1, d.sum()):.1%}   "
                       f"プレー中→D {(X['dead'] & p).sum() / max(1, p.sum()):.1%}")
    return out


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data-dir")
    ap.add_argument("--game", default="BAR-ATH")
    ap.add_argument("--video", help="video file (default: the game's tactical video from games.py)")
    ap.add_argument("--left", required=True, help="<prefix>_frames.csv from pipeline_v2 --dead-debug")
    ap.add_argument("--right", required=True, help="<out>_<game>_frames.csv from tools/dead_situations.py")
    ap.add_argument("--left-drop-short", type=float, default=0.0,
                    help="show the current method with D stretches shorter than this (s) turned into play")
    ap.add_argument("--start", default="0")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--auto", action="store_true", help="pick clips where the methods disagree / agree")
    ap.add_argument("--per-category", type=int, default=6)
    ap.add_argument("--pad-s", type=float, default=4.0)
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--scale", type=float, default=0.5, help="size of each video half (0.5 = 960x540)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-video", default="results/cmp/compare.mp4")
    a = ap.parse_args(argv)

    repo = os.path.abspath(a.repo)
    sys.path.insert(0, repo)
    import games  # noqa: E402
    if a.video:
        video = a.video
    else:
        data_dir = a.data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(repo, "data")
        video = games.asset_paths(a.game, data_dir)["video"]
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fl = pd.read_csv(a.left, usecols=["frame"])["frame"].max()
    fr_ = pd.read_csv(a.right, usecols=["frame"])["frame"].max()
    n = int(max(total, fl + 1, fr_ + 1))
    L = load(a.left, n, a.left_drop_short)
    R = load(a.right, n, 0.0)

    vw, vh = int(1920 * a.scale) // 2 * 2, int(1080 * a.scale) // 2 * 2
    txt = Text(15)
    head = Text(20)
    PANEL_H, STRIP_H, TOP_H, B = 235, 90, 34, 18
    W = 2 * vw
    H = TOP_H + vh + PANEL_H + STRIP_H

    def strip(t, span=10 * FPS):
        img = np.full((STRIP_H, W, 3), 15, np.uint8)
        rows = [("現在", L), ("新", R)] + ([("正解", {"dead": L["gt"] == "D", "play": np.isin(L["gt"], ["H", "A"])})]
                                         if L["gt"] is not None else [])
        x0, x1 = 70, W - 10
        ts = np.linspace(t - span, t + span, x1 - x0).astype(int)
        ok = (ts >= 0) & (ts < n)
        tsc = np.clip(ts, 0, n - 1)
        hh = (STRIP_H - 20) // max(3, len(rows))
        for i, (nm, X) in enumerate(rows):
            y = 6 + i * hh
            col = np.where((X["dead"][tsc] & ok)[:, None], DEAD, np.where((X["play"][tsc] & ok)[:, None], PLAY, NONE))
            img[y:y + hh - 4, x0:x1] = col.astype(np.uint8)[None, :, :]
        img = txt.draw(img, [nm for nm, _ in rows], x=8, y=4)
        xm = x0 + (x1 - x0) // 2
        cv2.line(img, (xm, 2), (xm, STRIP_H - 16), (255, 255, 255), 2)
        y = STRIP_H - 21
        img = txt.draw(img, ["−10秒"], x=x0, y=y)
        img = txt.draw(img, [f"いま {tstr(t)}"], x=xm - 50, y=y)
        img = txt.draw(img, ["+10秒"], x=x1 - 50, y=y)
        img = txt.draw(img, [("緑 = プレー中", PLAY), ("橙 = 停止", DEAD)][:1], x=x0 + 120, y=y)
        img = txt.draw(img, [("橙 = 停止", DEAD)], x=x0 + 260, y=y)
        return img

    def draw(t, img, title):
        frame = cv2.resize(img, (vw, vh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((H, W, 3), np.uint8)
        rname = "新しい手法（組み合わせ：芯→始まり・終わり→3秒ルール）" if R["hybrid"] else "新しい手法（状況別・ボールなし）"
        for k, (X, lines_fn, name) in enumerate(((L, None, "現在の手法（v2）"), (R, None, rname))):
            f = frame.copy()
            dead, play = bool(X["dead"][t]), bool(X["play"][t])
            c = DEAD if dead else (PLAY if play else NONE)
            cv2.rectangle(f, (0, 0), (vw - 1, vh - 1), c, B)
            canvas[TOP_H:TOP_H + vh, k * vw:(k + 1) * vw] = f
            panel = np.full((PANEL_H, vw, 3), 25, np.uint8)
            r = X["rows"].get(t, {})
            lines = left_lines(r, dead, play, a.left_drop_short) if k == 0 else right_lines(r, dead, play)
            out = [(name, (255, 255, 255))]
            for ln in lines:
                s, col = ln if isinstance(ln, tuple) else (ln, (230, 230, 230))
                out += [(w, col) for w in wrap(s, int(vw / 15.5))]
            canvas[TOP_H + vh:TOP_H + vh + PANEL_H, k * vw:(k + 1) * vw] = txt.draw(panel, out[:10])
        canvas[TOP_H + vh:TOP_H + vh + PANEL_H, vw - 1:vw + 1] = 255
        canvas[H - STRIP_H:] = strip(t)
        top = np.zeros((TOP_H, W, 3), np.uint8)
        same = "同じ判定" if L["dead"][t] == R["dead"][t] else "判定が違う！"
        canvas[:TOP_H] = head.draw(top, [f"{title}   {tstr(t)}   frame {t}   左: 現在   右: 新   → {same}"], y=4)
        return canvas

    def write_range(out, a0, a1, title):
        cap.set(cv2.CAP_PROP_POS_FRAMES, a0)
        t = a0
        ok, img = cap.read()
        while ok and t < a1:
            out.write(draw(t, img, title))
            t += 1
            ok, img = cap.read()

    os.makedirs(os.path.dirname(os.path.abspath(a.out_video)) or ".", exist_ok=True)
    base, ext = os.path.splitext(a.out_video)
    ext = ext or ".mp4"
    index = summary(L, R, a.left_drop_short) + [""]
    if a.auto:
        cats = pick(L, R, n, a.per_category, np.random.default_rng(a.seed))
        names = {"left_only_d": "現在だけ停止（新はプレー中）", "right_only_d": "新だけ停止（現在はプレー中）",
                 "both_d": "どちらも停止", "both_play": "どちらもプレー中"}
        for cat, clips in cats.items():
            if not clips:
                continue
            path = f"{base}_{cat}{ext}"
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
            index.append(f"[{names[cat]}] -> {path}")
            for i, (s, e) in enumerate(clips):
                m = (s + e) // 2
                rl, rr = L["rows"].get(m, {}), R["rows"].get(m, {})
                index.append(f"  {i + 1}. {tstr(s)} - {tstr(e)}（{(e - s) / FPS:.1f} 秒）")
                index.append(f"       現在: P(dead) {rl.get('p_dead', '')}  {RULE_JA.get(str(rl.get('pseudo_rule', '')), rl.get('pseudo_rule', ''))}"
                             f"  押した特徴 {feature_ja(rl.get('top1', ''))[0]} {feature_ja(rl.get('top1', ''))[1]}")
                index.append(f"       新  : {rr.get('situation_ja', '') or '-'}  開始 {rr.get('start_n', '')}/5  "
                             f"終了 {rr.get('end_n', '')}/5  {rr.get('note_ja', '')}")
                card = np.zeros((H, W, 3), np.uint8)
                card = head.draw(card, [f"{names[cat]}  #{i + 1}", f"{tstr(s)} - {tstr(e)}  （{(e - s) / FPS:.1f} 秒）"],
                                 x=60, y=H // 2 - 30)
                for _ in range(FPS):
                    out.write(card)
                a0 = max(0, int(s - a.pad_s * FPS))
                a1 = min(total, int(min(e + a.pad_s * FPS, a0 + a.max_s * FPS)))
                write_range(out, a0, a1, f"{names[cat]} #{i + 1}")
            out.release()
            print(f"wrote {path} ({len(clips)} clips)", flush=True)
    else:
        f0 = to_frame(a.start)
        f1 = min(total, f0 + int(a.seconds * FPS))
        out = cv2.VideoWriter(a.out_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        write_range(out, f0, f1, "")
        out.release()
        index.append(f"{a.out_video}: {tstr(f0)} - {tstr(f1)}")
        print(f"wrote {a.out_video}", flush=True)
    with open(f"{base}_index.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(index) + "\n")
    print("\n".join(index[:8]))
    print(f"wrote {base}_index.txt")
    cap.release()


if __name__ == "__main__":
    main()
