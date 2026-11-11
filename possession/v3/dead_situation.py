"""Situation-based dead-ball (D) detector. PLAYERS AND TAGS ONLY: the ball is not used.

Flow (settings and signs in params/dead_situations.txt):

  1. START   look for a possible stoppage: at least `need` (2) of the 5 start signs are true for
             start_hold_s. The D stretch begins start_back_s before that (players react late), or
             earlier: right after the last frame where the [start] `back_to` expression (clear play,
             e.g. several players running fast) was true, at most start_back_max_s before.
  2. WHICH   during the first classify_s seconds, every situation counts how often at least `need`
             of its 5 confirm signs are true. The best situation wins; if even the best is true in
             less than classify_min of the frames, the situation is "unknown".
  3. END     from min_s of that situation on, the stoppage ends at the first frame where at least
             `need` of its 5 end signs are true -- and fewer than `need` start signs -- for end_hold_s (the D stretch stops end_back_s
             earlier, the restart happens just before the players accelerate). At max_s it is ended
             by force.
  4. LENGTH  if the end signs already fire before min_s, the candidate was only a short slow-down:
             it is dropped (reported as "too short") and the search goes on after it.

Everything is per frame and uses the same small expression language as tools/dead_conditions.py.
"""
import numpy as np

from .expr import FPS, _runs, evaluate_expr, lasting

DEFAULTS = {"need": 2, "start_hold_s": 0.4, "start_back_s": 1.0, "classify_s": 4.0, "classify_min": 0.25,
            "end_hold_s": 0.3, "end_back_s": 0.4, "gap_after_s": 1.0, "start_back_max_s": 8.0,
            # hybrid (core) mode
            "core_need": 1, "core_min_s": 2.0, "core_merge_s": 1.0, "start_search_s": 10.0,
            "end_search_before_s": 3.0, "end_search_after_s": 1e9, "min_d_s": 3.0, "min_play_s": 3.0,
            "require_share": 0.5, "require_last_s": 3.0, "merge_max_gap_s": 30.0, "kickoff_from_tags": 1,
            "kickoff_after_s": 3.0, "kickoff_max_wait_s": 240.0,
            "ball_moves_max": 1, "ball_move_grace_s": 1.0, "allow_unknown": 0, "goal_play_before_s": 10.0,
            "goal_team_fast": 3, "goal_team_fast_s": 5.0, "kickoff_wrong_max": 2, "kickoff_shape_s": 1.0,
            "kickoff_after_tag_s": 10.0}


def _split(line):
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 3:
        raise ValueError(f"expected 'ja | en | expression': {line}")
    return {"ja": parts[0], "en": parts[1], "expr": " | ".join(parts[2:])}


def read_config(path):
    """[settings] key = value ; [start | 名前] sign = ja | en | expr ;
    [situation 日本語 | english] min_s, max_s, confirm = ..., end = ..."""
    cfg = {"settings": dict(DEFAULTS), "start": [], "situations": [], "back_to": None, "core": [], "event": None,
           "veto": []}
    cur = None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()          # '#' starts a comment (not used in expressions)
            if not line:
                continue
            if line.startswith("["):
                head = line.strip("[]").strip()
                if head == "settings":
                    cur = "settings"
                elif head.startswith("start"):
                    cur = "start"
                elif head.startswith("core"):
                    cur = "core"
                elif head.startswith("play"):
                    cur = "play"
                elif head.startswith("situation"):
                    ja, _, en = head[len("situation"):].partition("|")
                    cur = {"ja": ja.strip(), "en": en.strip() or ja.strip(), "min_s": 0.0, "max_s": 60.0,
                           "confirm": [], "end": [], "require": [], "must": []}
                    cfg["situations"].append(cur)
                else:
                    raise ValueError(f"unknown block [{head}]")
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if cur == "settings":
                cfg["settings"][key] = float(val.split("#")[0])
            elif cur == "start" and key == "sign":
                cfg["start"].append(_split(val))
            elif cur == "start" and key == "back_to":
                cfg["back_to"] = _split(val)
            elif cur == "start" and key == "event":
                cfg["event"] = _split(val)
            elif cur == "core" and key == "sign":
                cfg["core"].append(_split(val))
            elif cur == "play" and key == "sign":
                cfg["veto"].append(_split(val))
            elif isinstance(cur, dict) and key in ("confirm", "end", "require", "must"):
                cur[key].append(_split(val))
            elif isinstance(cur, dict) and key in ("min_s", "max_s"):
                cur[key] = float(val.split("#")[0])
            else:
                raise ValueError(f"line not understood: {raw.strip()}")
    if not (cfg["start"] or cfg["core"]) or not cfg["situations"]:
        raise ValueError("the config needs [start] signs (or a [core] block) and at least one [situation ...] block")
    if not any(s["en"] == "unknown" for s in cfg["situations"]):
        raise ValueError("one situation must be called 'unknown' (used when nothing else fits)")
    return cfg


def _count(signs, cols):
    masks = [evaluate_expr(s["expr"], cols) for s in signs]
    return (np.sum(masks, axis=0) if masks else np.zeros(len(next(iter(cols.values()))))), masks


def detect(cols, blocks, cfg):
    """Returns (mask (T,) bool, stretches list of dicts, dropped list of dicts, per-frame info dict)."""
    st = cfg["settings"]
    need = int(st["need"])
    T = len(next(iter(cols.values())))
    f = lambda s: int(round(s * FPS))  # noqa: E731
    n_start, start_masks = _count(cfg["start"], cols)
    start_ok = lasting(n_start >= need, st["start_hold_s"])
    back = evaluate_expr(cfg["back_to"]["expr"], cols) if cfg.get("back_to") else None
    sits = cfg["situations"]
    conf, endok, n_end_all, end_masks, conf_masks = [], [], [], [], []
    for s in sits:
        nc, cm = _count(s["confirm"], cols)
        ne, em = _count(s["end"], cols)
        conf.append(nc >= need if s["confirm"] else np.zeros(T, bool))
        # an end counts only when the stoppage signs are no longer true at the same time
        endok.append(lasting((ne >= need) & (n_start < need), st["end_hold_s"]))
        n_end_all.append(ne)
        end_masks.append(em)
        conf_masks.append(cm)
    unknown = [i for i, s in enumerate(sits) if s["en"] == "unknown"][0]
    conf_cs = [np.r_[0, np.cumsum(c)] for c in conf]

    mask = np.zeros(T, bool)
    sit_of = np.full(T, -1)
    stretches, dropped = [], []
    for _, a, b in blocks:
        t = a
        while t <= b:
            cand = np.flatnonzero(start_ok[t:b + 1])
            if not len(cand):
                break
            ts = t + cand[0]
            t0 = max(t, ts - f(st["start_hold_s"]) - f(st["start_back_s"]))
            if back is not None:
                lo_b = max(t, ts - f(st["start_back_max_s"]))
                hits = np.flatnonzero(back[lo_b:ts])
                t0 = min(t0, lo_b + (hits[-1] + 1 if len(hits) else 0))
            w1 = min(b + 1, ts + f(st["classify_s"]))
            scores = [(cs[w1] - cs[ts]) / max(1, w1 - ts) for cs in conf_cs]
            k = int(np.argmax(scores))
            if scores[k] < st["classify_min"] or k == unknown:
                k = unknown
            S = sits[k]
            lo, hi = t0 + f(S["min_s"]), min(b + 1, t0 + f(S["max_s"]))
            early = np.flatnonzero(endok[k][ts + f(0.5):lo]) if lo > ts + f(0.5) else []
            if len(early):
                e = ts + f(0.5) + early[0]
                dropped.append({"start": t0, "end": e, "situation": k, "score": scores[k]})
                t = e + 1
                continue
            hit = np.flatnonzero(endok[k][lo:hi]) if hi > lo else []
            if len(hit):
                e, how = lo + hit[0] - f(st["end_back_s"]), "end signs"
            else:
                e, how = hi, "max length"
            e = max(e, t0 + 1)
            mask[t0:e] = True
            sit_of[t0:e] = k
            stretches.append({"start": t0, "end": e, "situation": k, "score": scores[k], "ended_by": how,
                              "scores": scores})
            t = e + f(st["gap_after_s"])
    info = {"n_start": n_start, "start_masks": start_masks, "conf_masks": conf_masks, "end_masks": end_masks,
            "n_end": n_end_all, "sit_of": sit_of}
    return mask, stretches, dropped, info





# ====================================================================== hybrid: core first, then start / end
def fix_short(mask, blocks, min_d_s, min_play_s):
    """3-second rule: D shorter than min_d_s -> play; play shorter than min_play_s -> D (joined to the
    D around it). Returns the new mask and a (T,) int8 array: -1 = D removed, +1 = play filled."""
    m = mask.copy()
    changed = np.zeros(len(m), np.int8)
    k_d, k_p = int(round(min_d_s * FPS)), int(round(min_play_s * FPS))
    for _, a, b in blocks:
        seg = m[a:b + 1]
        for s, e in _runs(seg):
            if e - s < k_d:
                seg[s:e] = False
                changed[a + s:a + e] = -1
        for s, e in _runs(~seg):
            if e - s < k_p and seg.any():
                seg[s:e] = True
                changed[a + s:a + e] = 1
        for s, e in _runs(seg):                        # joining can never make a short D, but be safe
            if e - s < k_d:
                seg[s:e] = False
                changed[a + s:a + e] = -1
        m[a:b + 1] = seg
    return m, changed


def _kickoff_tags(tags):
    """[(start, end)] of SET PIECES tags whose Type is a kick-off, and [start] of GOALS tags (frames)."""
    ko, goals = [], []
    for g in tags or []:
        typ = " ".join(t.lower() for grp, t in g.get("labels", []) if grp.lower() == "type")
        if g.get("code") == "SET PIECES" and ("kick off" in typ or "kick-off" in typ or "kickoff" in typ):
            ko.append((int(g["start"]), int(g["end"])))
        elif g.get("code") == "GOALS":
            goals.append(int(g["start"]))
    return sorted(ko), sorted(goals)


def detect_hybrid(cols, blocks, cfg, tags=None, ko_events=None):
    """Core (e.g. the v2 D model) says WHERE play is stopped; the start and end are then searched around
    it with the situation signs; finally the 3-second rule.

      0. special  ko_events given (kick-off preprocessing, pipeline [C2]): [(start, kick, team, kind)] are D
                  as they are, and the tag-based search below is skipped for that half (a half where
                  the preprocessing found no kick-off falls back to it). Otherwise:
                  (kick-off tags, nearly complete) half start -> first kick-off, and goal -> next kick-off
                  are always D, at least until the kick-off shape (each half holds only one team) is
                  reached; the kick is when that shape breaks up. A goal stretch needs play before the goal (back_to within goal_play_before_s)
                  and one team that never has goal_team_fast+ players running fast for goal_team_fast_s. The kick is the first frame inside the kick-off tag where `need` of the
                  kick-off end signs hold, else kickoff_after_s before the tag end -- always at a moment
                  when the centre spot is in the picture (centre_visible; unknown counts as in it)
      1. core     at least core_need of the [core] signs hold for core_min_s (gaps < core_merge_s joined)
      2. start    within start_search_s before the core: the first frame of the [start] `event`
                  (e.g. the ball position goes outside the pitch); otherwise right after the last frame
                  of `back_to` (clear play, e.g. several players running fast); otherwise the core start.
                  Then walk further back: if there is no clear play (back_to or a [play] sign) between the
                  previous stoppage and this one (at most merge_max_gap_s), both are ONE stoppage.
      3. end      from end_search_before_s before the core end up to end_search_after_s after it: the first
                  frame where `need` of the 5 end signs hold for end_hold_s (minus end_back_s); else the
                  core end
      4. which    every situation is checked (confirm score >= classify_min, best first): its minimum
                  length (min_s), its `require` and `must` lines (in >= require_share of the last
                  require_last_s seconds before the restart). The best one that passes is used. Nothing
                  passes -> not a stoppage ("unknown" is never a stoppage unless allow_unknown = 1).
      4d. ball    the ball never starts moving twice in a stoppage: counting from ball_move_grace_s after the
                  start, the stretch ends at the (ball_moves_max + 1)-th start of `ball_moving`
      5. [play] signs (certainly play, e.g. the keeper holds the ball) are never core
         and are taken out of the D stretches (not out of the special kick-off stretches)
      6. 3-second rule (fix_short)
    """
    st = cfg["settings"]
    need = int(st["need"])
    T = len(next(iter(cols.values())))
    f = lambda s: int(round(s * FPS))  # noqa: E731
    n_core, core_masks = _count(cfg["core"], cols)
    n_veto, veto_masks = _count(cfg["veto"], cols)
    veto = n_veto >= 1
    core = lasting((n_core >= int(st["core_need"])) & ~veto, st["core_min_s"])
    core_full = np.zeros(T, bool)
    for s, e in _runs(core):
        core_full[max(0, s - f(st["core_min_s"])):e] = True
    for s, e in _runs(~core_full):
        if 0 < s and e < T and e - s < f(st["core_merge_s"]):
            core_full[s:e] = True
    event = evaluate_expr(cfg["event"]["expr"], cols) if cfg.get("event") else np.zeros(T, bool)
    back = evaluate_expr(cfg["back_to"]["expr"], cols) if cfg.get("back_to") else np.zeros(T, bool)
    clear_play = back | veto
    cp_cs = np.r_[0, np.cumsum(clear_play)]
    rise = event & ~np.r_[False, event[:-1]]
    sits = cfg["situations"]
    conf, endok, n_end_all, end_masks, conf_masks = [], [], [], [], []
    for S in sits:
        nc, cm = _count(S["confirm"], cols)
        ne, em = _count(S["end"], cols)
        conf.append(nc >= need if S["confirm"] else np.zeros(T, bool))
        endok.append(lasting(ne >= need, st["end_hold_s"]))
        n_end_all.append(ne)
        end_masks.append(em)
        conf_masks.append(cm)
    unknown = [i for i, S in enumerate(sits) if S["en"] == "unknown"][0]
    ko_sit = [i for i, S in enumerate(sits) if S["en"] == "kick-off"]
    ko_sit = ko_sit[0] if ko_sit else unknown
    conf_cs = [np.r_[0, np.cumsum(c)] for c in conf]
    req_cs = [[np.r_[0, np.cumsum(evaluate_expr(x["expr"], cols))] for x in S.get("require", [])] for S in sits]
    must_cs = [[np.r_[0, np.cumsum(evaluate_expr(x["expr"], cols))] for x in S.get("must", [])] for S in sits]
    share = lambda c, x0, x1: (c[x1] - c[x0]) / max(1, x1 - x0)  # noqa: E731

    # ---- 0. special stretches from the kick-off / goal tags
    forced = np.zeros(T, bool)
    stretches, dropped, skipped = [], [], []
    ko_tags, goal_tags = _kickoff_tags(tags) if st.get("kickoff_from_tags", 1) > 0 else ([], [])

    cvis = np.asarray(cols.get("centre_visible", np.full(T, -1.0)))

    wh = np.asarray(cols.get("n_wrong_half", np.full(T, 99.0)))
    shape = lasting(wh <= st["kickoff_wrong_max"], st["kickoff_shape_s"])     # each half holds one team

    def kick_frame(ks, ke, s0=None):
        """The kick. Play is stopped until the kick-off shape (each half holds only one team) is reached;
        the kick is when that shape breaks up (players cross the halfway line), inside / just after the
        kick-off tag, at a moment when the centre spot is in the picture (or unknown). Without the shape:
        the first moment in the tag where `need` kick-off end signs hold, else kickoff_after_s before the tag
        end. None = no moment with the centre spot in the picture (then no special stretch)."""
        s0 = ks if s0 is None else s0
        lim = min(T, ke + f(st["kickoff_after_tag_s"]))
        fs = np.flatnonzero(shape[s0:lim])
        if len(fs):
            first = s0 + fs[0] - f(st["kickoff_shape_s"])                    # lasting() is late by its length
            brk = np.flatnonzero(~shape[s0 + fs[0]:lim])
            if len(brk):
                kf = s0 + fs[0] + brk[0]
                if kf > max(ks - f(5.0), first) and cvis[min(kf, T - 1)] != 0:
                    return int(kf)
        ok = cvis[ks:ke] != 0
        if ke <= ks or not ok.any():
            return None
        hit = np.flatnonzero(endok[ko_sit][ks:ke] & ok)
        if len(hit):
            return ks + hit[0]
        k = max(ks, ke - f(st["kickoff_after_s"]))
        okk = np.flatnonzero(ok)
        return ks + okk[np.argmin(np.abs(ks + okk - k))]            # nearest moment with the centre visible

    if ko_events is not None:                                     # kick-off preprocessing decided them
        for s0, k, team, kind in ko_events:
            if k - s0 > f(st["kickoff_max_wait_s"]) and kind != "goal":
                skipped.append({"start": s0, "end": k, "why": "キックオフまでが長すぎる"})
                continue
            forced[s0:k] = True
            stretches.append({"start": s0, "end": k, "core": (s0, k), "situation": ko_sit, "score": 1.0,
                              "scores": [0.0] * len(sits), "start_by": "half start" if kind != "goal" else "goal",
                              "ended_by": "kick-off (preprocessing)", "first_choice": ko_sit, "why": "",
                              "special": True, "kick_team": team})
    for _, a, b in blocks:
        if ko_events and any(a <= k <= b for _, k, _, _ in ko_events):
            continue                                              # this half: the preprocessing decided
        kos = [(ks, ke) for ks, ke in ko_tags if a <= ks <= b]
        if kos:                                                   # half start -> first kick-off
            k = kick_frame(*kos[0], s0=a)
            if k is not None and k - a <= f(st["kickoff_max_wait_s"]):
                forced[a:k] = True
                stretches.append({"start": a, "end": k, "core": (a, k), "situation": ko_sit, "score": 1.0,
                                  "scores": [0.0] * len(sits), "start_by": "half start", "ended_by": "kick-off tag",
                                  "first_choice": ko_sit, "why": "", "special": True})
        for g in goal_tags:                                       # play -> goal -> kick-off
            if not a <= g <= b:
                continue
            nxt = [(ks, ke) for ks, ke in kos if ks > g]
            if not nxt:
                continue
            k = kick_frame(*nxt[0], s0=g)
            if k is None or not 0 < k - g <= f(st["kickoff_max_wait_s"]):
                continue
            # play before the goal: clear play (back_to) within goal_play_before_s before the tag
            if st["goal_play_before_s"] > 0 and not back[max(a, g - f(st["goal_play_before_s"])):g].any():
                skipped.append({"start": g, "end": k, "why": "ゴールの前にプレーがない"})
                continue
            # one team celebrates, the other walks: in at least one team, never goal_team_fast or more
            # players running fast (> 3 m/s) for goal_team_fast_s
            calm_team = st["goal_team_fast_s"] <= 0
            for key in ("n_fast_home", "n_fast_away") if not calm_team else ():
                if key not in cols:
                    calm_team = True
                    break
                busy = lasting(np.asarray(cols[key][g:k]) >= st["goal_team_fast"], st["goal_team_fast_s"])
                if not busy.any():
                    calm_team = True
            if not calm_team:
                skipped.append({"start": g, "end": k, "why": "両チームとも3人以上が5秒以上走っている（ゴール後に見えない）"})
                continue
            if True:
                forced[g:k] = True
                stretches.append({"start": g, "end": k, "core": (g, k), "situation": ko_sit, "score": 1.0,
                                  "scores": [0.0] * len(sits), "start_by": "goal tag", "ended_by": "kick-off tag",
                                  "first_choice": ko_sit, "why": "", "special": True})

    raw = forced.copy()
    sit_of = np.full(T, -1)
    sit_of[forced] = ko_sit
    for _, a, b in blocks:
        prev_end = a
        for cs, ce in _runs(core_full[a:b + 1]):
            cs, ce = cs + a, ce + a
            if forced[cs:ce].all():
                prev_end = max(prev_end, ce)
                continue
            fe = np.flatnonzero(forced[a:cs])                     # a special stretch ending before this core
            if len(fe):
                prev_end = max(prev_end, a + fe[-1] + 1)
            w1 = min(b + 1, max(ce, cs + f(st["classify_s"])))
            scores = [(c[w1] - c[cs]) / max(1, w1 - cs) for c in conf_cs]
            # ---- 2. start, then the backward check
            lo = max(a, cs - f(st["start_search_s"]))
            ev = np.flatnonzero(rise[lo:cs + 1])
            bk = np.flatnonzero(back[lo:cs])
            if len(ev):
                d0, how_s = lo + ev[0], "event"
            elif len(bk):
                d0, how_s = lo + bk[-1] + 1, "back_to"
            else:
                d0, how_s = cs, "core"
            merged = False
            if prev_end > a and d0 - prev_end <= f(st["merge_max_gap_s"]) and cp_cs[d0] - cp_cs[prev_end] == 0:
                d0, how_s, merged = prev_end, "joined", True

            def find_end(k):
                e_lo = max(d0 + 1, ce - f(st["end_search_before_s"]))
                e_hi = min(b + 1, max(ce, d0 + f(sits[k]["max_s"])), ce + f(st["end_search_after_s"]))
                hit = np.flatnonzero(endok[k][e_lo:e_hi]) if e_hi > e_lo else []
                if len(hit):
                    return max(d0 + 1, e_lo + hit[0] - f(st["end_back_s"])), "end signs"
                return ce, "core end"

            # ---- 4. which situation: check them all, best score first
            order = [int(i) for i in np.argsort(scores)[::-1]
                     if scores[i] >= st["classify_min"] and i not in (unknown, ko_sit)]
            if st.get("allow_unknown", 0) > 0 and (not order or st.get("fallback_unknown", 0) > 0):
                order.append(unknown)
            first = order[0] if order else unknown
            chosen, why, i = None, [], 0
            d1 = ce
            while i < len(order):
                k = order[i]
                i += 1
                d1, how_e = find_end(k)
                if d1 - d0 < f(sits[k]["min_s"]):
                    why.append(f"{sits[k]['ja']}: 最低 {sits[k]['min_s']:.0f} 秒より短い")
                    continue
                w0 = max(d0, d1 - f(st["require_last_s"])) if st["require_last_s"] > 0 else d0
                bad = [x["ja"] for x, c in zip(sits[k].get("must", []), must_cs[k]) if share(c, w0, d1) < st["require_share"]]
                if bad:
                    why.append(f"{sits[k]['ja']}: 「{' / '.join(bad)}」が成り立たない")
                    continue
                bad = [x["ja"] for x, c in zip(sits[k].get("require", []), req_cs[k]) if share(c, w0, d1) < st["require_share"]]
                if bad:
                    why.append(f"{sits[k]['ja']}: 「{' / '.join(bad)}」が成り立たない")
                    if st.get("allow_unknown", 0) > 0 and unknown not in order:
                        order.append(unknown)
                    continue
                chosen = k
                break
            if chosen is None:
                if not why:
                    why.append("どの状況にも当てはまらない（不明な状況では停止にしない）")
                if merged:                    # keep the stretch we joined to, without this core
                    d0 = cs
                dropped.append({"start": d0, "end": max(d1, d0 + 1), "situation": first, "score": scores[first],
                                "core": (cs, ce), "why": "、".join(why)})
                continue
            k = chosen
            # the ball never starts moving twice in a stoppage: from its 2nd move on, it is play
            cut = ""
            if st.get("ball_moves_max", 0) > 0 and "ball_moving" in cols:
                bm = np.asarray(cols["ball_moving"], bool)
                g0 = min(d1, d0 + f(st["ball_move_grace_s"]))
                on = np.flatnonzero(bm[g0:d1] & ~bm[g0 - 1:d1 - 1]) + g0 if d1 > g0 else []
                if len(on) > st["ball_moves_max"]:
                    d1 = int(on[int(st["ball_moves_max"])])
                    how_e, cut = "ball moved twice", "ボールが2回動いたのでそこからプレー中"
                    if d1 - d0 < f(st["min_d_s"]):
                        dropped.append({"start": d0, "end": max(d1, d0 + 1), "situation": k, "score": scores[k],
                                        "core": (cs, ce), "why": cut})
                        continue
            raw[d0:d1] = True
            sit_of[d0:d1] = np.where(forced[d0:d1], sit_of[d0:d1], k)
            stretches.append({"start": d0, "end": d1, "core": (cs, ce), "situation": k, "score": scores[k],
                              "scores": scores, "start_by": how_s, "ended_by": how_e,
                              "first_choice": first, "why": "、".join(why + ([cut] if cut else []))})
            prev_end = max(prev_end, d1)
    stretches.sort(key=lambda x: x["start"])

    # ---- 5. play signs, 6. 3-second rule (special kick-off stretches always stay D)
    vetoed = raw & veto & ~forced
    raw = raw & ~vetoed
    mask, changed = fix_short(raw, blocks, st["min_d_s"], st["min_play_s"])
    back_on = mask & veto & ~forced
    if back_on.any():
        mask &= ~back_on
        vetoed |= back_on
        m2, ch2 = fix_short(mask, blocks, st["min_d_s"], 0.0)
        changed[ch2 == -1] = -1
        mask = m2
    mask |= forced
    for s, e in _runs(changed == 1):
        sit_of[s:e] = sit_of[s - 1] if s > 0 else -1
    sit_of[~mask] = -1
    info = {"n_start": n_core, "start_masks": core_masks + [event, back], "conf_masks": conf_masks,
            "end_masks": end_masks, "n_end": n_end_all, "sit_of": sit_of, "core": core_full, "raw": raw,
            "changed": changed, "veto": veto, "vetoed": vetoed, "veto_masks": veto_masks, "forced": forced,
            "skipped_special": skipped}
    return mask, stretches, dropped, info
