"""[E] of version 3: H / A only, with play-stopped (D) already decided by [D2].

Every play stretch (between two D stretches) is decoded on its own with a two-state Viterbi (H, A);
an H <-> A switch costs switch_cost[t] (cost_switch_team x (1 - switch_signal_drop x signal), as in v2).
"""
import numpy as np


def viterbi_ha(p_home, switch_cost, floor=1e-4):
    """Two states only (0 = H, 1 = A) for one play stretch whose D is already decided.
    p_home (T,), switch_cost (T,) = cost of H<->A INTO frame t. Returns 0/1 per frame."""
    ph = np.clip(np.nan_to_num(p_home, nan=0.5), floor, 1 - floor)
    lh, la = np.log(ph), np.log(1 - ph)
    T = len(ph)
    back = np.zeros((T, 2), np.int8)
    c0, c1 = -lh[0], -la[0]
    for t in range(1, T):
        sc = switch_cost[t]
        b0 = 0 if c0 <= c1 + sc else 1
        b1 = 1 if c1 <= c0 + sc else 0
        n0 = (c0 if b0 == 0 else c1 + sc) - lh[t]
        n1 = (c1 if b1 == 1 else c0 + sc) - la[t]
        back[t, 0], back[t, 1] = b0, b1
        c0, c1 = n0, n1
    s = 0 if c0 <= c1 else 1
    path = np.empty(T, np.int8)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    return path


def scorer_before_goals(labels, tags, side_of, blocks, lead_s, fps=25, shift_s=0.0):
    """A goal is scored by a shot of the scoring team, so the lead_s seconds before every GOALS tag are that
    team's possession (H / A), even if they were D (e.g. a penalty kick: the D of the penalty ran straight
    into the D after the goal). General football rule; own goals (tag text says "own") are left alone.
    Returns (labels, number of goals used)."""
    if lead_s <= 0:
        return labels, 0
    w = int(round(lead_s * fps))
    used = 0
    for g in tags or []:
        if g.get("code") != "GOALS":
            continue
        text = " ".join(t.lower() for _, t in g.get("labels", []))
        if "own" in text:
            continue
        side = side_of(g.get("tags", {}).get("Team", ""))
        if side not in (0, 1):
            continue
        s = int(g["start"]) + int(round(shift_s * fps))      # shift_s > 0: the tag comes before the goal
        blk = [(a, b) for _, a, b in blocks if a <= s <= b]
        if not blk:
            continue
        a = max(blk[0][0], s - w)
        labels[a:s] = "H" if side == 0 else "A"
        used += 1
    return labels, used


def scorer_after_restart(labels, tags, side_of, blocks, stretches, len_s, before_s, after_s, fps=25):
    """A restart that scores at once (a penalty kick, a direct free kick): the short play between the kick
    and the goal is shorter than the 3-second rule, so it was joined into D. If a GOALS tag lies from
    before_s before to after_s after the end of a stoppage (= the kick), the len_s seconds from the kick
    are the scoring team's possession. Own goals are left alone. Returns (labels, number used)."""
    if len_s <= 0:
        return labels, 0
    goals = []
    for g in tags or []:
        if g.get("code") != "GOALS" or "own" in " ".join(t.lower() for _, t in g.get("labels", [])):
            continue
        side = side_of(g.get("tags", {}).get("Team", ""))
        if side in (0, 1):
            goals.append((int(g["start"]), side))
    used = 0
    for x in stretches:
        if x.get("special"):
            continue
        e = int(x["end"])
        hit = [side for s, side in goals if e - before_s * fps <= s <= e + after_s * fps]
        if not hit:
            continue
        blk = [(a, b) for _, a, b in blocks if a <= e <= b]
        if not blk:
            continue
        labels[e:min(blk[0][1] + 1, e + int(round(len_s * fps)))] = "H" if hit[0] == 0 else "A"
        used += 1
    return labels, used


def decode_with_dead(dead, p_home, switch_cost, blocks, n_frames, floor=1e-4):
    """Final labels when D is fixed: 'D' where dead, H/A from viterbi_ha in every play stretch,
    'X' outside the graded halves."""
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, a, b in blocks:
        seg = dead[a:b + 1]
        labels[a:b + 1] = np.where(seg, "D", "")
        m = np.r_[False, ~seg, False].astype(np.int8)
        d = np.diff(m)
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            path = viterbi_ha(p_home[a + s:a + e], switch_cost[a + s:a + e], floor)
            labels[a + s:a + e] = np.where(path == 0, "H", "A")
    return labels
