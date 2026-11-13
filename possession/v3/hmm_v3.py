"""[E] of version 3: H / A only, with play-stopped (D) already decided by [D2].

Every play stretch (between two D stretches) is decoded on its own with a two-state Viterbi (H, A);
an H <-> A switch costs switch_cost[t] (cost_switch_team x (1 - switch_signal_drop x signal), as in v2).
"""
import numpy as np

from .expr import _runs


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


def drop_unsupported_spells(labels, px, py, team_mat, bx, by, max_s, dist_m, fps=25):
    """H / A runs of at most max_s seconds in which no player of that team is ever within dist_m of the
    (trusted) ball while nearer than every opponent: no touch, so no change of possession -- the run gets
    the other team (the one before it, else the one after it). Runs at the very start / end are kept.
    team_mat: 0 = home, 1 = away. Returns (labels, number of runs changed)."""
    out = np.asarray(labels).copy()
    if max_s <= 0 or len(out) == 0:
        return out, 0
    with np.errstate(all="ignore"):
        d = np.hypot(px - bx[:, None], py - by[:, None])
        dh = np.nanmin(np.where(team_mat == 0, d, np.nan), 1)
        da = np.nanmin(np.where(team_mat == 1, d, np.nan), 1)
    br = np.flatnonzero(out[1:] != out[:-1]) + 1
    bounds = np.r_[0, br, len(out)]
    runs = [(int(bounds[i]), int(bounds[i + 1]), out[bounds[i]]) for i in range(len(bounds) - 1)]
    lim = int(round(max_s * fps))
    changed = 0
    for i, (a, b, k) in enumerate(runs):
        if k not in ("H", "A") or b - a > lim or i == 0 or i == len(runs) - 1:
            continue
        own, oth = (dh, da) if k == "H" else (da, dh)
        if np.nansum((own[a:b] < dist_m) & ~(oth[a:b] < own[a:b])) >= 1:
            continue
        prev, nxt = runs[i - 1][2], runs[i + 1][2]
        if prev in ("H", "A") and prev != k:
            out[a:b] = prev
        elif nxt in ("H", "A") and nxt != k:
            out[a:b] = nxt
        else:
            continue
        changed += 1
    return out, changed


def advance_confirmed_turnovers(labels, px, py, team_mat, bx, by, search_s, control_s, fps=25):
    """Move an H/A boundary back to sustained control by the incoming team."""
    if search_s <= 0:
        return labels, 0
    out = labels.copy()
    width = round(search_s * fps)
    minimum = max(1, round(control_s * fps))
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    changed = 0
    for start, boundary, stop in zip(cuts[:-2], cuts[1:-1], cuts[2:]):
        if labels[start] not in ("H", "A") or labels[boundary] not in ("H", "A"):
            continue
        left = max(start + 1, boundary - width)
        right = min(stop, boundary + minimum - 1)
        if left >= boundary:
            continue

        window = slice(left, right)
        incoming_team = 0 if labels[boundary] == "H" else 1
        distances = np.hypot(px[window] - bx[window, None], py[window] - by[window, None])
        distances = np.where(np.isfinite(distances), distances, np.inf)
        own = np.min(np.where(team_mat[window] == incoming_team, distances, np.inf), axis=1)
        opponent = np.min(np.where(team_mat[window] == 1 - incoming_team, distances, np.inf), axis=1)
        control = (own <= 1.5) & (opponent >= own + 0.75)
        for a, b in reversed(_runs(control)):
            if left + a < boundary and b - a >= minimum:
                out[left + a:boundary] = labels[boundary]
                changed += 1
                break
    return out, changed



def _ball_contact_team(px, py, team_mat, bx, by, max_distance, opponent_margin, gap_frames):
    """Identify the clearly nearest team, bridging only brief gaps within that team."""
    distance = np.hypot(px - bx[:, None], py - by[:, None])
    valid = np.isfinite(distance)
    nearest = [
        np.min(np.where(valid & (team_mat == side), distance, np.inf), axis=1)
        for side in (0, 1)
    ]
    contact = np.full(len(bx), -1, dtype=np.int8)
    for side in (0, 1):
        own, opponent = nearest[side], nearest[1 - side]
        contact[(own <= max_distance) & (opponent >= own + opponent_margin)] = side
    for start, stop in _runs(contact < 0):
        if (0 < start and stop < len(contact) and stop - start <= gap_frames
                and contact[start - 1] == contact[stop]):
            contact[start:stop] = contact[start - 1]
    return contact


def restore_brief_turnovers(labels, p_home, px, py, team_mat, bx, by, fps=25, *, tight_contact=False):
    """Recover a brief opposing possession bracketed by control of the original team."""
    contact_frames = max(1, round((0.12 if tight_contact else 0.16) * fps))
    distance, margin = (0.65, 0.25) if tight_contact else (0.7, 0.3)
    contact = _ball_contact_team(px, py, team_mat, bx, by, distance, margin, contact_frames)

    out = labels.copy()
    support_frames = max(1, round(0.12 * fps))
    max_gap = round((1.0 if tight_contact else 0.8) * fps)
    max_turnover = round(2.0 * fps)
    bounds = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    restored = 0
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if start == stop or labels[start] not in ("H", "A") or stop - start < fps:
            continue
        original = int(labels[start] == "A")
        other = 1 - original
        local = contact[start:stop]
        episodes = sorted((a, b, side) for side in (0, 1) for a, b in _runs(local == side))
        for before, capture, after in zip(episodes, episodes[1:], episodes[2:]):
            a, b, side = capture
            if side != other or before[2] != original or after[2] != original:
                continue
            if b - a < contact_frames or min(before[1] - before[0], after[1] - after[0]) < support_frames:
                continue
            regain = after[0]
            if max(a - before[1], regain - b) > max_gap or regain - a > max_turnover:
                continue
            support_end = regain if tight_contact else b
            home = p_home[start + a:start + support_end].mean()
            support = home if other == 0 else 1 - home
            if support >= (0.5 if tight_contact else 0.35):
                out[start + a:start + regain] = "H" if other == 0 else "A"
                restored += 1
    return out, restored



def advance_turnovers_at_ball_slowdown(labels, px, py, team_mat, bx, by, search_s, fps=25):
    """Move a switch back to a slowdown corroborated by incoming-team ball proximity."""
    step = max(1, round(0.12 * fps))
    window = round(search_s * fps)
    if window <= 0 or len(labels) <= 2 * step:
        return labels, 0

    speed = np.hypot(bx[step:] - bx[:-step], by[step:] - by[:-step]) * (fps / step)
    incoming_speed, outgoing_speed = speed[:-step], speed[step:]
    slowdown = np.zeros(len(labels), dtype=bool)
    slowdown[step:-step] = (incoming_speed >= 2) & (outgoing_speed <= 0.55 * incoming_speed)
    contact = _ball_contact_team(px, py, team_mat, bx, by, 0.75, 0.3, round(0.16 * fps))
    support = max(1, round(0.08 * fps))

    out = labels.copy()
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    changed = 0
    for start, boundary, stop in zip(cuts[:-2], cuts[1:-1], cuts[2:]):
        if labels[start] not in ("H", "A") or labels[boundary] not in ("H", "A"):
            continue
        left = max(start + 1, boundary - window)
        incoming_team = int(labels[boundary] == "A")
        for frame in (np.flatnonzero(slowdown[left:boundary]) + left)[::-1]:
            nearby = contact[max(start, frame - support):min(stop, frame + support + 1)]
            if np.count_nonzero(nearby == incoming_team) >= support:
                out[frame:boundary] = labels[boundary]
                changed += 1
                break
    return out, changed



def delay_turnovers_at_ball_redirect(labels, px, py, team_mat, bx, by, search_s, fps=25):
    """Delay an H/A switch to a nearby ball redirection supported by the incoming team."""
    step = max(1, round(0.12 * fps))
    window = round(search_s * fps)
    if window <= 0 or len(labels) <= 2 * step:
        return labels, 0

    dx = (bx[step:] - bx[:-step]) * (fps / step)
    dy = (by[step:] - by[:-step]) * (fps / step)
    speed = np.hypot(dx, dy)
    incoming, outgoing = speed[:-step], speed[step:]
    dot = dx[:-step] * dx[step:] + dy[:-step] * dy[step:]
    redirect = np.zeros(len(labels), dtype=bool)
    redirect[step:-step] = ((incoming >= 2) & (outgoing >= 1.5)
                            & (dot <= 0.45 * incoming * outgoing))
    support = max(1, round(0.16 * fps))
    contact = _ball_contact_team(px, py, team_mat, bx, by, 1.0, 0.4, support)

    out = labels.copy()
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    changed = 0
    for start, boundary, stop in zip(cuts[:-2], cuts[1:-1], cuts[2:]):
        if labels[start] not in ("H", "A") or labels[boundary] not in ("H", "A"):
            continue
        right = min(stop - 1, boundary + window)
        incoming_team = int(labels[boundary] == "A")
        for frame in np.flatnonzero(redirect[boundary + 1:right + 1]) + boundary + 1:
            nearby = contact[max(start, frame - support):min(stop, frame + support + 1)]
            if np.count_nonzero(nearby == incoming_team) >= support:
                out[boundary:frame] = labels[start]
                changed += 1
                break
    return out, changed



def merge_weak_turnovers(labels, p_home, px, py, team_mat, bx, by, max_s, fps=25):
    """Bridge a brief, unsupported reversal between two spells of the same team."""
    if max_s <= 0:
        return labels, 0
    contact = _ball_contact_team(px, py, team_mat, bx, by, 1.0, 0.4, round(0.16 * fps))
    out = labels.copy()
    limit = round(max_s * fps)
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    merged = 0
    for previous, start, stop in zip(cuts[:-3], cuts[1:-2], cuts[2:-1]):
        side = labels[start]
        neighbor = labels[previous]
        if (side not in ("H", "A") or neighbor not in ("H", "A")
                or neighbor != labels[stop] or stop - start > limit):
            continue
        probability = p_home[start:stop]
        support = probability if side == "H" else 1 - probability
        if support.mean() < 0.7 and not np.any(contact[start:stop] == int(side == "A")):
            out[start:stop] = neighbor
            merged += 1
    return out, merged



def support_contact_team(p_home, px, py, team_mat, bx, by, log_odds, fps=25):
    """Give sustained nearby control a small vote when its team has weak model support."""
    if log_odds <= 0:
        return p_home
    contact = _ball_contact_team(px, py, team_mat, bx, by, 0.75, 0.3, round(0.12 * fps))
    minimum = max(1, round(0.32 * fps))
    odds_factor = np.exp(log_odds)
    adjusted = p_home.copy()
    for side in (0, 1):
        for start, stop in _runs(contact == side):
            if stop - start < minimum:
                continue
            probability = p_home[start:stop] if side == 0 else 1 - p_home[start:stop]
            supported = odds_factor * probability / (1 - probability + odds_factor * probability)
            updated = np.where(probability <= 0.55, supported, probability)
            adjusted[start:stop] = updated if side == 0 else 1 - updated
    return adjusted



def flight_switch_multiplier(flight, seen, scale, tail_s, fps=25):
    """Discourage switching on tracked ball flight and immediately after it."""
    active = np.asarray(flight, dtype=bool).copy()
    tail = max(0, round(tail_s * fps))
    for _, stop in _runs(active):
        active[stop:stop + tail] = True
    return np.where(active & seen, scale, 1.0)



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

def advance_controlled_restarts(labels, p_home, contact, stretches, fps=25):
    """Shorten a non-special stop if the next team already controls the ball."""
    corrected = labels.copy()
    count = 0
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    spans = [(a, b, labels[a]) for a, b in zip(cuts[:-1], cuts[1:]) if a < b]
    for previous, stopped, following in zip(spans, spans[1:], spans[2:]):
        start, end, label = stopped
        next_team = following[2]
        if label != "D" or previous[2] not in ("H", "A") or next_team not in ("H", "A"):
            continue
        if any(s.get("special") and s["start"] < end and s["end"] > start for s in stretches):
            continue
        left = max(start + round(2 * fps), end - round(0.5 * fps))
        if left >= end:
            continue
        team_code = int(next_team == "A")
        support = p_home[left:end] if team_code == 0 else 1 - p_home[left:end]
        for a, b in _runs(contact[left:end] == team_code):
            if b - a >= round(0.12 * fps) and end - left - b <= round(0.2 * fps):
                if support[a:b].mean() >= 0.45:
                    corrected[left + a:end] = next_team
                    count += 1
                    break
    return corrected, count

def restore_repeated_contact_turnovers(labels, p_home, px, py, team_mat, bx, by, fps=25, *, extended=False):
    """Restore brief turnovers supported by separate contacts by the incoming team."""
    if extended:
        distance, margin, bridge, minimum, length, max_gap, peak = 2.4, 0.65, 0.4, 0.36, 5, 1.8, 0.5
    else:
        distance, margin, bridge, minimum, length, max_gap, peak = 1.9, 0.55, 0.55, 0.48, 2, 1, 0.65
    contact = _ball_contact_team(px, py, team_mat, bx, by, distance, margin, 0)
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    out = labels.copy()
    restored = 0
    for start, stop in zip(cuts[:-1], cuts[1:]):
        if stop - start < fps or labels[start] not in ("H", "A"):
            continue
        original = int(labels[start] == "A")
        local = contact[start:stop]
        episodes = sorted((a, b, side) for side in (0, 1) for a, b in _runs(local == side))
        grouped = []
        for a, b, side in episodes:
            if grouped and side == grouped[-1][2] and a - grouped[-1][1] <= round(bridge * fps):
                left, _, _, count, observed = grouped[-1]
                grouped[-1] = (left, b, side, count + 1, observed + b - a)
            else:
                grouped.append((a, b, side, 1, b - a))
        for previous, incoming, following in zip(grouped, grouped[1:], grouped[2:]):
            a, b, side, count, observed = incoming
            regain = following[0]
            if side == original or previous[2] != original or following[2] != original:
                continue
            if count < 2 or observed / fps < minimum or regain - a > length * fps:
                continue
            if min(previous[1] - previous[0], following[1] - following[0]) < round(0.12 * fps):
                continue
            if max(a - previous[1], regain - b) > max_gap * fps:
                continue
            support = p_home[start + a:start + regain]
            if side == 1:
                support = 1 - support
            if support.mean() >= 0.25 and support.max() >= peak:
                out[start + a:start + regain] = "HA"[side]
                restored += 1
    return out, restored

def delay_switches_during_contact(labels, p_home, px, py, team_mat, bx, by, fps=25):
    """Keep the outgoing team briefly when its control continues across a switch."""
    contact = _ball_contact_team(px, py, team_mat, bx, by, 0.8, 0.4, round(0.08 * fps))
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    corrected = labels.copy()
    count = 0
    for start, switch, stop in zip(cuts, cuts[1:], cuts[2:]):
        previous, following = labels[start], labels[switch]
        if previous not in ("H", "A") or following not in ("H", "A"):
            continue
        if min(switch - start, stop - switch) < 1.2 * fps:
            continue
        end = min(stop - round(0.6 * fps), switch + round(0.32 * fps))
        support = p_home[switch:end] if previous == "H" else 1 - p_home[switch:end]
        best = (0.0, 0)
        for a, b in _runs(contact[switch:end] == int(previous == "A")):
            if b - a >= round(0.12 * fps) and a <= round(0.2 * fps) and support[:b].mean() >= 0.6:
                best = max(best, (float((support[:b] - 0.5).sum()), b))
        if best[1]:
            corrected[switch:switch + best[1]] = previous
            count += 1
    return corrected, count

def merge_unsupported_turnovers_before_stops(labels, p_home, px, py, team_mat, bx, by, fps=25):
    """Keep the previous team across unsupported short switches into a stoppage."""
    contact = _ball_contact_team(px, py, team_mat, bx, by, 2.0, 0.4, round(0.12 * fps))
    corrected = labels.copy()
    count = 0
    for stop in np.flatnonzero((labels[1:] == "D") & (labels[:-1] != "D")) + 1:
        current = labels[stop - 1]
        if current not in ("H", "A"):
            continue
        left = max(0, stop - round(4 * fps) - 1)
        changes = np.flatnonzero(labels[left:stop] != current)
        if not len(changes):
            continue
        start = left + changes[-1] + 1
        previous = labels[start - 1]
        if previous not in ("H", "A") or np.any(contact[start:stop] == int(current == "A")):
            continue
        support = p_home[start:stop] if previous == "H" else 1 - p_home[start:stop]
        if support.mean() >= 0.28:
            corrected[start:stop] = previous
            count += 1
    return corrected, count

def merge_contested_reversals(labels, p_home, px, py, team_mat, bx, by, fps=25):
    """Merge brief reversals when the original team has more clear ball contact."""
    contact = _ball_contact_team(px, py, team_mat, bx, by, 1.0, 0.4, 0)
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    corrected = labels.copy()
    count = 0
    for left, start, end, right in zip(cuts, cuts[1:], cuts[2:], cuts[3:]):
        original = labels[left]
        if original not in ("H", "A") or labels[start] not in ("H", "A") or labels[end] != original:
            continue
        if end - start > 3 * fps:
            continue
        side = int(original == "A")
        local = contact[start:end]
        if np.count_nonzero(local == side) < 0.2 * fps or np.count_nonzero(local == 1 - side) > 0.16 * fps:
            continue
        before = contact[max(left, start - round(fps)):start]
        after = contact[end:min(right, end + round(fps))]
        if not (np.any(before == side) or np.any(after == side)):
            continue
        support = p_home[start:end] if side == 0 else 1 - p_home[start:end]
        if support.mean() >= 0.25:
            corrected[start:end] = original
            count += 1
    return corrected, count

def restore_turnovers_before_stops(labels, p_home, px, py, team_mat, bx, by, stretches, fps=25):
    """Recover the final brief possession when contact changes just before a stop."""
    contact = _ball_contact_team(px, py, team_mat, bx, by, 0.8, 0.4, round(0.12 * fps))
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    corrected = labels.copy()
    count = 0
    for start, stop, resume in zip(cuts, cuts[1:], cuts[2:]):
        if labels[start] not in ("H", "A") or labels[stop] != "D" or stop - start < fps:
            continue
        if any(s.get("special") and s["start"] < resume and s["end"] > stop for s in stretches):
            continue
        left = max(start + round(0.6 * fps), stop - round(4 * fps))
        local = contact[left:stop]
        episodes = sorted((a, b, side) for side in (0, 1) for a, b in _runs(local == side))
        if len(episodes) < 2:
            continue
        a, b, side = episodes[-1]
        before, release, previous = episodes[-2]
        if previous != int(labels[start] == "A") or side == previous:
            continue
        if release - before < round(0.12 * fps) or (b - a) / fps < 0.28:
            continue
        if stop - left - a > fps or (stop - left - b) / fps > 0.2 or a - release > fps:
            continue
        support = p_home[left + a:stop] if side == 0 else 1 - p_home[left + a:stop]
        if support.mean() >= 0.35:
            corrected[left + a:stop] = "HA"[side]
            count += 1
    return corrected, count

def merge_weak_possession_at_stops(labels, p_home, contact, fps=25):
    """Merge a brief spell beside a stop when there is little close ball contact."""
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    corrected = labels.copy()
    count = 0
    for left, start, end in zip(cuts, cuts[1:], cuts[2:-1]):
        previous, current, following = labels[left], labels[start], labels[end]
        if current not in ("H", "A") or end - start > 3 * fps:
            continue
        if previous == "D" and following in ("H", "A"):
            keep = following
        elif following == "D" and previous in ("H", "A"):
            keep = previous
        else:
            continue
        if np.count_nonzero(contact[start:end] == int(current == "A")) > 0.16 * fps:
            continue
        support = p_home[start:end] if keep == "H" else 1 - p_home[start:end]
        if support.mean() >= 0.38:
            corrected[start:end] = keep
            count += 1
    return corrected, count

def anchor_switches_to_paired_contacts(labels, p_home, px, py, team_mat, bx, by, fps=25):
    """Anchor a nearby switch to incoming contact following outgoing contact."""
    contact = _ball_contact_team(px, py, team_mat, bx, by, 1.0, 0.4, round(0.08 * fps))
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    corrected = labels.copy()
    count = 0
    for start, switch, end in zip(cuts, cuts[1:], cuts[2:]):
        old, new = labels[start], labels[switch]
        if old not in ("H", "A") or new not in ("H", "A") or min(switch - start, end - switch) < 1.2 * fps:
            continue
        radius = round(0.6 * fps)
        lo = max(start + radius, switch - radius)
        hi = min(end - radius, switch + radius)
        side = int(new == "A")
        local = contact[lo:hi]
        support = p_home[lo:hi] if side == 0 else 1 - p_home[lo:hi]
        minimum = round(0.12 * fps)
        candidates = []
        for a, b in _runs(local == 1 - side):
            if b - a < minimum:
                continue
            for c, d in _runs(local == side):
                if d - c < minimum or not 0 <= c - b <= round(0.7 * fps):
                    continue
                x = lo + c
                if x < switch and support[c:switch - lo].mean() < 0.6:
                    continue
                if x > switch and (1 - support[switch - lo:c]).mean() < 0.6:
                    continue
                candidates.append(x)
        x = min(candidates, key=lambda x: abs(x - switch), default=switch)
        if x != switch:
            corrected[min(switch, x):max(switch, x)] = new if x < switch else old
            count += 1
    return corrected, count

def restore_turnovers_with_unseen_start(labels, p_home, px, py, team_mat, bx, by, fps=25):
    """Recover a short turnover whose opening contact was not tracked."""
    contact = _ball_contact_team(px, py, team_mat, bx, by, 1.9, 0.4, round(0.12 * fps))
    unseen = ~np.isfinite(bx)
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    corrected = labels.copy()
    count = 0
    for start, stop in zip(cuts[:-1], cuts[1:]):
        if stop - start < fps or labels[start] not in ("H", "A"):
            continue
        original = int(labels[start] == "A")
        local = contact[start:stop]
        episodes = sorted((a, b, side) for side in (0, 1) for a, b in _runs(local == side))
        for incoming, regain in zip(episodes, episodes[1:]):
            a, b, side = incoming
            if side == original or regain[2] != original or regain[1] - regain[0] < round(0.12 * fps):
                continue
            if (b - a) / fps < 0.28 or (regain[0] - b) / fps > 0.5:
                continue
            left = start + max(round(0.4 * fps), b - round(5 * fps))
            right = start + a
            if left >= right:
                continue
            support = p_home[left:start + b] if side == 0 else 1 - p_home[left:start + b]
            possible = _runs(unseen[left:right] & (support[:right - left] <= 0.42))
            edge = next((left + y for x, y in reversed(possible) if y - x >= round(0.32 * fps)), None)
            if edge is None or (right - edge) / fps > 0.5 or (start + b - edge) / fps > 2:
                continue
            if np.any(contact[edge:right] >= 0):
                continue
            evidence = support[edge - left:]
            if evidence.mean() >= 0.45 and evidence.max() >= 0.5:
                corrected[edge:start + b] = "HA"[side]
                count += 1
    return corrected, count

def extend_brief_turnover_ends(labels, p_home, px, py, team_mat, bx, by, fps=25, *, longer=False, earlier=False):
    """Extend a brief turnover at its boundary if the other team has no clear contact."""
    if earlier:
        labels, p_home, px, py, team_mat, bx, by = (
            values[::-1] for values in (labels, p_home, px, py, team_mat, bx, by)
        )
    contact = _ball_contact_team(px, py, team_mat, bx, by, 1.4, 0.4, round(0.08 * fps))
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    corrected = labels.copy()
    count = 0
    if earlier:
        min_duration, max_duration = 2 + 1 / fps, 3
        max_extension, min_contact, min_support = 0.64, 0.48, 0.32
    elif longer:
        min_duration, max_duration = 2 + 1 / fps, 6
        max_extension, min_contact, min_support = 0.64, 0.4, 0.35
    else:
        min_duration, max_duration = 0.48, 2
        max_extension, min_contact, min_support = 1.2, 0.32, 0.42
    for left, start, end, right in zip(cuts, cuts[1:], cuts[2:], cuts[3:]):
        original, incoming = labels[left], labels[start]
        if original not in ("H", "A") or incoming not in ("H", "A") or labels[end] != original:
            continue
        duration = (end - start) / fps
        if not min_duration <= duration <= max_duration:
            continue
        side = int(incoming == "A")
        if np.count_nonzero(contact[start:end] == side) / fps < min_contact:
            continue
        limit = min(right - round(0.6 * fps), end + round(max_extension * fps))
        if limit <= end:
            continue
        returning = np.flatnonzero(contact[end:limit] == 1 - side)
        if returning.size:
            limit = end + returning[0]
        support = p_home[end:limit] if side == 0 else 1 - p_home[end:limit]
        for length in range(len(support), max(1, round(0.12 * fps)) - 1, -1):
            if support[:length].mean() >= min_support:
                corrected[end:end + length] = incoming
                count += 1
                break
    return (corrected[::-1] if earlier else corrected), count

def advance_switches_after_ball_departure(labels, p_home, px, py, team_mat, bx, by, fps=25, *, min_departure=0.8):
    """Advance a switch when the ball leaves one team and reaches the other."""
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    corrected = labels.copy()
    count = 0
    step = max(1, round(0.16 * fps))
    for start, switch, end in zip(cuts, cuts[1:], cuts[2:]):
        old, new = labels[start], labels[switch]
        if old not in ("H", "A") or new not in ("H", "A") or min(switch - start, end - switch) < 0.8 * fps:
            continue
        lo = max(start + step, start + round(0.5 * fps), switch - round(0.4 * fps))
        hi = min(end, switch + step)
        window = slice(lo - step, hi)
        distance = np.hypot(px[window] - bx[window, None], py[window] - by[window, None])
        known = np.isfinite(distance)
        side = int(old == "A")
        teams = team_mat[window]
        outgoing = np.min(np.where(known & (teams == side), distance, np.inf), axis=1)
        incoming = np.min(np.where(known & (teams == 1 - side), distance, np.inf), axis=1)
        for t in range(lo, min(switch, end - step + 1)):
            i = t - lo + step
            before = outgoing[i - step:i]
            after = outgoing[i:i + step]
            if not (np.isfinite(before).all() and np.isfinite(after).all()):
                continue
            close = before.mean()
            if close > 0.8 or after.mean() - close < min_departure or incoming[i:].min() > 1.2:
                continue
            support = p_home[t:switch] if new == "H" else 1 - p_home[t:switch]
            if support.mean() >= 0.4:
                corrected[t:switch] = new
                count += 1
                break
    return corrected, count


def delay_switches_until_ball_arrival(labels, p_home, px, py, team_mat, bx, by, fps=25):
    """Delay a switch to the next team's approach, using the reverse-time departure test."""
    corrected, count = advance_switches_after_ball_departure(
        labels[::-1], p_home[::-1], px[::-1], py[::-1], team_mat[::-1], bx[::-1], by[::-1],
        fps, min_departure=1.2,
    )
    return corrected[::-1], count
