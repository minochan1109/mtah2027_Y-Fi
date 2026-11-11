"""[F] Kick-off and goal detection, version 3: players only -- NO camera information is used.

v3 change: the goal signal's third part is chosen once for every goal (P.ko_goal_third):
"oneside" (default: a HUMP of one-sided movement -- scorers gone/away minus conceders), or
"none" / "still" (a lasting step in the players' signal, without or with the conceders'
stillness). The v2 camera part is removed: where the camera looks depends on how a match is
filmed, so it would not carry over to other videos. compare_goal_modes() runs the three
players-only ways on the same kick-offs.
The kick-off part is unchanged from version 2.


Smart Tagging tells us roughly WHEN; the tracking tells us EXACTLY when. The offset between the two
is never written as a constant -- it is found per kick-off from the players (generic rules only):

  A. after a GOAL (most kick-offs)
     1. celebration : the scoring team runs off towards the fans -> it leaves the picture (fewer
                      of its players tracked) and moves away from the centre spot,
                      while the conceding team walks back towards the centre spot
     2. who kicks   : Laws of the Game -> the CONCEDING team kicks off.
                      Goal tag Team > celebration pattern > formation, in that order of trust
     3. dead ball   : from the goal to the kick-off the ball is dead (often a minute or more)
  B. every kick-off (also start of each half, where there is no goal)
     4. formation   : both teams in their own half, the kicking team has a player on the centre
                      spot, every opponent outside the centre circle, almost nobody moving
     5. restart     : first frame after the formation where the players start moving
                      (or the trusted ball leaves the spot, if that is earlier)
  No formation found -> reported as not found; NO fixed offset is used.
"""
import warnings

import numpy as np

# all-NaN frames (halftime, untracked) are expected: their mean is NaN and is handled below
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

KICKOFF_TYPES = ("kick off (start)", "kick off (after goal)")
GOAL_CODE = "GOALS"
CENTRE_CIRCLE_M = 9.15


def _runs(mask):
    """[(a, b), ...] half-open runs of True. Same contract as preprocess.run_lengths."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[::2], d[1::2]))


def _types(g):
    return [t.lower() for grp, t in g["labels"] if grp.lower() == "type"]


def kickoff_tags(tags):
    return sorted(g["start"] for g in tags or [] if any(t in KICKOFF_TYPES for t in _types(g)))


def goal_tags(tags, side_of=None):
    """[(start_frame, scoring_side or None)] -- side_of is the same callable tag_policy uses."""
    out = []
    for g in tags or []:
        if g["code"] == GOAL_CODE:
            side = side_of(g["tags"].get("Team")) if side_of else None
            out.append((g["start"], side))
    return sorted(out, key=lambda x: x[0])


# ---------------------------------------------------------------------------------------------
# A. goal celebration
# ---------------------------------------------------------------------------------------------
def team_tracks(px, py, team_mat, P):
    """(T, 2) players tracked per team, (T, 2) mean distance to the centre spot."""
    L, W = P.pitch_length, P.pitch_width
    dc = np.hypot(px - L / 2, py - W / 2)
    n = np.stack([(team_mat == k).sum(axis=1) for k in (0, 1)], axis=1).astype(float)
    with np.errstate(invalid="ignore"):
        d = np.stack([np.nanmean(np.where(team_mat == k, dc, np.nan), axis=1) for k in (0, 1)], axis=1)
    return n, d


def celebration(n, d, g, T, P):
    """Compare the seconds before the goal with the celebration after it.

    Returns dict with, per team: drop (fraction of players lost from the picture),
    away (metres moved away from the centre spot), and the inferred scorer (0/1/-1).
    """
    pre = slice(max(0, g - P.frames(P.ko_goal_pre_s)), g)
    post = slice(g, min(T, g + P.frames(P.ko_celebrate_s)))
    if pre.stop - pre.start < 1 or post.stop - post.start < 1:
        return dict(drop=[0, 0], away=[0, 0], scorer=-1)
    with np.errstate(invalid="ignore"):
        n0, n1 = np.nanmedian(n[pre], axis=0), np.nanpercentile(n[post], 20, axis=0)
        d0, d1 = np.nanmedian(d[pre], axis=0), np.nanpercentile(d[post], 80, axis=0)
    drop = np.where(n0 > 0, (n0 - n1) / np.maximum(n0, 1), 0.0)
    away = np.nan_to_num(d1 - d0)
    # celebrating team: loses more players from view AND moves away more than the other
    s = (drop[0] - drop[1]) + (away[0] - away[1]) / P.ko_celebrate_scale_m
    scorer = -1 if abs(s) < P.ko_celebrate_min_score else (0 if s > 0 else 1)
    return dict(drop=drop.tolist(), away=away.tolist(), scorer=scorer)


def returning_to_centre(d, k, a, b, P):
    """First frame in [a, b) after which team k's mean distance to the spot stays small."""
    near = np.nan_to_num(d[a:b, k], nan=99.0) < P.ko_return_m
    for r0, r1 in _runs(near):
        if r1 - r0 >= P.frames(P.ko_formation_min_s):
            return a + r0
    return None


# ---------------------------------------------------------------------------------------------
# B. kick-off formation and restart
# ---------------------------------------------------------------------------------------------
def formation_score(px, py, team_mat, attack, speed, P, with_stats=False):
    """Per frame: ok (T,) bool, kick_team (T,) 0/1/-1 (+ the numbers behind it if with_stats)."""
    L, W = P.pitch_length, P.pitch_width
    dc = np.hypot(px - L / 2, py - W / 2)
    st = []
    for k in (0, 1):
        m = team_mat == k
        n = m.sum(axis=1)
        xn = (px - L / 2) * attack[:, k][:, None]                 # + = opponent's half
        with np.errstate(invalid="ignore"):
            own_half = np.where(m, xn <= P.ko_half_margin_m, False).sum(axis=1)
            on_spot = np.where(m, dc < P.ko_spot_m, False).sum(axis=1)
            outside = np.where(m, dc > CENTRE_CIRCLE_M - P.ko_circle_margin_m, False).sum(axis=1)
        frac = lambda c: np.where(n > 0, c / np.maximum(n, 1), 0.0)
        st.append(dict(n=n, own_half=frac(own_half), on_spot=on_spot, outside=frac(outside)))
    with np.errstate(invalid="ignore"):
        still = np.nan_to_num(np.nanmean(speed, axis=1), nan=99.0) < P.ko_still_speed
    enough = (st[0]["n"] >= P.ko_min_players) & (st[1]["n"] >= P.ko_min_players)
    halves = (st[0]["own_half"] >= P.ko_half_frac) & (st[1]["own_half"] >= P.ko_half_frac)
    kick = np.full(len(px), -1, np.int8)
    ok = np.zeros(len(px), bool)
    for k in (0, 1):
        o = 1 - k
        this = (enough & halves & still & (st[k]["on_spot"] >= 1) & (st[o]["on_spot"] == 0)
                & (st[o]["outside"] >= P.ko_circle_frac))
        kick[this & ~ok] = k
        ok |= this
    if with_stats:
        with np.errstate(invalid="ignore"):
            mean_speed = np.nanmean(speed, axis=1)
        return ok, kick, dict(team=st, mean_speed=mean_speed, enough=enough, halves=halves, still=still)
    return ok, kick


def diagnose(px, py, team_mat, attack, speed, P, centre, half_window_s=10.0, step_s=0.5, out=print):
    """Why is (or isn't) there a kick-off formation around `centre`? One row per step_s.

    Each condition is printed with its value; a failing one is marked with '!'.
    Prints only counts and fractions, no positions.
    """
    ok, kick, z = formation_score(px, py, team_mat, attack, speed, P, with_stats=True)
    a = max(0, centre - P.frames(half_window_s))
    b = min(len(px), centre + P.frames(half_window_s) + 1)
    H, A = z["team"]
    mark = lambda good: " " if good else "!"
    out(f"diagnose around frame {centre}  (need: n>={P.ko_min_players}, own half>={P.ko_half_frac}, "
        f"speed<{P.ko_still_speed}, kicker on spot>=1 & other 0, other outside circle>={P.ko_circle_frac})")
    out(f"  {'frame':>7} {'dt s':>6} | {'nH':>3} {'nA':>3} | {'halfH':>6} {'halfA':>6} | {'speed':>6} |"
        f" {'spotH':>5} {'spotA':>5} | {'outH':>5} {'outA':>5} | ok kicks")
    for t in range(a, b, max(1, P.frames(step_s))):
        nH, nA = int(H["n"][t]), int(A["n"][t])
        sp_ = z["mean_speed"][t]
        sH, sA = int(H["on_spot"][t]), int(A["on_spot"][t])
        oH, oA = H["outside"][t], A["outside"][t]
        # which side would kick: the one with somebody on the spot
        kicker = 0 if sH >= 1 and sA == 0 else (1 if sA >= 1 and sH == 0 else -1)
        spot_ok = kicker >= 0
        out_ok = (kicker == 0 and oA >= P.ko_circle_frac) or (kicker == 1 and oH >= P.ko_circle_frac)
        out(f"  {t:>7} {(t - centre) / P.fps:>+6.1f} |"
            f" {nH:>2}{mark(nH >= P.ko_min_players)}{nA:>2}{mark(nA >= P.ko_min_players)}|"
            f" {H['own_half'][t]:>5.2f}{mark(H['own_half'][t] >= P.ko_half_frac)}"
            f"{A['own_half'][t]:>5.2f}{mark(A['own_half'][t] >= P.ko_half_frac)}|"
            f" {sp_:>5.2f}{mark(bool(z['still'][t]))}|"
            f" {sH:>4}{sA:>5}{mark(spot_ok)}|"
            f" {oH:>5.2f}{oA:>5.2f}{mark(out_ok)}|"
            f" {'Y' if ok[t] else '.':>2} {({0: 'H', 1: 'A'}.get(int(kick[t]), '-'))}")


def _restart(ok, kick, mean_sp, ball_spot, a, b, want_team, P):
    """Search [a, b) for the last formation (of want_team if given) and its restart frame."""
    T = len(ok)
    mask = ok[a:b] if want_team < 0 else (ok[a:b] & (kick[a:b] == want_team))
    runs = [(a + r0, a + r1) for r0, r1 in _runs(mask) if r1 - r0 >= P.frames(P.ko_formation_min_s)]
    if not runs:
        return None
    f0, f1 = runs[-1]
    team = want_team if want_team >= 0 else int(np.bincount(kick[f0:f1][kick[f0:f1] >= 0], minlength=2).argmax())
    lo, hi = max(f0, f1 - P.frames(P.ko_refine_s)), min(T, f1 + P.frames(P.ko_refine_s))
    moving = np.flatnonzero(mean_sp[lo:hi] > P.ko_move_speed)
    k = lo + int(moving[0]) if len(moving) else f1
    leave = np.flatnonzero(ball_spot[lo:hi - 1] & ~ball_spot[lo + 1:hi])
    if len(leave) and lo + int(leave[0]) + 1 < k:
        k = lo + int(leave[0]) + 1
    return dict(formation_start=int(f0), kickoff_frame=int(k), team=int(team))


def find_kickoffs(prep, team_mat, bq, tags, P, side_of=None):
    px, py, sp = prep["px"], prep["py"], prep["pspeed"]
    T = len(px)
    L, W = P.pitch_length, P.pitch_width
    ok, kick = formation_score(px, py, team_mat, prep["attack"], sp, P)
    n, d = team_tracks(px, py, team_mat, P)
    with np.errstate(invalid="ignore"):
        mean_sp = np.nan_to_num(np.nanmean(sp, axis=1), nan=0.0)
    bx, by, conf = bq["bx"], bq["by"], bq["conf"]
    ball_spot = np.isfinite(bx) & (conf >= P.ball_conf_min) & \
        (np.hypot(np.nan_to_num(bx) - L / 2, np.nan_to_num(by) - W / 2) < P.ko_spot_m)

    events, used_ko_tags = [], set()
    ko_list = kickoff_tags(tags)

    # A. goals first: the goal anchors a long window and tells us who kicks.
    #    Its kick-off tag is the first kick-off tag after it and before the next goal, as long as
    #    it is within ko_goal_pair_max_s (a VAR check can stop play for minutes).
    goals = goal_tags(tags, side_of)
    for i, (g, tag_scorer) in enumerate(goals):
        cel = celebration(n, d, g, T, P)
        scorer = tag_scorer if tag_scorer is not None else cel["scorer"]
        kicker = 1 - scorer if scorer in (0, 1) else -1
        nxt = goals[i + 1][0] if i + 1 < len(goals) else T
        limit = min(nxt, g + P.frames(P.ko_goal_pair_max_s))
        end = min(T, g + P.frames(P.ko_after_goal_max_s))
        pair = None
        for s in ko_list:
            if g <= s < limit and s not in used_ko_tags:
                pair = s
                used_ko_tags.add(s)
                end = min(T, max(end, s + P.frames(P.ko_search_after_s)))
                break
        start = g
        if kicker >= 0:
            back = returning_to_centre(d, kicker, g, end, P)
            start = back if back is not None else g
        r = _restart(ok, kick, mean_sp, ball_spot, start, end, kicker, P)
        if r is None and kicker >= 0:                              # pattern wrong? any team
            r = _restart(ok, kick, mean_sp, ball_spot, start, end, -1, P)
        ev = dict(kind="after_goal", goal_frame=int(g), tag_frame=int(g), ko_tag_frame=pair,
                  formation_start=None, kickoff_frame=None, team=kicker, scorer=scorer,
                  celebration=cel, offset_s=None, found=False)
        if r:
            ev.update(r, offset_s=(r["kickoff_frame"] - g) / P.fps, found=True)
        events.append(ev)

    # B. remaining kick-off tags (start of each half, or a goal the tagging missed)
    for s in ko_list:
        if s in used_ko_tags:
            continue
        a = max(0, s - P.frames(P.ko_search_before_s))
        b = min(T, s + P.frames(P.ko_search_after_s))
        r = _restart(ok, kick, mean_sp, ball_spot, a, b, -1, P)
        ev = dict(kind="kickoff_tag", goal_frame=None, tag_frame=int(s), ko_tag_frame=int(s), formation_start=None,
                  kickoff_frame=None, team=-1, scorer=None, celebration=None,
                  offset_s=None, found=False)
        if r:
            ev.update(r, offset_s=(r["kickoff_frame"] - s) / P.fps, found=True)
        events.append(ev)

    events.sort(key=lambda e: e["tag_frame"])
    rep = {"events": len(events), "found": sum(e["found"] for e in events),
           "summary": [(e["kind"], e["tag_frame"], e["kickoff_frame"], e["team"], e["offset_s"])
                       for e in events]}
    return events, rep


def _mmss(frame, fps):
    if frame is None:
        return "-"
    sec = frame / fps
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


def _why(e):
    if e["found"]:
        return "kick-off"
    if e.get("rejected_by_goal"):
        return f"NOT kick-off: formation at {e['kickoff_frame']}, but {e['rejected_by_goal']}"
    if e["kind"] == "after_goal":
        cel = e["celebration"] or {}
        if cel.get("scorer", -1) < 0:
            return "NOT kick-off: no formation after the goal, no celebration (goal not given / false tag?)"
        return "NOT kick-off: no formation after the goal"
    return "NOT kick-off: no formation near the tag"


def kickoff_stats(events, P, out=print, csv_path=None):
    """Where a kick-off was / was not judged, as tactical video frames AND video time (mm:ss).

    The confirmed count is whatever the rules give -- it is never forced to a number known about
    this match (that would be human-produced data about the game, RULES.md 2).
    """
    names = {0: "H", 1: "A"}
    n_tag = len(events)
    n_ok = sum(e["found"] for e in events)
    out(f"KICK OFF  tags: {n_tag}  ->  judged kick-off: {n_ok}   not kick-off: {n_tag - n_ok}")
    out(f"  {'#':>2} {'kind':<12} {'tag frame':>9} {'tag time':>9}   {'kickoff frame':>13} "
        f"{'kickoff time':>12} {'kicks':>5}  result")
    rows = []
    for i, e in enumerate(events, 1):
        k = e["kickoff_frame"]
        row = dict(no=i, kind=e["kind"], tag_frame=e["tag_frame"], tag_time=_mmss(e["tag_frame"], P.fps),
                   kickoff_frame="" if k is None else k, kickoff_time="" if k is None else _mmss(k, P.fps),
                   kicks=names.get(e["team"], "") if e["found"] else "",
                   tag_to_kickoff_s="" if e["offset_s"] is None else round(e["offset_s"], 2),
                   result=_why(e))
        rows.append(row)
        out(f"  {i:>2} {e['kind']:<12} {e['tag_frame']:>9} {row['tag_time']:>9}   "
            f"{('-' if k is None else k):>13} {(row['kickoff_time'] or '-'):>12} {(row['kicks'] or '-'):>5}  {row['result']}")
    if csv_path:
        import csv
        import os
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["no"])
            w.writeheader()
            w.writerows(rows)
        out(f"  -> {csv_path}")
    return n_tag, n_ok


# ---------------------------------------------------------------------------------------------
# C. the goal before each kick-off, from the tracking (find_kickoffs above is NOT changed)
# ---------------------------------------------------------------------------------------------
def _smooth(x, w):
    """Moving mean over w frames, NaN-safe (NaN counts as missing)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    k = np.ones(max(1, w))
    num = np.convolve(np.where(ok, x, 0.0), k, "same")
    den = np.convolve(ok.astype(float), k, "same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def goal_signal(n, d, csp, scorer, a, b, P, third="none"):
    """0..1 per frame in [a, b): how much the picture looks like 'just after a goal'.

    Players only -- no camera information is used anywhere in this module.
      gone    share of the scoring team's usual players that are out of the picture
      apart   scoring team far from the centre spot, conceding team near it
    plus, for third="still", the conceding team no longer playing: 1 once its speed is down to
    ko_goal_still_frac of its own in-play speed in this window, 0 at its in-play speed.
    """
    L = P.pitch_length
    conceder = 1 - scorer
    with np.errstate(invalid="ignore"):
        usual = np.nanmax(n[a:b, scorer])
        gone = np.clip(1.0 - n[a:b, scorer] / max(usual, 1.0), 0, 1)
        usual_c = np.nanmax(n[a:b, conceder])
        gone_c = np.clip(1.0 - n[a:b, conceder] / max(usual_c, 1.0), 0, 1)
        apart = np.clip(0.5 + (d[a:b, scorer] - d[a:b, conceder]) / L, 0, 1)
        parts = [gone, apart]
        if third == "still" and csp is not None:
            sm = _smooth(csp[a:b], P.fps)
            ref = np.nanpercentile(sm, P.ko_goal_move_pct) if np.isfinite(sm).any() else np.nan
            if np.isfinite(ref) and ref > 0:
                low = P.ko_goal_still_frac * ref
                parts.append(np.clip((ref - sm) / max(ref - low, 1e-6), 0, 1))
        sig = np.nanmean(np.vstack(parts), axis=0)
    return sig, dict(gone=gone, gone_c=gone_c, apart=apart)


def change_point(sig):
    """Index where the signal steps UP for good: the split that best separates a low part before
    from a high part after (largest drop of the cumulative sum of sig - its mean). No thresholds.
    Returns (index, level before, level after)."""
    x = np.nan_to_num(sig, nan=np.nanmean(sig) if np.isfinite(sig).any() else 0.0)
    c = np.cumsum(x - x.mean())
    i = int(np.argmin(c)) + 1
    if i <= 0 or i >= len(x):
        return None, float("nan"), float("nan")
    return i, float(x[:i].mean()), float(x[i:].mean())


def bump(sig):
    """The stretch [i, j) where the signal stands highest above its own mean (a 'hump' that rises
    and falls again): the largest rise of the cumulative sum of sig - its mean. No thresholds.
    Returns (i, j, mean inside, mean outside)."""
    x = np.nan_to_num(sig, nan=np.nanmean(sig) if np.isfinite(sig).any() else 0.0)
    c = np.r_[0.0, np.cumsum(x - x.mean())]
    best, bi, bj, lo = -np.inf, 0, 0, 0
    for j in range(1, len(c)):
        if c[j] - c[lo] > best:
            best, bi, bj = c[j] - c[lo], lo, j
        if c[j] < c[lo]:
            lo = j
    if bj <= bi:
        return None, None, float("nan"), float("nan")
    inside = float(x[bi:bj].mean())
    rest = np.r_[x[:bi], x[bj:]]
    return bi, bj, inside, float(rest.mean()) if len(rest) else 0.0


def find_goal_kickoff_sets(events, prep, team_mat, bq, P, tags=None, side_of=None, third=None,
                           half_of=None):
    """For every kick-off find_kickoffs() FOUND, look back for the goal that led to it.

    The conceding team kicks off (Laws of the Game), so the scorer is the other team. Between the
    previous kick-off and this one, the goal is where goal_signal() steps up for good
    (change_point or, for 'oneside', bump). The step must be at least ko_goal_min_step, else: no goal (start of a half).
    Then the moment is refined: the last trusted ball in the conceding team's goal mouth shortly
    before the step; no ball -> the frame the scoring team was closest to that goal (the shot);
    neither -> the step itself. GOALS tags are used only to compare at the end.
    """
    px, py = prep["px"], prep["py"]
    T = len(px)
    L, W = P.pitch_length, P.pitch_width
    n, d = team_tracks(px, py, team_mat, P)
    bx, by, conf = bq["bx"], bq["by"], bq["conf"]
    found = sorted((e for e in events if e["found"]), key=lambda e: e["kickoff_frame"])

    sets, prev = [], None
    for e in found:
        k, kicker = e["kickoff_frame"], e["team"]
        row = dict(kickoff_frame=k, kicker=kicker, goal_frame=None, goal_from=None, scorer=None,
                   step=None, asym=None, asym_gone=None, asym_pos=None, dead_s=None, goal_tag=None,
                   goal_shot=None, goal_onset=None, goal_stop=None, refine_note="", tag_match="-")
        # Laws of the Game: every half starts with a kick-off and a goal can only come after one.
        # So the first kick-off found in a half never follows a goal (walk-out, warm-up and
        # halftime movement can look one-sided), and a goal is searched only after the previous
        # kick-off of the SAME half. half_of: frame -> half (None = no half information).
        same_half_prev = prev if (prev is not None and (half_of is None or half_of(prev) == half_of(k))) else None
        prev = k
        if half_of is not None and same_half_prev is None:
            row["tag_match"] = "first kick-off of the half: no goal before it"
            sets.append(row)
            continue
        a = max(0, k - P.frames(P.ko_goal_search_s), same_half_prev if same_half_prev is not None else 0)
        b = e["formation_start"] if e["formation_start"] else k
        if kicker not in (0, 1) or b - a < P.fps:
            row["tag_match"] = "no goal looked for"
            sets.append(row)
            continue
        scorer = 1 - kicker
        with np.errstate(invalid="ignore"):                   # conceding team's mean speed
            csp_all = np.nanmean(np.where(team_mat[a:b] == 1 - scorer, prep["pspeed"][a:b], np.nan), axis=1)
        csp_full = np.full(T, np.nan)
        csp_full[a:b] = csp_all
        mode = third or getattr(P, "ko_goal_third", "oneside")
        if mode not in ("oneside", "none", "still"):
            raise ValueError(f"ko_goal_third must be 'oneside', 'none' or 'still', not {mode!r}")
        sig, parts = goal_signal(n, d, csp_full, scorer, a, b, P,
                                 third="none" if mode == "oneside" else mode)
        if mode == "oneside":
            # v3: the one-sided movement itself, frame by frame -- scorers out of the picture and
            # away from the centre MINUS the conceders. It rises at the goal and falls again once
            # both teams line up for the kick-off (symmetric), so look for a HUMP, not a step.
            with np.errstate(invalid="ignore"):
                one = (parts["gone"] - parts["gone_c"]) + 2.0 * (parts["apart"] - 0.5)
            i, j_end, inside, outside = bump(_smooth(one, P.fps))
            step = inside - outside if i is not None else 0.0
            row["step"] = round(step, 2)
            if i is None or step < P.ko_goal_min_step:
                row["tag_match"] = "no goal: no one-sided movement (half start, or a stoppage)"
                sets.append(row)
                continue
            with np.errstate(invalid="ignore"):
                asym_gone = np.nanmean(parts["gone"][i:j_end]) - np.nanmean(parts["gone_c"][i:j_end])
                asym_pos = 2.0 * (np.nanmean(parts["apart"][i:j_end]) - 0.5)
        else:
            i, lo, hi = change_point(_smooth(sig, P.fps))
            step = hi - lo if i is not None else 0.0
            row["step"] = round(step, 2)
            if i is None or step < P.ko_goal_min_step:
                row["tag_match"] = "no goal before it (half start?)"
                sets.append(row)
                continue
            # a goal is ONE-SIDED: the scorers leave the picture / go to their fans, the conceders
            # go to the centre. A stoppage (drinks, injury) is two-sided: both teams do the same.
            with np.errstate(invalid="ignore"):
                asym_gone = np.nanmean(parts["gone"][i:]) - np.nanmean(parts["gone_c"][i:])
                asym_pos = 2.0 * (np.nanmean(parts["apart"][i:]) - 0.5)
        asym = float(np.nan_to_num(asym_gone) + np.nan_to_num(asym_pos))
        row.update(asym=round(asym, 2), asym_gone=round(float(np.nan_to_num(asym_gone)), 2),
                   asym_pos=round(float(np.nan_to_num(asym_pos)), 2))
        if asym < P.ko_goal_min_asym:
            row["tag_match"] = "no goal: both teams moved alike (stoppage, drinks?)"
            sets.append(row)
            continue
        c0 = a + i
        goal_x = 0.0 if prep["attack"][min(k, T - 1), kicker] > 0 else L   # conceder's goal
        lo_ = max(a, c0 - P.frames(P.ko_goal_ball_before_s))
        hi_ = min(T, c0 + P.fps)
        with np.errstate(invalid="ignore"):
            mouth = (np.isfinite(bx[lo_:hi_]) & (conf[lo_:hi_] >= P.ball_conf_min)
                     & (np.abs(bx[lo_:hi_] - goal_x) < P.ko_goal_mouth_m)
                     & (np.abs(by[lo_:hi_] - W / 2) < P.ko_goal_half_width_m))
        arrivals = _runs(mouth)
        if arrivals:                                   # 1st choice: the ball ARRIVING in the goal mouth
            g, how = lo_ + int(arrivals[-1][0]), "ball in goal mouth"   # (last arrival; it may lie there)
        else:                                          # 2nd: the scorers closest to that goal = the shot
            with np.errstate(invalid="ignore"):
                dg = np.hypot(px[lo_:c0 + 1] - goal_x, py[lo_:c0 + 1] - W / 2)
                near = np.nanmin(np.where(team_mat[lo_:c0 + 1] == scorer, dg, np.inf), axis=1)
            attacks = _runs(near < P.ko_goal_attack_m)      # scorers inside the box
            if attacks:                                  # the LAST attack before the celebration
                r0, r1 = attacks[-1]
                g, how = lo_ + r0 + int(np.argmin(near[r0:r1])), "scorers nearest goal"
            else:                                      # 3rd: where the celebration starts
                g, how = c0, "celebration starts"
        # the shot is not yet the goal: once the ball is in the net the conceding team (keeper,
        # defenders) slows down for good. Three candidates are kept, and ONE rule for every goal
        # (P.ko_goal_moment) picks which is reported:
        #   shot  = the frame above (ball arriving in the goal mouth / scorers nearest the goal)
        #   onset = where the conceders START to slow (last frame still at their 'before' speed)
        #   stop  = where they ARE slow (the strongest step down of their mean speed)
        shot, onset, stop, why_not = g, None, None, ""
        stop_end = c0 if c0 - g >= P.fps else b
        if how == "celebration starts":
            why_not = "no shot found"
        elif stop_end - g < P.fps:
            why_not = "under 1 s to search"
        else:
            with np.errstate(invalid="ignore"):
                csp = np.nanmean(np.where(team_mat[g:stop_end] == 1 - scorer,
                                          prep["pspeed"][g:stop_end], np.nan), axis=1)
            sm = _smooth(csp, max(1, P.fps // 2))
            j, before, after = change_point(-sm)
            if j is None:
                why_not = "no step in their speed"
            elif not (-after < -before):
                why_not = "they did not slow down"
            else:
                fast = np.flatnonzero(np.nan_to_num(sm[:j], nan=0.0) >= -before)
                onset = g + (int(fast[-1]) if len(fast) else j)
                stop = g + j
        moment = getattr(P, "ko_goal_moment", "onset")   # older configs lack it
        pick = {"shot": shot, "onset": onset, "stop": stop}.get(moment)
        if pick is not None and pick != shot:
            g, how = pick, f"{how} + {moment}"
        row.update(goal_shot=int(shot), goal_onset=onset, goal_stop=stop, refine_note=why_not)
        row.update(goal_frame=int(g), goal_from=how, scorer=int(scorer), dead_s=(k - g) / P.fps)
        sets.append(row)

    # compare with the GOALS tags only now
    gt = goal_tags(tags, side_of) if tags else []
    used = set()
    for r in sets:
        if r["goal_frame"] is None:
            continue
        near = [(abs(f - r["goal_frame"]), f, side) for f, side in gt
                if abs(f - r["goal_frame"]) <= P.frames(P.ko_tag_match_s) and f not in used]
        if near:
            _, f, side = min(near)
            used.add(f)
            r["goal_tag"] = f
            r["tag_match"] = "tag + tracking" if side in (None, r["scorer"]) else "tag + tracking (team differs)"
        elif gt:
            r["tag_match"] = "tracking only"
    for f, side in gt:
        if f not in used:
            sets.append(dict(kickoff_frame=None, kicker=None, goal_frame=None, goal_from=None,
                             scorer=side, step=None, asym=None, asym_gone=None, asym_pos=None,
                             dead_s=None, goal_tag=f, goal_shot=None, goal_onset=None,
                             goal_stop=None, refine_note="",
                             tag_match="tag only (tracking found no goal here)"))
    sets.sort(key=lambda r: r["kickoff_frame"] if r["kickoff_frame"] is not None else r["goal_tag"])
    return sets


def compare_goal_modes(events, prep, team_mat, bq, P, tags=None, side_of=None,
                       modes=("oneside", "none", "still"), out=print, half_of=None):
    """One line per found kick-off and mode: goal time, step, asym and the verdict."""
    import copy
    m = lambda f: "-" if f is None else _mmss(f, P.fps)
    res = {}
    for mode in modes:
        res[mode] = find_goal_kickoff_sets(copy.deepcopy(events), prep, team_mat, bq, P, tags=tags,
                                           side_of=side_of, third=mode, half_of=half_of)
    out("COMPARE the third part of the goal signal  (goal time / step / asym)")
    out(f"  {'kick-off':>9}  " + "  ".join(f"{mo:^30}" for mo in res))
    ks = sorted({r["kickoff_frame"] for rows in res.values() for r in rows if r["kickoff_frame"] is not None})
    for k in ks:
        cells = []
        for mo, rows in res.items():
            r = next((x for x in rows if x["kickoff_frame"] == k), None)
            if r is None:
                cells.append(f"{'':30}")
            elif r["goal_frame"] is not None:
                cells.append(f"{m(r['goal_frame']):>9} {r['step']!s:>5} {r['asym']!s:>5} {'goal':<8}")
            else:
                verdict = ("stoppage" if "moved alike" in r["tag_match"]
                           else "half start" if "first kick-off" in r["tag_match"] else "no goal")
                st = "-" if r["step"] is None else r["step"]
                asy = "-" if r.get("asym") is None else r["asym"]
                cells.append(f"{'-':>9} {st!s:>5} {asy!s:>5} {verdict:<8}")
        out(f"  {m(k):>9}  " + "  ".join(cells))
    return res


def confirm_kickoffs_with_goals(events, sets):
    """A kick-off found after a GOALS tag counts only if the tracking found a goal before it.

    A stoppage (drinks, injury) can end in a picture that looks like a kick-off, and a wrong GOALS
    tag would then make it one. Kick-offs at the start of a half (kickoff tags) need no goal.
    Changes `events` in place and returns how many were turned down.
    """
    by_k = {r["kickoff_frame"]: r for r in sets if r["kickoff_frame"] is not None}
    n = 0
    for e in events:
        if not e["found"] or e["kind"] != "after_goal":
            continue
        r = by_k.get(e["kickoff_frame"])
        if r is not None and r["goal_frame"] is None:
            e["found"] = False
            e["rejected_by_goal"] = r["tag_match"]
            n += 1
    return n


def goal_kickoff_table(sets, P, out=print, csv_path=None):
    """Print the sets as frames and mm:ss; optionally save a CSV."""
    names = {0: "H", 1: "A"}
    m = lambda f: "-" if f is None else _mmss(f, P.fps)
    n_goal = sum(s_["goal_frame"] is not None for s_ in sets)
    n_ko = sum(s_["kickoff_frame"] is not None for s_ in sets)
    out(f"GOAL -> KICK OFF  kick-offs: {n_ko}   with a goal found before them: {n_goal}")
    out(f"  {'#':>2} {'goal':>9} {'(frame)':>8} {'kick-off':>9} {'(frame)':>8} {'dead s':>7} "
        f"{'scored':>6} {'kicks':>5} {'step':>5} {'asym':>5}  {'goal from':<20} result / check with tags")
    rows = []
    for i, s_ in enumerate(sets, 1):
        row = dict(no=i, step=s_["step"], asym=s_.get("asym"), asym_gone=s_.get("asym_gone"),
                   asym_pos=s_.get("asym_pos"), goal_time=m(s_["goal_frame"]), goal_frame=s_["goal_frame"],
                   kickoff_time=m(s_["kickoff_frame"]), kickoff_frame=s_["kickoff_frame"],
                   dead_s=None if s_["dead_s"] is None else round(s_["dead_s"], 1),
                   scored=names.get(s_["scorer"], ""), kicks=names.get(s_["kicker"], ""),
                   goal_from=s_["goal_from"] or "", goal_tag=s_["goal_tag"], check=s_["tag_match"])
        rows.append(row)
        out(f"  {i:>2} {row['goal_time']:>9} {('-' if s_['goal_frame'] is None else s_['goal_frame']):>8} "
            f"{row['kickoff_time']:>9} {('-' if s_['kickoff_frame'] is None else s_['kickoff_frame']):>8} "
            f"{('-' if row['dead_s'] is None else row['dead_s']):>7} {row['scored'] or '-':>6} "
            f"{row['kicks'] or '-':>5} {('-' if row['step'] is None else row['step']):>5} "
            f"{('-' if row['asym'] is None else row['asym']):>5}  {row['goal_from'] or '-':<20} {row['check']}")
    cand = [s_ for s_ in sets if s_.get("goal_shot") is not None]
    if cand:
        moment = getattr(P, "ko_goal_moment", "onset")
        out(f"\n  goal moment candidates (reported: ko_goal_moment = '{moment}')")
        out(f"  {'kick-off':>9}   {'shot':>9} {'onset':>9} {'stop':>9}   note")
        for s_ in cand:
            out(f"  {m(s_['kickoff_frame']):>9}   {m(s_['goal_shot']):>9} {m(s_['goal_onset']):>9} "
                f"{m(s_['goal_stop']):>9}   {s_.get('refine_note') or ''}")
        for row, s_ in zip(rows, sets):
            row.update(goal_shot=s_.get("goal_shot"), goal_onset=s_.get("goal_onset"),
                       goal_stop=s_.get("goal_stop"), refine_note=s_.get("refine_note", ""))
    if csv_path and rows:
        import csv
        import os
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        out(f"  -> {csv_path}")
    return rows


def match_timeline(events, sets, half_of, P, names=None, out=print, csv_path=None):
    """The final answer, in time order: each half's opening kick-off and every goal -> kick-off set.

    events   : find_kickoffs() after confirm_kickoffs_with_goals() (only `found` ones are used)
    sets     : find_goal_kickoff_sets()
    half_of  : frame -> 1 or 2
    Kick-offs turned down (a stoppage, a wrong goal tag) and goal tags the tracking did not confirm
    are left out. A kick-off with no goal before it that is NOT its half's first is reported as
    such, so nothing found is hidden.
    """
    names = names or {0: "H", 1: "A"}
    m = lambda f: "-" if f is None else _mmss(f, P.fps)
    by_k = {r["kickoff_frame"]: r for r in sets if r["kickoff_frame"] is not None}
    rows, seen_half = [], set()
    for e in sorted((e for e in events if e["found"]), key=lambda e: e["kickoff_frame"]):
        k = e["kickoff_frame"]
        r = by_k.get(k, {})
        half = half_of(k)
        if r.get("goal_frame") is not None:
            kind = "goal -> kick-off"
        elif half not in seen_half:
            kind = "1st half start" if half == 1 else "2nd half start"
        else:
            kind = "kick-off (no goal found)"
        seen_half.add(half)
        g = r.get("goal_frame")
        rows.append(dict(event=kind, half=half,
                         goal_time="" if g is None else m(g), goal_frame="" if g is None else g,
                         scored="" if g is None else names.get(r.get("scorer"), ""),
                         kickoff_time=m(k), kickoff_frame=k, kicks=names.get(e["team"], ""),
                         dead_s="" if g is None else round((k - g) / P.fps, 1)))
    out(f"MATCH EVENTS  ({len(rows)})")
    out(f"  {'#':>2}  {'event':<24} {'goal':>9} {'(frame)':>8}  {'scored':<14} "
        f"{'kick-off':>9} {'(frame)':>8}  {'kicks':<14}")
    for i, r in enumerate(rows, 1):
        out(f"  {i:>2}  {r['event']:<24} {r['goal_time'] or '-':>9} {str(r['goal_frame']) or '-':>8}  "
            f"{r['scored'] or '-':<14} {r['kickoff_time']:>9} {r['kickoff_frame']:>8}  {r['kicks'] or '-':<14}")
    if csv_path and rows:
        import csv
        import os
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        out(f"  -> {csv_path}")
    return rows


def apply_to_logE(logE, events, P, D=2):
    """dead from the goal (or the formation) to the kick-off, alive and kicking team just after.

    logE (T, 3) in the order H, A, D. Modified in place and returned.
    """
    T = len(logE)
    for e in events:
        if not e["found"]:
            continue
        k = e["kickoff_frame"]
        a = e["goal_frame"] if e["goal_frame"] is not None else e["formation_start"]
        logE[max(0, a):k, D] += P.ko_dead_bonus
        b = min(T, k + P.frames(P.ko_live_s))
        logE[k:b, :D] += P.ko_live_bonus
        if e["team"] >= 0:
            logE[k:b, e["team"]] += P.ko_team_bonus
    return logE


def since_until_features(events, n_frames, P, cap=120.0):
    """Extra columns for dead_extra(): seconds since / until the kick-off, and goal-to-kick-off."""
    t = np.arange(n_frames)
    since = np.full(n_frames, np.inf)
    until = np.full(n_frames, np.inf)
    after_goal = np.zeros(n_frames)
    for e in events:
        if not e["found"]:
            continue
        k = e["kickoff_frame"]
        since[t >= k] = np.minimum(since[t >= k], (t[t >= k] - k) / P.fps)
        until[t < k] = np.minimum(until[t < k], (k - t[t < k]) / P.fps)
        if e["goal_frame"] is not None:
            after_goal[e["goal_frame"]:k] = 1.0
    return {"ko_since_s": np.minimum(since, cap), "ko_until_s": np.minimum(until, cap),
            "ko_after_goal": after_goal}