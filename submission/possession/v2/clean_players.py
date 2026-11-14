"""[A] Remove false player detections and fix the goalkeeper's team.

Input : Tracking (team per SLOT), attack direction per frame (from preprocess.attack_sign),
        optional keeper_gt: known goalkeeper boxes per team.
Output: px, py with false detections set to NaN, and team_mat (T, S): the team of each slot IN
        EACH FRAME (0 home, 1 away, -1 removed). Later modules use team_mat instead of the
        fixed slot team, so a goalkeeper that the tracker put in the wrong team can be corrected.

Steps (all generic rules, the same for every match):
  1. outside   : far behind a goal line or outside a touch line
  1b. GT keeper: the detection that overlaps the known keeper box IS the keeper: give it the right
                 team (also when overlapping boxes swapped the teams), protect it from every
                 static rule, and remove duplicate detections on the same box
  1c. candidate: where no box is known, the deepest object of each keeper zone is a keeper candidate
  1d. lone     : exactly ONE detection in the penalty area + lone_margin_m of a goal = its keeper:
                 protected from rules 1 and 2, gets the defending team in step 3, kept by step 4
  2. static    : not moving for a long time
                   - near a goal line (goal post, net, ball boy ...) or anywhere for a very long time
                   - in the penalty area + margin of either goal       (rule 1)
                   - near the penalty area of the goal the team ATTACKS (rule 2)
                 rules 1 and 2 never remove a GT keeper or a keeper candidate
  2b. swaps    : a short track that continues another slot's track (both ends) gets that slot's team
                 (only with swap_link)
  3. keeper    : the deepest object in a goalkeeper zone belongs to the WRONG team for a while
                 -> it is the goalkeeper, give it the defending team (only where no GT box exists)
  4. max 11    : still more than 11 in a team -> drop the most suspicious (stillest) ones
"""
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from ..preprocess import run_lengths, velocity


def _bridge_gaps(mask, max_gap):
    """Set to True the False-runs of at most max_gap frames that have True on both sides
    (speed noise must not cut a long still run into pieces)."""
    out = mask.copy()
    if max_gap <= 0:
        return out
    for a, b in run_lengths(~mask):
        if b - a <= max_gap and a > 0 and b < len(mask):
            out[a:b] = True
    return out


def _still_run_seconds(still, fps):
    """(T, S): length in seconds of the still-run each frame belongs to (0 when moving)."""
    out = np.zeros(still.shape)
    for s in range(still.shape[1]):
        for a, b in run_lengths(still[:, s]):
            out[a:b, s] = (b - a) / fps
    return out


def _link_swaps(px, py, team_mat, P):
    """Team of a short track = team of the track it continues.

    A tracker that swaps a player into a slot of the other team shows it as: the player's own slot
    stops at frame a, a slot of the other team starts at frame a at (almost) the same place, and the
    own slot picks the player up again where the short run ends. So a short run (<= swap_max_s) that
    starts right after another slot's run ended nearby (and/or ends right before another slot's run
    starts nearby) gets the team of that slot. Needs only positions, no labels."""
    T, S = px.shape
    gap = max(1, int(round(P.swap_link_gap_s * P.fps)))
    maxlen = int(round(P.swap_max_s * P.fps))
    fin = np.isfinite(px)
    runs = []                                          # (start, end, slot)
    for s in range(S):
        for a, b in run_lengths(fin[:, s]):
            runs.append((a, b, s))
    ends_at, starts_at = {}, {}                        # frame -> [(slot, x, y)]
    for a, b, s in runs:
        starts_at.setdefault(a, []).append((s, px[a, s], py[a, s]))
        ends_at.setdefault(b, []).append((s, px[b - 1, s], py[b - 1, s]))   # b = first frame after the run

    def nearest(table, frame_lo, frame_hi, s, x, y):
        best, bd = None, P.swap_link_dist_m
        for f in range(frame_lo, frame_hi):
            for s2, x2, y2 in table.get(f, ()):
                if s2 == s:
                    continue
                d = np.hypot(x - x2, y - y2)
                if d <= bd:
                    best, bd = s2, d
        return best

    fixed = 0
    for a, b, s in sorted(runs):
        if b - a > maxlen or a == 0 or b >= T:
            continue
        pred = nearest(ends_at, max(1, a - gap + 1), a + 1, s, px[a, s], py[a, s])       # ended at a-gap+1 .. a
        succ = nearest(starts_at, b, b + gap, s, px[b - 1, s], py[b - 1, s])             # starts at b .. b+gap-1
        teams = []
        if pred is not None:
            teams.append(int(team_mat[a - 1, pred]))
        if succ is not None:
            teams.append(int(team_mat[min(T - 1, b), succ]))
        teams = [k for k in teams if k >= 0]
        if not teams or (P.swap_need_both and len(teams) < 2) or len(set(teams)) != 1:
            continue
        k = teams[0]
        seg = slice(a, b)
        fixed += int((team_mat[seg, s] != k).sum())
        team_mat[seg, s] = k
    return fixed


def _iou(px, py, box, half):
    """(T, S): IoU of a square box (side 2*half) centred on each position and the GT box (T, 4)."""
    x1, y1, x2, y2 = (box[:, i][:, None] for i in range(4))
    iw = np.clip(np.minimum(x2, px + half) - np.maximum(x1, px - half), 0, None)
    ih = np.clip(np.minimum(y2, py + half) - np.maximum(y1, py - half), 0, None)
    inter = iw * ih
    union = (x2 - x1) * (y2 - y1) + (2 * half) ** 2 - inter
    return np.where(union > 0, inter / union, 0.0)


def clean_players(tr, attack, P, keeper_gt=None, event_mask=None):
    """keeper_gt: None, or {team: (T, 4) array of x1,y1,x2,y2 in pitch metres (NaN = unknown)}.
    If None, tr.keeper_gt is used when it exists.
    event_mask: None, or (T,) bool = frames inside a set piece / goal tag (tag_policy.event_frames).
    With P.event_skip, rule 1 (penalty area + margin, both ends) ignores those frames: they do not
    count towards the still time and are not removed by rule 1."""
    T, S = tr.px.shape
    L, W = P.pitch_length, P.pitch_width
    rows = np.arange(T)
    px = tr.px.astype(np.float64).copy()
    py = tr.py.astype(np.float64).copy()
    team_mat = np.repeat(tr.team[None, :].astype(np.int8), T, axis=0)
    team_mat[np.isnan(px)] = -1
    rep = {"tracked_before": int(np.isfinite(px).sum())}

    # 1. outside the pitch
    out = ((px < -P.goal_line_margin_m) | (px > L + P.goal_line_margin_m)
           | (py < -P.side_line_margin_m) | (py > W + P.side_line_margin_m))
    rep["removed_outside"] = int(out.sum())
    px[out] = py[out] = np.nan
    team_mat[np.isnan(px)] = -1

    # 1b. known goalkeeper box
    if keeper_gt is None:
        keeper_gt = getattr(tr, "keeper_gt", None)
    protect_gt = np.zeros((T, S), bool)              # confirmed keeper: protected from every static rule
    has_gt = np.zeros((T, 2), bool)                  # [:, k] = the box of team k's keeper is known
    rep["gt_team_fixed"] = 0
    rep["gt_dup_removed"] = 0
    if keeper_gt is not None and P.use_keeper_gt:
        for k in (0, 1):
            if keeper_gt.get(k) is None:
                continue
            box = np.asarray(keeper_gt[k], dtype=np.float64)
            ok = np.isfinite(box).all(axis=1)
            has_gt[:, k] = ok
            iou = _iou(px, py, np.nan_to_num(box), P.det_box_m / 2)
            iou[~ok] = 0.0
            iou[~np.isfinite(px)] = 0.0
            best = iou.argmax(axis=1)
            hit = ok & (iou[rows, best] >= P.keeper_iou_min)
            r, sb = rows[hit], best[hit]
            rep["gt_team_fixed"] += int((team_mat[r, sb] != k).sum())
            team_mat[r, sb] = k                      # right team, even if overlap swapped it
            protect_gt[r, sb] = True
            # another detection on (almost) the same box = duplicate -> not a real player
            dup = (iou >= P.keeper_dup_iou) & ~protect_gt & ok[:, None]
            rep["gt_dup_removed"] += int(dup.sum())
            px[dup] = py[dup] = np.nan
            team_mat[dup] = -1

    # 1c. keeper candidates where no box is known: deepest object of each keeper zone (any team)
    protect_pen = protect_gt.copy()                  # protected from rules 1 and 2 only
    if P.use_keeper_candidate:
        for end_x in (0.0, L):
            depth = np.abs(px - end_x)
            in_zone = (depth < P.keeper_zone_depth_m) & (np.abs(py - W / 2) < P.keeper_zone_half_width_m)
            deepest = np.where(np.isfinite(depth), depth, np.inf).argmin(axis=1)
            cand = in_zone[rows, deepest]
            protect_pen[rows[cand], deepest[cand]] = True

    # 1d. lone player: exactly ONE detection in the penalty area (+ lone_margin_m) of a goal for at least
    #     lone_gk_s seconds = that is the goalkeeper (any team label). It is protected from rules 1 and 2
    #     and (step 3) gets the team that defends this goal.
    lone_mark = {}
    lone_any = np.zeros((T, S), bool)
    rep["lone_gk_frames"] = 0
    if P.lone_gk:
        dpt = P.penalty_depth_m + P.lone_margin_m
        wid = P.penalty_half_width_m + P.lone_margin_m
        need = P.frames(P.lone_gk_s)
        for end_x in (0.0, L):
            in_a = np.isfinite(px) & (np.abs(px - end_x) < dpt) & (np.abs(py - W / 2) < wid)
            only = np.where(in_a.sum(axis=1) == 1, in_a.argmax(axis=1), -1)
            mark = np.zeros((T, S), bool)
            for s in np.unique(only[only >= 0]):
                for a, b in run_lengths(only == s):
                    if b - a >= need:
                        mark[a:b, s] = True
            lone_mark[end_x] = mark
            lone_any |= mark
        protect_pen |= lone_any
        rep["lone_gk_frames"] = int(lone_any.sum())

    # 2. static objects
    hw = P.frames(P.velocity_half_window_s)
    sp = np.hypot(velocity(px, hw, P.fps), velocity(py, hw, P.fps))
    still = np.isfinite(px) & (np.nan_to_num(sp, nan=0.0) < P.static_speed)
    still_s = _still_run_seconds(still, P.fps)
    near_goal = np.minimum(np.abs(px), np.abs(L - px)) < P.static_goal_zone_m

    # penalty area + margin at each end
    pen_d = P.penalty_depth_m + P.penalty_margin_m
    pen_w = P.penalty_half_width_m + P.penalty_margin_m
    lateral = np.abs(py - W / 2) < pen_w
    near0 = (px < pen_d) & lateral                    # around the goal at x = 0
    nearL = (px > L - pen_d) & lateral                # around the goal at x = L
    in_pen = near0 | nearL                            # rule 1: either end
    # attack[:, k] = +1: team k attacks +x (its own goal is x = 0), so the goal it attacks is x = L
    atk = np.where(team_mat == 0, attack[:, [0]], attack[:, [1]])
    opp_pen = np.where(atk > 0, nearL, near0)         # rule 2: the goal this team attacks

    with np.errstate(invalid="ignore"):
        live = np.nanmean(sp, axis=1) > P.static_live_speed          # the others are playing
    static = np.zeros_like(still)
    gap = int(round(P.static_gap_s * P.fps))                          # 0 = strict (as before)
    for s in range(S):
        for a, b in run_lengths(_bridge_gaps(still[:, s], gap)):
            if protect_gt[a:b, s].mean() > 0.5:
                continue                                              # confirmed keeper: keep
            dur = (b - a) / P.fps
            if not live[a:b].mean() > 0.5:
                continue                                              # whole game paused: keep
            cand = protect_pen[a:b, s].mean() > 0.5                   # keeper candidate
            if (dur >= P.static_anywhere_s
                    or (dur >= P.static_near_goal_s and near_goal[a:b, s].mean() > 0.5)
                    or (not cand and dur >= P.static_opp_pen_s and opp_pen[a:b, s].mean() > 0.5)):
                static[a:b, s] = True
                continue
            if not cand and in_pen[a:b, s].mean() > 0.5:              # rule 1
                ev = event_mask[a:b] if (event_mask is not None and P.event_skip) else np.zeros(b - a, bool)
                if (b - a - int(ev.sum())) / P.fps >= P.static_pen_s:  # time outside set pieces / goals
                    static[a:b, s] = ~ev
    rep["removed_static"] = int(static.sum())
    px[static] = py[static] = np.nan
    team_mat[np.isnan(px)] = -1

    # 2b. a short track that continues another slot's track belongs to that slot's team
    rep["swap_frames_linked"] = _link_swaps(px, py, team_mat, P) if P.swap_link else 0

    # 3. goalkeeper on the wrong team (heuristic; only where the keeper box is not known)
    # team k attacks +x (attack[:,k] = +1) -> it defends the goal at x = 0
    reassigned = 0
    followed = 0
    for end_x in (0.0, L):
        depth = np.abs(px - end_x)                                  # distance to that goal line
        fix_team = np.full((T, S), -1, np.int8)                     # team given to the keeper by this step
        marked = np.zeros((T, S), bool)
        in_zone = (depth < P.keeper_zone_depth_m) & (np.abs(py - W / 2) < P.keeper_zone_half_width_m)
        d = np.where(np.isfinite(depth), depth, np.inf)
        deepest = d.argmin(axis=1)
        defend = np.where(end_x == 0.0, (attack[:, 1] > 0).astype(int), (attack[:, 1] < 0).astype(int))
        # defend[t] = team that defends this goal (0 or 1)
        slot_team = team_mat[rows, deepest]
        # nobody of the defending team in the zone, and the deepest object there is "attacking"
        def_in_zone = (in_zone & (team_mat == defend[:, None])).any(axis=1)
        wrong = (in_zone[rows, deepest] & (slot_team >= 0) & (slot_team != defend) & ~def_in_zone
                 & ~has_gt[rows, defend])
        for a, b in run_lengths(wrong):
            if b - a < P.frames(P.keeper_min_s):
                continue
            seg = np.arange(a, b)
            if np.nanmean(sp[seg, deepest[seg]]) > P.keeper_max_speed:
                continue
            team_mat[seg, deepest[seg]] = defend[seg]
            fix_team[seg, deepest[seg]] = defend[seg]
            marked[seg, deepest[seg]] = True
            reassigned += b - a

        if P.lone_gk:                                               # lone player in the penalty area = keeper
            lm = lone_mark[end_x] & np.isfinite(px) & ~has_gt[rows, defend][:, None]
            lm &= team_mat >= 0
            reassigned += int((lm & (team_mat != defend[:, None])).sum())
            dm = np.broadcast_to(defend[:, None], (T, S))
            team_mat[lm] = dm[lm]
            fix_team[lm] = dm[lm]
            marked |= lm

        # the keeper does not stop being the keeper when he leaves the 6 m zone: give the same team to
        # the whole continuous track of that slot (not jumping, not further than keeper_follow_depth_m
        # from this goal line, gaps up to keeper_follow_gap_s bridged)
        if P.keeper_follow_depth_m > 0 and marked.any():
            gap = int(round(P.keeper_follow_gap_s * P.fps))
            step = np.zeros((T, S))
            step[1:] = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)) * P.fps   # NaN where missing
            link_ok = ~(np.nan_to_num(step, nan=0.0) > P.keeper_follow_max_mps)     # False = identity jump
            near = depth < P.keeper_follow_depth_m
            for s in np.flatnonzero(marked.any(axis=0)):
                ok = np.isfinite(px[:, s]) & near[:, s] & link_ok[:, s]
                for a, b in run_lengths(_bridge_gaps(ok, gap)):
                    m = marked[a:b, s]
                    if not m.any():
                        continue
                    idx = np.arange(a, b)[np.isfinite(px[a:b, s])]
                    k = fix_team[a:b, s][m][0]
                    followed += int((team_mat[idx, s] != k).sum())
                    team_mat[idx, s] = k
    rep["keeper_frames_reassigned"] = int(reassigned)
    rep["keeper_frames_followed"] = int(followed)

    # 4. at most max_per_team per team per frame: drop the most suspicious (confirmed keepers last)
    #    drop_rule "still"  : longest still run first (before)
    #              "motion" : least movement within +-drop_window_s first
    #              "overlap": closest to a team mate first (duplicates)
    #              "both"   : overlap first when a team mate is closer than drop_overlap_m, else motion
    rule = P.drop_rule
    if rule == "still":
        key = np.where(np.isfinite(px), still_s, -1.0)            # larger = dropped first
    else:
        w = max(1, int(round(P.drop_window_s * P.fps))) * 2 + 1
        rng = np.zeros((T, S))
        for arr in (px, py):
            hi = maximum_filter1d(np.where(np.isfinite(arr), arr, -np.inf), w, axis=0, mode="nearest")
            lo = minimum_filter1d(np.where(np.isfinite(arr), arr, np.inf), w, axis=0, mode="nearest")
            r = np.where(np.isfinite(arr), hi - lo, 0.0)
            rng += r ** 2
        key = -np.sqrt(rng)                                       # small movement -> large key
    key = np.where(protect_gt | lone_any, -np.inf, key)
    dropped = 0
    excess_frames = 0
    for k in (0, 1):
        cnt = (team_mat == k).sum(axis=1)
        for t in np.flatnonzero(cnt > P.max_per_team):
            excess_frames += 1
            slots = np.flatnonzero(team_mat[t] == k)
            kk = key[t, slots].copy()
            if rule in ("overlap", "both") and len(slots) > 1:
                dx = px[t, slots][:, None] - px[t, slots][None, :]
                dy = py[t, slots][:, None] - py[t, slots][None, :]
                dd = np.hypot(dx, dy)
                np.fill_diagonal(dd, np.inf)
                nn = dd.min(axis=1)
                if rule == "overlap":
                    kk = np.where(protect_gt[t, slots], -np.inf, -nn)
                else:
                    kk = np.where(nn < P.drop_overlap_m, 1e6 - nn, kk)
                    kk = np.where(protect_gt[t, slots], -np.inf, kk)
            order = slots[np.argsort(-kk, kind="stable")]
            drop = order[: cnt[t] - P.max_per_team]
            px[t, drop] = py[t, drop] = np.nan
            team_mat[t, drop] = -1
            dropped += len(drop)
    rep["frames_over_max"] = int(excess_frames)
    rep["removed_over_max"] = int(dropped)
    rep["tracked_after"] = int(np.isfinite(px).sum())
    return px, py, team_mat, rep
