"""Version 3 end to end (the submission).

    python -m possession.v3.pipeline_v3 --repo ../more-than-a-hack-2027 --out results/final/submission.csv \
        --report results/final/report.txt \
        --team-train results/team_feat/EVAL-G1.npz results/team_feat/EVAL-G2.npz

--team-train: the H/A team model learns from OTHER games' answer keys only (team_external.py); make the
files first with --external external/EVAL-G1 --team-features-out results/team_feat/EVAL-G1.npz (and G2).
Without --team-train the team model learns from this game's pseudo labels (as in version 2).

[A] players -> [B] ball -> [C] tags -> [D] P(dead), P(home) -> [D2] play stopped (hybrid detector,
params/dead_hybrid.txt) -> [E] H/A by the time model inside every play stretch -> CSV.
The defaults are the decided version: no ball motion and no set-piece tags in the D model
(config_v3.Params3), --dead-method hybrid. --dead-method v2 decides D with the v2 time model instead.
Same other options as version 2 (--external, --dump, --check, --params, --save-state, --dead-debug).
--check is a PIPELINE CHECK ONLY: never compare versions or settings with it.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from .. import data, features, hmm, models, postprocess, preprocess, pseudo_labels
from ..v2 import ball_quality, clean_players, features_v2, hmm_v2, model_v2, pseudo_v2, tag_policy
from ..v2.ball_quality import TRUSTED
from . import hmm_v3, team_external
from .config_v3 import Params3

REPORT = []


def _external_gt(folder):
    """Truth labels of an external game: game.json "gt" (make_eval_data.py) or gt.csv (convert_metrica.py)."""
    try:
        with open(os.path.join(folder, "game.json")) as fh:
            return os.path.join(folder, json.load(fh).get("gt", "gt.csv"))
    except OSError:
        return os.path.join(folder, "gt.csv")


def log(msg):
    line = time.strftime("%H:%M:%S") + " " + str(msg)
    print(line, flush=True)
    REPORT.append(line)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _paths(args, P):
    args.repo = os.path.abspath(args.repo)
    if not os.path.isfile(os.path.join(args.repo, "games.py")):
        raise SystemExit(f"--repo does not point at more-than-a-hack-2027 (games.py not found): {args.repo}")
    sys.path.insert(0, args.repo)
    import games  # noqa: E402
    if args.external:
        with open(os.path.join(args.external, "game.json")) as fh:
            cfg = json.load(fh)
        stem = os.path.join(args.external, cfg.get("input_dir", ""), cfg["stem"])
        paths = {"atd_meta": stem + "_atd.xml", "atd_raw": stem + "_atd.txt",
                 "homography": stem + "_homography.csv", "smart_tagging": stem + "_smart_tagging.xml"}
        P.pitch_length, P.pitch_width = cfg.get("pitch", (P.pitch_length, P.pitch_width))
        return cfg["stem"], paths, int(cfg["n_frames"]), [tuple(b) for b in cfg["blocks"]]
    data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(args.repo, "data")
    paths = games.asset_paths(args.game, data_dir)
    n = args.n_frames or games.source_frames("tactical", args.game)
    blocks = json.loads(args.blocks) if args.blocks else data.graded_blocks(args.repo, args.game)
    if args.game in games.GAMES:
        P.pitch_length, P.pitch_width = games.GAMES[args.game]["pitch"]
    return args.game, paths, n, blocks


def kickoff_stretches(prep, team_mat, bq, tags, side_of, H, valid, blocks, P):
    """Kick-off preprocessing. Returns [(start, kick, team, kind)] and a short report.

    start = the goal moment (goal -> kick-off) or the half start (first kick-off of a half);
    kick = the frame play restarts; team = 0 (H) / 1 (A) kicking, -1 unknown.
    """
    try:                                              # 髙橋's module: v2 if pushed there, else the v3 copy
        from ..v2 import kickoff
    except ImportError:
        from . import kickoff
    events, _ = kickoff.find_kickoffs(prep, team_mat, bq, tags, P, side_of)
    def half_of(t):                                   # frame -> half number (Laws: each half starts with a kick-off)
        for h, a, b in blocks:
            if a <= t <= b:
                return h
        return 0
    sets = kickoff.find_goal_kickoff_sets(events, prep, team_mat, bq, P, tags=tags, side_of=side_of,
                                          half_of=half_of)        # players only, no camera
    n_rej = kickoff.confirm_kickoffs_with_goals(events, sets)
    goal_of = {r["kickoff_frame"]: r["goal_frame"] for r in sets if r["kickoff_frame"] is not None}
    out = []
    for _, a, b in blocks:
        ks = sorted((e for e in events if e["found"] and a <= e["kickoff_frame"] <= b),
                    key=lambda e: e["kickoff_frame"])
        for i, e in enumerate(ks):
            k = e["kickoff_frame"]
            g = goal_of.get(k)
            if g is not None and a <= g < k:
                out.append((int(g), int(k), int(e["team"]), "goal"))
            elif i == 0:
                out.append((int(a), int(k), int(e["team"]), "half start"))
    rep = {"kick-offs found": sum(e["found"] for e in events), "of tags/goals": len(events),
           "turned down (no goal before)": n_rej,
           "D stretches": [(kind, round((k - s) / P.fps, 1)) for s, k, _, kind in out]}
    return out, rep


def run(args):
    P = Params3()
    if args.params:
        with open(args.params) as fh:
            for k, v in json.load(fh).items():
                setattr(P, k, v)
    name, paths, n, blocks = _paths(args, P)
    os.makedirs(args.cache_dir, exist_ok=True)

    log("loading tracking")
    tr = data.load_tracking(paths["atd_meta"], paths["atd_raw"], n, (P.pitch_length, P.pitch_width),
                            cache=os.path.join(args.cache_dir, f"{name}_tracking.npz"))
    if os.path.isfile(paths["homography"]):
        H, _, valid = data.load_homography(paths["homography"], n)
    else:
        H, valid = np.full((n, 3, 3), np.nan), np.zeros(n, bool)
    tags = tag_policy.load_tags(paths["smart_tagging"], P.fps) if os.path.isfile(paths["smart_tagging"]) else []
    side_of = lambda nm: data.tag_team_side(nm, tr.team_names)  # noqa: E731

    # [A] players
    attack = preprocess.attack_sign(tr, blocks, P.fps)
    px, py, team_mat, rep_a = clean_players.clean_players(tr, attack, P, event_mask=tag_policy.event_frames(tags, n, P))  # same call as v2
    log(f"[A] clean_players: {rep_a}")

    # [B] ball
    bq, rep_b = ball_quality.assess_ball(tr.bx, tr.by, px, py, P)
    log(f"[B] ball_quality: {rep_b}")

    # shared preprocessing on the cleaned data (version 1 code, cleaned inputs)
    tr2 = data.Tracking(px.astype(np.float32), py.astype(np.float32), tr.pspeed, tr.team,
                        bq["bx"].astype(np.float32), bq["by"].astype(np.float32), tr.team_names)
    prep = preprocess.prepare(tr2, H, valid, blocks, P)
    prep["ball_seen"] = bq["status"] == TRUSTED

    # [C] tags
    tag_logit, tag_dead, rep_c = tag_policy.tag_evidence(tags, side_of, n, P)
    log(f"[C] tag_policy: {rep_c}")

    # [C2] kick-offs (preprocessing, 髙橋's kickoff.py): the exact kick and the goal before it, from the players.
    #      Used by the hybrid D detector instead of its own tag-based kick-off search.
    ko_events = None
    if getattr(P, "kickoff_pre", True) and args.dead_method == "hybrid":
        ko_events, rep_k = kickoff_stretches(prep, team_mat, bq, tags, side_of, H, valid, blocks, P)
        log(f"[C2] kick-offs: {rep_k}")

    # [D] features, pseudo labels, models
    log("features")
    fd = features.dead_features(prep, tags, P)
    fd = pd.concat([fd, features_v2.dead_extra(bq, tag_dead, P)], axis=1)
    Xh, Xa = features_v2.team_features_v2(prep, team_mat, bq, P)
    # MT-26 clues (11/5, priority high): off unless switched on in --params
    from ..v2 import clue_features  # MT-26 clues live in v2 (shared)
    if any(getattr(P, k, False) for k in ("clue_score", "clue_backline", "clue_crowd")):
        per_c = clue_features.team_clues(prep, team_mat, bq, P, tags, side_of)
        Xh, Xa = clue_features.add_team_clues(Xh, Xa, per_c)
        log(f"[MT-26] team clues added: {sorted(per_c[0])}")
    if getattr(P, "clue_align", False):
        fa = clue_features.align_features(prep, team_mat, P)
        fd = pd.concat([fd, fa], axis=1)
        log(f"[MT-26] D clues added: {list(fa.columns)}")
    if P.team_lasttouch:
        Xh, Xa = team_external.add_lasttouch(Xh, Xa, prep, team_mat, bq, P)
        log("[D] last-touch team features added")
    if args.team_features_out:
        if not (args.external and os.path.isfile(_external_gt(args.external))):
            raise SystemExit("--team-features-out needs --external with an answer key (other games only)")
        gt_ext = pd.read_csv(_external_gt(args.external))["label"].astype(str).to_numpy()[:n]
        team_external.save_team_features(args.team_features_out, name, Xh, Xa, bq["conf"], gt_ext)
        log(f"wrote {args.team_features_out} (team features + true H/A of {name})")

    control = pseudo_v2.control_events_v2(prep, team_mat, bq, P)
    y_dead = pseudo_labels.dead_labels(fd, P, control)
    y_team = pseudo_v2.team_labels_v2(control, y_dead, P)
    log(f"[D] pseudo labels: dead={int((y_dead == 1).sum())} alive={int((y_dead == 0).sum())} "
        f"home={int((y_team == 1).sum())} away={int((y_team == 0).sum())}")
    motion_cols = [c for c in fd.columns if c.startswith(("ball_speed", "ball_flight"))]
    motion = getattr(P, "dead_ball_motion", True)
    fd_model = fd if motion else fd.drop(columns=motion_cols)
    if not motion:
        log(f"[D] D model without ball motion: dropped {motion_cols}")
    if not getattr(P, "dead_set_piece_tags", True):
        tag_cols = [c for c in fd_model.columns if c.startswith(("to_set_piece_s", "in_set_piece_tag",
                                                                   "tag_to_restart_s", "tag_since_restart_s"))]
        fd_model = fd_model.drop(columns=tag_cols)
        log(f"[D] D model without set-piece tags: dropped {tag_cols}")
    if args.team_features_out:
        # this external game's D features and its true D, for --team-train with dead_external
        dpath = team_external.dead_path(args.team_features_out)
        tb = os.path.join(args.external, "truth", "ball.csv")
        tseen = pd.read_csv(tb)["x_m"].notna().to_numpy() if os.path.isfile(tb) else None
        team_external.save_dead_features(dpath, name, fd_model, gt_ext, tseen)
        log(f"wrote {dpath} (D features + true D of {name})")
    if getattr(P, "dead_external", False) and args.team_train:
        # P(dead) learned from OTHER games' true D only (no pseudo labels of this game)
        dm = team_external.fit_dead(fd_model, P, args.team_train, name, log=log)
    else:
        dm = models.DeadModel(P).fit(fd_model, y_dead)
    p_dead = dm.predict(fd)
    log("dead model top coefficients: " + ", ".join(f"{a}={c:+.2f}" for a, c in dm.coefficients()[:6]))
    if args.team_train:
        p_model = team_external.fit_predict(Xh, Xa, P, args.team_train, name, log=log)
        if getattr(P, "team_self_train_conf", 0) > 0:
            p_model = team_external.self_train(Xh, Xa, P, args.team_train, name, p_model, log=log)
        w_ps = getattr(P, "team_blend_pseudo", 0.0)
        if w_ps > 0:
            p_ps, _ = model_v2.fit_predict(Xh, Xa, y_team, bq["conf"], P, log=log)
            p_model = (1 - w_ps) * p_model + w_ps * p_ps
            log(f"[D] team P(home) blended with this game's pseudo-label model (weight {w_ps})")
        win =int(round(getattr(P, "team_smooth_s", 0.0) * P.fps)) | 1
        if win > 1:
            unseen_win = int(round(P.team_smooth_unseen_s * P.fps)) | 1
            raw = pd.Series(p_model)
            p_model = raw.rolling(win, center=True, min_periods=1).mean().to_numpy()
            if unseen_win != win:
                p_unseen = raw.rolling(unseen_win, center=True, min_periods=1).mean().to_numpy()
                p_model = np.where(bq["status"] == TRUSTED, p_model, p_unseen)
            log(f"[D] team model P(home) averaged over {win} frames with trusted ball, "
                f"{unseen_win} otherwise")
    else:
        p_model, _ = model_v2.fit_predict(Xh, Xa, y_team, bq["conf"], P, log=log)
    p_home = 1 / (1 + np.exp(-(_logit(p_model) + tag_logit)))
    ca, cb = getattr(P, "home_calib_a", 1.0), getattr(P, "home_calib_b", 0.0)   # off unless set in --params
    if ca != 1.0 or cb != 0.0:
        p_home = 1 / (1 + np.exp(-(ca * _logit(p_home) + cb)))
        log(f"[E] P(home) calibrated: a={ca} b={cb}, switch cost {P.cost_switch_team}")

    # [E] state model
    log("state model")
    sig = hmm_v2.switch_signal(Xh, P)
    if getattr(P, "clue_crowd_switch", False):
        sig = clue_features.crowd_switch_signal(prep, team_mat, bq, P)
        log(f"[MT-26] switch signal = crowding onset (mean {sig.mean():.3f}, share > 0.5: {(sig > 0.5).mean():.3f})")
    if getattr(P, "switch_model", False) and args.team_train:
        p_sw = team_external.fit_predict_switch(Xh, P, args.team_train, name, log=log)
        sig = np.clip(P.switch_model_scale * p_sw, 0.0, 1.0)
    switch_cost = P.cost_switch_team * (1.0 - P.switch_signal_drop * sig)
    hybrid_info = None
    if args.dead_method == "hybrid":
        # [D2] play stopped? -- hybrid detector (core from P(dead), start / end from the situation signs,
        #      3-second rule), then H / A by the time model inside every play stretch (D stays fixed)
        from . import dead_situation, situation_features as sf
        cfg = dead_situation.read_config(args.dead_config)
        bx_t = np.where(bq["status"] == TRUSTED, bq["bx"], np.nan)
        by_t = np.where(bq["status"] == TRUSTED, bq["by"], np.nan)
        # Without a trusted ball position, require stronger evidence to change teams.
        switch_cost *= np.where(np.isfinite(bx_t), 1.0, P.unseen_ball_switch_cost_scale)
        switch_cost *= hmm_v3.flight_switch_multiplier(
            bq["flight"], np.isfinite(bx_t), P.ball_flight_switch_cost_scale, P.ball_flight_switch_tail_s, P.fps)
        cols = sf.situation_features(px, py, team_mat, attack, tags, (P.pitch_length, P.pitch_width), P.fps,
                                     bx_t, by_t, H, valid)
        cols["p_dead"] = np.asarray(p_dead, float)
        if getattr(P, "clue_align", False):        # MT-26: usable in params/dead_hybrid*.txt as align_frac / align_drop
            cols["align_frac"] = np.nan_to_num(fd["align_frac"].to_numpy(float), nan=1.0)
            cols["align_drop"] = np.nan_to_num(fd["align_drop"].to_numpy(float), nan=0.0)
        cols["v2_D"] = np.zeros(n)
        dead_mask, stretches, dropped, dinfo = dead_situation.detect_hybrid(cols, blocks, cfg, tags, ko_events)
        if P.recover_short_ball_out:
            n_out = dead_situation.recover_short_ball_out(
                dead_mask, bx_t, by_t, blocks, tags, (P.pitch_length, P.pitch_width), P.fps)
            log(f"[D2] {n_out} short ball-out stops recovered")
        if P.stop_control_search_s > 0:
            contact = hmm_v3._ball_contact_team(px, py, team_mat, bx_t, by_t, 1.4, 0.4, round(0.12 * P.fps))
            moving = (cols["ball_moving"] > 0) & (cols["ball_out"] == 0)
            n_delayed = dead_situation.delay_stops_during_control(
                dead_mask, dinfo["forced"], blocks, p_home, contact, moving, P.stop_control_search_s, P.fps)
            log(f"[D2] {n_delayed} stoppage starts delayed by continuing control")
        sits = cfg["situations"]
        p_ha = p_home
        if cfg["settings"].get("soccer_rules", 0) > 0:
            from . import soccer_rules
            st_like = {"n": n, "px": px, "py": py, "team_mat": team_mat, "attack": attack, "pitch":
                       (P.pitch_length, P.pitch_width), "blocks": blocks, "bx": bx_t, "by": by_t, "p_home": p_home}
            p_ha, _, rep_r = soccer_rules.apply(st_like, dead_mask, stretches, sits, cols)
            log(f"[E] football rules for H/A: {rep_r}")
        p_decode = hmm_v3.support_contact_team(
            p_ha, px, py, team_mat, bx_t, by_t, P.sustained_contact_log_odds, P.fps)
        labels = hmm_v3.decode_with_dead(dead_mask, p_decode, switch_cost, blocks, n, P.prob_floor)
        if getattr(P, "switch_evidence_max_s", 0) > 0:
            labels, n_ev = hmm_v3.drop_unsupported_spells(labels, px, py, team_mat, bx_t, by_t,
                                                          P.switch_evidence_max_s, P.switch_evidence_m, P.fps)
            log(f"[E] {n_ev} possessions of <= {P.switch_evidence_max_s} s without a touch of that team: "
                f"given to the team before")
        labels, n_advanced = hmm_v3.advance_confirmed_turnovers(
            labels, px, py, team_mat, bx_t, by_t, P.turnover_backtrack_s, P.turnover_control_s, P.fps)
        log(f"[E] {n_advanced} H/A switches moved back to sustained control")
        if P.restore_brief_turnovers:
            labels, n_restored = hmm_v3.restore_brief_turnovers(
                labels, p_ha, px, py, team_mat, bx_t, by_t, P.fps)
            log(f"[E] {n_restored} brief possessions restored from control by both teams")
        labels, n_slowdown = hmm_v3.advance_turnovers_at_ball_slowdown(
            labels, px, py, team_mat, bx_t, by_t, P.turnover_slowdown_backtrack_s, P.fps)
        log(f"[E] {n_slowdown} H/A switches moved back to ball slowdowns")
        labels, n_redirect = hmm_v3.delay_turnovers_at_ball_redirect(
            labels, px, py, team_mat, bx_t, by_t, P.turnover_redirect_search_s, P.fps)
        log(f"[E] {n_redirect} H/A switches delayed to ball redirections")
        labels, n_merged = hmm_v3.merge_weak_turnovers(
            labels, p_ha, px, py, team_mat, bx_t, by_t, P.weak_turnover_merge_s, P.fps)
        log(f"[E] {n_merged} unsupported brief reversals merged")
        by_sit = {}
        for x in stretches:
            nm = sits[x["situation"]]["en"]
            by_sit[nm] = by_sit.get(nm, 0) + 1
        log(f"[D2] hybrid D ({os.path.basename(args.dead_config)}): {len(stretches)} stretches {by_sit}, "
            f"{len(dropped)} candidates not used, {sum(1 for x in stretches if x.get('special'))} from kick-off / goal tags")
        hybrid_info = {"stretches": stretches, "dropped": dropped, "mask": dead_mask}
    else:
        logE = hmm.emissions(p_dead, p_home, P)
        states = [hmm_v2.viterbi_tv(logE[a:b + 1], P, switch_cost[a:b + 1]) for _, a, b in blocks]
        labels = postprocess.to_labels(states, n, blocks)
    if args.drop_short_d > 0 and args.dead_method != "hybrid":
        # D stretches shorter than --drop-short-d seconds are treated as play; H/A is decided again
        keep = labels == "D"
        dropped = 0
        for _, a, b in blocks:
            m = np.r_[False, keep[a:b + 1], False].astype(np.int8)
            d = np.diff(m)
            for s0, e0 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                if e0 - s0 < P.frames(args.drop_short_d):
                    keep[a + s0:a + e0] = False
                    dropped += 1
        logE = hmm.emissions(np.where(keep, 0.999, 0.001), p_home, P)
        states = [hmm_v2.viterbi_tv(logE[a:b + 1], P, switch_cost[a:b + 1]) for _, a, b in blocks]
        labels = postprocess.to_labels(states, n, blocks)
        log(f"[E] dropped {dropped} D stretches shorter than {args.drop_short_d} s")
    if P.restart_goal_len_s > 0 and hybrid_info is not None:
        labels, n_rg = hmm_v3.scorer_after_restart(labels, tags, side_of, blocks, hybrid_info["stretches"],
                                                   P.restart_goal_len_s, P.restart_goal_before_s,
                                                   P.restart_goal_after_s, P.fps)
        log(f"[E] {n_rg} restarts that scored at once: {P.restart_goal_len_s} s after the kick = the scorer's possession")
    if P.goal_lead_s > 0:
        labels, n_goal = hmm_v3.scorer_before_goals(labels, tags, side_of, blocks, P.goal_lead_s, P.fps,
                                                    P.goal_lead_shift_s)
        log(f"[E] {P.goal_lead_s} s before {n_goal} goals = the scoring team's possession")
    if hybrid_info is not None:
        contact = hmm_v3._ball_contact_team(px, py, team_mat, bx_t, by_t, 1.4, 0.4, round(0.12 * P.fps))
        labels, n_restarted = hmm_v3.advance_controlled_restarts(
            labels, p_ha, contact, hybrid_info["stretches"], P.fps)
        log(f"[D2] {n_restarted} stoppage ends advanced by control from the next team")
        if P.restore_brief_turnovers:
            labels, n_recovered = hmm_v3.restore_brief_turnovers(
                labels, p_ha, px, py, team_mat, bx_t, by_t, P.fps, tight_contact=True)
            log(f"[E] {n_recovered} brief turnovers restored from close contact")
            labels, n_repeated = hmm_v3.restore_repeated_contact_turnovers(
                labels, p_ha, px, py, team_mat, bx_t, by_t, P.fps)
            log(f"[E] {n_repeated} brief turnovers restored from repeated contact")
        labels, n_delayed_switches = hmm_v3.delay_switches_during_contact(
            labels, p_ha, px, py, team_mat, bx_t, by_t, P.fps)
        log(f"[E] {n_delayed_switches} switches delayed by continuing contact")
        labels, n_merged_before_stops = hmm_v3.merge_unsupported_turnovers_before_stops(
            labels, p_ha, px, py, team_mat, bx_t, by_t, P.fps)
        log(f"[E] {n_merged_before_stops} unsupported turnovers before stoppages merged")
        labels, n_contested = hmm_v3.merge_contested_reversals(
            labels, p_ha, px, py, team_mat, bx_t, by_t, P.fps)
        log(f"[E] {n_contested} brief reversals merged by continuing contact")
        labels, n_restored_before_stops = hmm_v3.restore_turnovers_before_stops(
            labels, p_ha, px, py, team_mat, bx_t, by_t, hybrid_info["stretches"], P.fps)
        log(f"[E] {n_restored_before_stops} brief turnovers recovered before stoppages")
        close_contact = hmm_v3._ball_contact_team(px, py, team_mat, bx_t, by_t, 0.8, 0.4, round(0.12 * P.fps))
        labels, n_weak_at_stops = hmm_v3.merge_weak_possession_at_stops(labels, p_ha, close_contact, P.fps)
        log(f"[E] {n_weak_at_stops} weak possessions adjacent to stoppages merged")
        labels, n_paired = hmm_v3.anchor_switches_to_paired_contacts(
            labels, p_ha, px, py, team_mat, bx_t, by_t, P.fps)
        log(f"[E] {n_paired} switches aligned to consecutive contact from both teams")
        labels, n_unseen_start = hmm_v3.restore_turnovers_with_unseen_start(
            labels, p_ha, px, py, team_mat, bx_t, by_t, P.fps)
        log(f"[E] {n_unseen_start} brief turnovers recovered with an unseen start")
        n_trimmed = 0
        if getattr(P, "trim_low_prob_stop_starts", True):
            labels, n_trimmed = dead_situation.trim_low_probability_stop_starts(
                labels, p_dead, hybrid_info["stretches"], P.fps)
        log(f"[D2] {n_trimmed} low-confidence stoppage starts delayed")
        labels, n_extended_ends = hmm_v3.extend_brief_turnover_ends(
            labels, p_ha, px, py, team_mat, bx_t, by_t, P.fps)
        log(f"[E] {n_extended_ends} brief turnovers extended before the regain")
        labels, n_departure = hmm_v3.advance_switches_after_ball_departure(
            labels, p_ha, px, py, team_mat, bx_t, by_t, P.fps)
        log(f"[E] {n_departure} switches advanced after ball departure")
        labels, n_arrival = hmm_v3.delay_switches_until_ball_arrival(
            labels, p_ha, px, py, team_mat, bx_t, by_t, P.fps)
        log(f"[E] {n_arrival} switches delayed until ball arrival")
        labels, n_longer_ends = hmm_v3.extend_brief_turnover_ends(
            labels, p_ha, px, py, team_mat, bx_t, by_t, P.fps, longer=True)
        log(f"[E] {n_longer_ends} turnovers of up to six seconds extended before the regain")
        labels, n_earlier_starts = hmm_v3.extend_brief_turnover_ends(
            labels, p_ha, px, py, team_mat, bx_t, by_t, P.fps, earlier=True)
        log(f"[E] {n_earlier_starts} brief turnovers extended at the start")
        if P.restore_brief_turnovers:
            labels, n_extended = hmm_v3.restore_repeated_contact_turnovers(
                labels, p_ha, px, py, team_mat, bx_t, by_t, P.fps, extended=True)
            log(f"[E] {n_extended} longer turnovers restored from repeated contact")
        fdr = getattr(P, "final_drop_short_s", 0.0)
        if fdr > 0:
            # last pass: a short possession between two of the other team's, in which that team never
            # comes within final_drop_touch_m of the (trusted) ball nearer than every opponent, is dropped
            labels, n_fdr = hmm_v3.drop_unsupported_spells(labels, px, py, team_mat, bx_t, by_t,
                                                           fdr, P.final_drop_touch_m, P.fps)
            log(f"[E] {n_fdr} short possessions without a touch dropped (last pass)")
        fd_s = getattr(P, "force_d_min_s", 0.0)
        if fd_s > 0:
            # play in which P(dead) stays high this long is a stoppage the situation rules missed
            n_fd = 0
            for s, e in preprocess.run_lengths((p_dead >= P.force_d_pdead) & np.isin(labels, ["H", "A"])):
                if e - s >= fd_s * P.fps:
                    labels[s:e] = "D"; n_fd += 1
            log(f"[D2] {n_fd} play stretches with P(dead) >= {P.force_d_pdead} for {fd_s} s made D")
        ud_s = getattr(P, "unseen_d_min_s", 0.0)
        if ud_s > 0:
            # play with no ball detected this long while P(dead) leans to stopped is a stoppage too
            seen_u = bq["status"] == TRUSTED
            n_un = 0
            for s, e in preprocess.run_lengths(~seen_u & np.isin(labels, ["H", "A"])):
                if e - s >= ud_s * P.fps and p_dead[s:e].mean() >= P.unseen_d_pdead:
                    labels[s:e] = "D"; n_un += 1
            log(f"[D2] {n_un} unseen-ball play stretches with P(dead) >= {P.unseen_d_pdead} made D")
        us = getattr(P, "undo_d_seen_min", 0.0)
        if us > 0:
            # a D stretch in which the ball is still detected and P(dead) stays low is play missed:
            # it goes back to the team that had the ball before it
            seen_t = bq["status"] == TRUSTED
            n_ud = 0
            for s, e in preprocess.run_lengths(labels == "D"):
                if (e - s > P.undo_d_max_s * P.fps or seen_t[s:e].mean() < us
                        or p_dead[s:e].mean() > P.undo_d_pdead_max):
                    continue
                if s > 0 and labels[s - 1] in ("H", "A"):
                    labels[s:e] = labels[s - 1]
                elif e < n and labels[e] in ("H", "A"):
                    labels[s:e] = labels[e]
                else:
                    continue
                n_ud += 1
            log(f"[D2] {n_ud} stoppages with the ball still seen and low P(dead) put back to play")
        et = getattr(P, "d_edge_trim_max_s", 0.0)
        if et > 0:
            # the first / last moments of a D in which the ball is still seen and P(dead) is low are
            # play: the D starts later / ends earlier (at most d_edge_trim_max_s each side)
            w = int(P.d_edge_win_s * P.fps) | 1
            seen_w = pd.Series((bq["status"] == TRUSTED).astype(float)).rolling(w, center=True, min_periods=1).mean().to_numpy()
            pd_w = pd.Series(p_dead).rolling(w, center=True, min_periods=1).mean().to_numpy()
            playish = (seen_w >= 0.5) & (pd_w <= P.d_edge_pdead_max)
            cap = int(et * P.fps)
            n_et = 0
            for s, e in preprocess.run_lengths(labels == "D"):
                k = s
                while k < e and playish[k] and k - s < cap:
                    k += 1
                k2 = e
                while k2 > k and playish[k2 - 1] and e - k2 < cap:
                    k2 -= 1
                if k2 <= k:
                    continue                      # the whole D looks like play: left to undo_d
                if k > s and s > 0 and labels[s - 1] in ("H", "A"):
                    labels[s:k] = labels[s - 1]; n_et += 1
                if k2 < e and e < n and labels[e] in ("H", "A"):
                    labels[k2:e] = labels[e]; n_et += 1
            log(f"[D2] {n_et} stoppage edges trimmed where play signs remain")
        sv = getattr(P, "shot_fix_speed", 0.0)
        if sv > 0:
            # a fast ball flying at a goal from near it is a shot of the team attacking that goal
            # (a team does not shoot at its own goal), so that team has the ball during the shot
            Lp, Wp = P.pitch_length, P.pitch_width
            hb = max(1, int(0.12 * P.fps))
            with np.errstate(invalid="ignore"):
                vx = (np.r_[bx_t[hb:], np.full(hb, np.nan)] - np.r_[np.full(hb, np.nan), bx_t[:-hb]]) / (2 * hb / P.fps)
                vy = (np.r_[by_t[hb:], np.full(hb, np.nan)] - np.r_[np.full(hb, np.nan), by_t[:-hb]]) / (2 * hb / P.fps)
            spd = np.hypot(vx, vy)
            att_hi = np.where(attack[:, 0] > 0, 0, 1)             # team attacking the x = L goal
            shot_team = np.full(n, -1, np.int8)
            cos_max = np.cos(np.radians(P.shot_fix_angle_deg))
            for gx, team in ((Lp, att_hi), (0.0, 1 - att_hi)):
                tx, ty = gx - bx_t, Wp / 2 - by_t
                dist = np.hypot(tx, ty)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cosang = (vx * tx + vy * ty) / (spd * dist)
                ok = (spd >= sv) & (dist <= P.shot_fix_dist_m) & (cosang >= cos_max)
                shot_team[np.nan_to_num(ok, nan=0).astype(bool)] = team[np.nan_to_num(ok, nan=0).astype(bool)]
            pre = int(P.shot_fix_pre_s * P.fps)
            n_sf = 0
            whole = getattr(P, "shot_fix_whole_spell", False)
            for k in (0, 1):
                wrong_lab, right_lab = ("A", "H") if k == 0 else ("H", "A")
                for s, e in preprocess.run_lengths(shot_team == k):
                    if e - s < 2:
                        continue
                    if whole:
                        # the predicted team would be shooting at its own goal: the whole possession
                        # (the run of that label holding the shot) belongs to the other team
                        mid = s + (e - s) // 2
                        if labels[mid] != wrong_lab:
                            continue
                        a, b = mid, mid
                        while a > 0 and labels[a - 1] == wrong_lab:
                            a -= 1
                        while b < n and labels[b] == wrong_lab:
                            b += 1
                        labels[a:b] = right_lab
                        n_sf += 1
                        continue
                    s2 = max(0, s - pre)
                    seg = labels[s2:e]
                    wrong = seg == wrong_lab
                    if wrong.any():
                        n_sf += 1
                        seg[wrong] = right_lab
            log(f"[E] {n_sf} shots at a goal given to the attacking team")
        fs = getattr(P, "flip_short_s", 0.0)
        if fs > 0:
            # a short H/A spell between two spells of the other team that the team model itself
            # does not back (mean prob of that team <= flip_short_conf) is a blip: give it to the other team
            n_fl = 0
            bounds = np.flatnonzero(labels[1:] != labels[:-1]) + 1
            starts = np.r_[0, bounds]; ends = np.r_[bounds, n]
            for i in range(1, len(starts) - 1):
                a, b = starts[i], ends[i]; lab = labels[a]
                if lab not in ("H", "A") or b - a >= fs * P.fps:
                    continue
                other = "A" if lab == "H" else "H"
                if labels[a - 1] != other or labels[b] != other:
                    continue
                conf = p_home[a:b].mean() if lab == "H" else 1 - p_home[a:b].mean()
                if conf <= P.flip_short_conf:
                    labels[a:b] = other; n_fl += 1
            log(f"[F] {n_fl} short unbacked spells given to the surrounding team")
        rm = getattr(P, "run_reopen_n", 0.0)
        if rm > 0:
            # inside a stoppage, a stretch where several players keep running (and the D model is unsure)
            # is play: give it to the team that had the ball before the stoppage
            kk = max(1, int(0.48 * P.fps))
            with np.errstate(invalid="ignore"):
                vv = np.hypot(px[kk:] - px[:-kk], py[kk:] - py[:-kk]) / (kk / P.fps)
                nrun = np.r_[np.sum(vv > P.run_reopen_speed, 1), np.zeros(kk)].astype(float)
            w = max(1, int(P.run_reopen_win_s * P.fps))
            nr_w = pd.Series(nrun[:n]).rolling(w, center=True, min_periods=1).mean().to_numpy()
            pd_w = pd.Series(p_dead).rolling(w, center=True, min_periods=1).mean().to_numpy()
            run_sig = (nr_w >= rm) & (pd_w <= P.run_reopen_pdead)
            n_ro = 0
            for a, b in preprocess.run_lengths(labels == "D"):
                if a == 0 or labels[a - 1] not in ("H", "A"):
                    continue
                for c, d in preprocess.run_lengths(run_sig[a:b]):
                    c, d = a + c, a + d
                    if c == a or d == b or d - c < P.run_reopen_min_s * P.fps:
                        continue
                    labels[c:d] = labels[a - 1]; n_ro += 1
            log(f"[R] {n_ro} stoppage middles with players running reopened as play")
        tiny = getattr(P, "drop_tiny_s", 0.0)
        if tiny > 0:
            # a sub-second H/A spell between two spells of the other team is not a possession
            bounds = np.flatnonzero(labels[1:] != labels[:-1]) + 1
            starts = np.r_[0, bounds]; ends = np.r_[bounds, n]
            n_ti = 0
            for i in range(1, len(starts) - 1):
                a, b = starts[i], ends[i]
                if labels[a] in ("H", "A") and b - a < tiny * P.fps and labels[a - 1] == labels[b]                         and labels[b] in ("H", "A") and labels[b] != labels[a]:
                    labels[a:b] = labels[b]; n_ti += 1
            log(f"[T] {n_ti} sub-second spells between the other team's spells removed")
        sdw = getattr(P, "sandwich_d_s", 0.0)
        if sdw > 0:
            # a short H/A spell between two stoppages that the team model does not back is part of the stoppage
            bounds = np.flatnonzero(labels[1:] != labels[:-1]) + 1
            starts = np.r_[0, bounds]; ends = np.r_[bounds, n]
            n_sw = 0
            for i in range(1, len(starts) - 1):
                a, b = starts[i], ends[i]; lab = labels[a]
                if lab not in ("H", "A") or b - a >= sdw * P.fps or labels[a - 1] != "D" or labels[b] != "D":
                    continue
                conf = p_home[a:b].mean() if lab == "H" else 1 - p_home[a:b].mean()
                if conf <= P.sandwich_d_conf:
                    labels[a:b] = "D"; n_sw += 1
            log(f"[S] {n_sw} short unbacked spells between stoppages made part of the stoppage")
    data.write_submission(args.out, labels)
    log(f"wrote {args.out}")
    summ = postprocess.summary(labels, blocks, P.fps)
    log("summary: " + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in summ.items()}))

    if args.save_state:
        import pickle
        extra = Xh[[c for c in Xh.columns if c.startswith(("r_own_", "r_opp_", "k_own_", "k_opp_"))
                    and not c.endswith(("_p0.5", "_f0.5", "_d0.5", "_p1.0", "_f1.0", "_d1.0", "_p2.0", "_f2.0", "_d2.0"))]]
        state = {"name": name, "n": n, "blocks": blocks, "fps": P.fps, "fd": fd, "home_view": extra,
                 "p_dead": p_dead, "p_home": p_home, "switch_cost": switch_cost, "labels": labels,
                 "y_dead": y_dead, "params": {k: getattr(P, k) for k in vars(P)}, "ko_events": ko_events, "dead_method": args.dead_method,
                 # player-only data for tools/dead_situations.py (ball not needed)
                 "px": px.astype(np.float32), "py": py.astype(np.float32), "team_mat": team_mat,
                 "attack": attack, "tags": tags, "pitch": (P.pitch_length, P.pitch_width),
                 "H": H.astype(np.float32), "H_valid": valid,       # camera view (what is in the picture)
                 "bx": np.where(bq["status"] == TRUSTED, bq["bx"], np.nan).astype(np.float32),
                 "by": np.where(bq["status"] == TRUSTED, bq["by"], np.nan).astype(np.float32),
                 "external": os.path.abspath(args.external) if args.external else None}
        if args.external and os.path.isfile(_external_gt(args.external)):
            state["gt"] = pd.read_csv(_external_gt(args.external))["label"].astype(str).to_numpy()
            tb = os.path.join(args.external, "truth", "ball.csv")
            if os.path.isfile(tb):
                state["true_ball_seen"] = pd.read_csv(tb)["x_m"].notna().to_numpy()
        os.makedirs(os.path.dirname(os.path.abspath(args.save_state)) or ".", exist_ok=True)
        with open(args.save_state, "wb") as fh:
            pickle.dump(state, fh)
        log(f"wrote {args.save_state}")

    if args.dead_debug:
        from ..v2 import dead_debug
        gt_arr = None
        if args.external and os.path.isfile(_external_gt(args.external)):
            gt_arr = pd.read_csv(_external_gt(args.external))["label"].astype(str).to_numpy()
        os.makedirs(os.path.dirname(os.path.abspath(args.dead_debug)) or ".", exist_ok=True)
        dead_debug.explain(fd, dm, P, control, y_dead, p_dead, labels, blocks, args.dead_debug, gt=gt_arr, fps=P.fps)
        log(f"wrote {args.dead_debug}_frames.csv and {args.dead_debug}_stretches.txt")

    if args.dump:
        t = np.arange(n) / P.fps
        pd.DataFrame({"frame": np.arange(n),
                      "video_time": [f"{int(s // 60):02d}:{s % 60:05.2f}" for s in t],
                      "label": labels, "p_dead": np.round(p_dead, 3), "p_home": np.round(p_home, 3),
                      "p_model": np.round(p_model, 3), "tag_logit": np.round(tag_logit, 2),
                      "ball_status": bq["status"], "ball_conf": np.round(bq["conf"], 2),
                      "ball_flight": bq["flight"].astype(int), "switch_signal": np.round(sig, 2),
                      "ball_seen": (bq["status"] == TRUSTED).astype(int),
                      "pseudo_dead": y_dead, "pseudo_team": y_team}).to_csv(args.dump, index=False)
        log(f"wrote {args.dump}")

    sys.path.insert(0, os.path.join(args.repo, "evaluation"))
    if args.external and os.path.isfile(_external_gt(args.external)):
        import possession_metrics as pm  # noqa: E402
        gt = pm.label_codes(pm.read_label_csv(_external_gt(args.external)))
        res = pm.evaluate(gt, pm.label_codes(labels))
        log("EXTERNAL GAME score (allowed for tuning): macroF1={:.4f} spellF1={:.4f} possessions={}".format(
            res["macro_f1"], res["spell_f1_mean"], res["n_pred_spells"]))
    if args.check:
        import possession_metrics as pm  # noqa: E402
        gt_path = args.gt or os.path.join(args.repo, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
        res = pm.evaluate(pm.label_codes(pm.read_label_csv(gt_path)), pm.label_codes(labels))
        log("PIPELINE CHECK ONLY (not for tuning): macroF1={:.4f} spellF1={:.4f}".format(
            res["macro_f1"], res["spell_f1_mean"]))
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("\n".join(REPORT) + "\n")
        print(f"wrote {args.report}")


def main(argv=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*fragmented.*")
    warnings.filterwarnings("ignore", message=".*Skipping features without any observed values.*")
    warnings.filterwarnings("ignore", message=".*All-NaN.*")
    REPORT.clear()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data-dir")
    ap.add_argument("--game", default="BAR-ATH")
    ap.add_argument("--out", default="results/v2.csv")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--params", help="JSON overriding Params3 fields")
    ap.add_argument("--report", help="text file with what every module did")
    ap.add_argument("--dump", help="per-frame CSV for the debug tools")
    ap.add_argument("--save-state", help="pickle with features and probabilities for tools/dead_conditions.py")
    ap.add_argument("--dead-debug", help="prefix for the D debug files (why each frame / stretch is D)")
    ap.add_argument("--external", help="another game converted with tools/convert_metrica.py")
    ap.add_argument("--dead-method", choices=["v2", "hybrid"], default="hybrid",
                    help="how play-stopped (D) is decided: v2 = the time model alone (as before), "
                         "hybrid = the situation-based detector (params/dead_hybrid.txt)")
    ap.add_argument("--dead-config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))), "params", "dead_hybrid.txt"),
                    help="settings of the hybrid detector")
    ap.add_argument("--drop-short-d", type=float, default=0.0,
                    help="treat D stretches shorter than this many seconds as play (0 = off)")
    ap.add_argument("--team-train", nargs="+", metavar="NPZ",
                    help="train the H/A team model on these OTHER games' answer keys (files from "
                         "--team-features-out); a file of the game being labelled is skipped")
    ap.add_argument("--team-features-out", metavar="NPZ",
                    help="with --external: save this game's team features and its true H/A for --team-train")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gt")
    ap.add_argument("--n-frames", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--blocks", help=argparse.SUPPRESS)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
