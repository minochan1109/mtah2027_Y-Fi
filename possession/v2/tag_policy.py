"""[C] Which Smart Tagging tags are used, how, and how strongly.

Edit the two tables below; nothing else needs to change. Matching ignores upper/lower case.
Decide per CODE / TYPE TEXT for the whole match -- never per single tag instance (that would be
hand-labelling the provided game). Check the tables with tools/tag_stats.py.

ROLE meaning
    "attack"   the tagged team is probably in possession      -> pushes P(that team) up
    "defend"   the tagged team is probably defending           -> pushes P(other team) up
    "restart"  a set piece: play was dead just before, the tagged team restarts with the ball
    "goal"     a goal: play is dead after it until the kick-off
    "ignore"   not used

WEIGHT is in log-odds per tag (1.0 = moderate, 0.3 = weak). The intervals are phases, not
possessions (DATA.md), so keep the weights of attack/defend small.
"""
import xml.etree.ElementTree as ET

import numpy as np

CODE_POLICY = {
    # code                           role       weight
    "BUILD UP":                     ("attack",  0.3),
    "PROGRESSION":                  ("attack",  0.3),
    "CREATING CHANCES":             ("attack",  0.3),
    "ATTACKING TRANSITION":         ("attack",  0.3),
    "BALL IN FINAL THIRD":          ("attack",  0.3),
    "BALL IN THE BOX":              ("attack",  0.2),
    "PLAYERS IN THE BOX":           ("attack",  0.1),
    "LONG BALL":                    ("attack",  0.1),
    "DEFENDING IN ATTACKING THIRD": ("defend",  0.3),
    "DEFENDING IN MIDDLE THIRD":    ("defend",  0.3),
    "DEFENDING IN DEFENSIVE THIRD": ("defend",  0.3),
    "DEFENSIVE TRANSITION":         ("defend",  0.3),
    "SET PIECES":                   ("restart", 1.0),
    "GOALS":                        ("goal",    1.0),
}

# For SET PIECES: Type text -> (used?, team weight after the restart)
RESTART_TYPES = {
    "throw-in":              (True, 0.0),   # Team is always N/A for throw-ins
    "corner kick":           (True, 1.0),
    "goal kick":             (True, 1.0),
    "free kick":             (True, 1.0),
    "kick off (after goal)": (True, 0.0),
    "kick off (start)":      (True, 0.0),
}
RESTART_TEAM_S = 3.0      # the restarting team is favoured for this long after the tag start


def load_tags(path, fps=25):
    """Instances with all labels: start/end in video frames, code, labels [(group, text)]."""
    out = []
    for inst in ET.parse(path).iter("instance"):
        labels = [((lab.findtext("group") or "").strip(), (lab.findtext("text") or "").strip())
                  for lab in inst.findall("label")]
        tags = {}
        for g, t in labels:
            tags[g] = f"{tags[g]} / {t}" if g in tags else t
        out.append({"start": int(round(float(inst.findtext("start")) * fps)),
                    "end": int(round(float(inst.findtext("end")) * fps)),
                    "code": (inst.findtext("code") or "").strip().upper(),
                    "labels": labels, "tags": tags})
    return out


def _types(tag):
    return [t.lower() for g, t in tag["labels"] if g.lower() == "type"]


def tag_evidence(tags, side_of, n_frames, P):
    """Returns
        team_logit (T,)  log-odds evidence for HOME from tags (0 = no evidence)
        dead_feats dict  restart / goal timing features for the dead-ball model
        rep              how many tags were used per role
    """
    fps = P.fps
    logit = np.zeros(n_frames)
    to_restart = np.full(n_frames, np.inf)
    since_restart = np.full(n_frames, np.inf)
    since_goal = np.full(n_frames, np.inf)
    t = np.arange(n_frames)
    rep = {"attack": 0, "defend": 0, "restart": 0, "goal": 0, "ignored": 0}
    for g in tags or []:
        role, w = CODE_POLICY.get(g["code"], ("ignore", 0.0))
        s, e = max(0, g["start"]), min(n_frames, g["end"])
        if s >= n_frames or e <= 0:
            continue
        side = side_of(g["tags"].get("Team"))
        if role in ("attack", "defend"):
            if side is None:
                rep["ignored"] += 1
                continue
            home = (side == 0) == (role == "attack")
            logit[s:e] += w if home else -w
            rep[role] += 1
        elif role == "restart":
            types = _types(g)
            use = [RESTART_TYPES.get(ty, (False, 0.0)) for ty in types] or [(True, 0.0)]
            if not any(u for u, _ in use):
                rep["ignored"] += 1
                continue
            rep["restart"] += 1
            before = t <= s
            to_restart[before] = np.minimum(to_restart[before], (s - t[before]) / fps)
            after = t >= s
            since_restart[after] = np.minimum(since_restart[after], (t[after] - s) / fps)
            tw = max(wt for _, wt in use)
            if side is not None and tw > 0:
                k = min(n_frames, s + int(RESTART_TEAM_S * fps))
                logit[s:k] += tw if side == 0 else -tw
        elif role == "goal":
            rep["goal"] += 1
            after = t >= s
            since_goal[after] = np.minimum(since_goal[after], (t[after] - s) / fps)
        else:
            rep["ignored"] += 1
    cap = 120.0
    dead = {"tag_to_restart_s": np.minimum(to_restart, cap), "tag_since_restart_s": np.minimum(since_restart, cap),
            "tag_since_goal_s": np.minimum(since_goal, cap)}
    return logit * P.tag_logit_scale, dead, rep
