"""Read the provided assets and write the submission.

Everything is indexed by VIDEO frame of the tactical recording (row i of the submission).
ATD frame_id counts from 1, so video_frame = frame_id - 1.

Coordinates: the ATD is normalised to 0-1. We convert to metres once, here, so every later
module works in metres and seconds.
"""
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

HOME, AWAY = 0, 1


@dataclass
class Tracking:
    px: np.ndarray        # (T, S) player x in metres, NaN when not tracked
    py: np.ndarray        # (T, S)
    pspeed: np.ndarray    # (T, S) provided speed m/s
    team: np.ndarray      # (S,) 0 = home, 1 = away
    bx: np.ndarray        # (T,) ball x in metres, NaN when not visible
    by: np.ndarray        # (T,)
    team_names: tuple     # (home name, away name) from the metadata

    @property
    def n_frames(self):
        return self.px.shape[0]


def _team_of_each_track(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(teams)}")
    home_id = teams[0].get("id")
    names = tuple((t.findtext("Name") or "").strip() for t in teams)
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([HOME if s == home_id else AWAY for s in slots], dtype=np.int8), names


def load_tracking(meta_path, raw_path, n_frames, pitch=(105.3, 68.0), cache=None):
    """Parse the raw ATD ("<frame>:<p1>;<p2>;...:<ball>", groups "x,y,speed").

    One streaming pass. With `cache` (an .npz path) the parsed arrays are reused next time.
    """
    team, names = _team_of_each_track(meta_path)
    if cache and os.path.isfile(cache):
        z = np.load(cache)
        return Tracking(z["px"], z["py"], z["pspeed"], team, z["bx"], z["by"], names)

    L, W = pitch
    S = len(team)
    px = np.full((n_frames, S), np.nan, np.float32)
    py = np.full((n_frames, S), np.nan, np.float32)
    ps = np.full((n_frames, S), np.nan, np.float32)
    bx = np.full(n_frames, np.nan, np.float32)
    by = np.full(n_frames, np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            t = int(parts[0]) - 1
            if not 0 <= t < n_frames:
                continue
            for s, group in enumerate(parts[1].split(";")[:S]):
                if not group or group[0] == "N":
                    continue
                v = group.split(",")
                px[t, s] = float(v[0]) * L
                py[t, s] = float(v[1]) * W
                if len(v) > 2 and v[2][:1] != "N":
                    ps[t, s] = float(v[2])
            ball = parts[2]
            if ball and ball[0] != "N":
                v = ball.split(",")
                bx[t] = float(v[0]) * L
                by[t] = float(v[1]) * W
    if cache:
        np.savez_compressed(cache, px=px, py=py, pspeed=ps, bx=bx, by=by)
    return Tracking(px, py, ps, team, bx, by, names)


MATRIX_COLUMNS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]


def load_homography(path, n_frames):
    """(H (T,3,3) mapping normalised pitch -> pixels, reprojection_error (T,), valid (T,))."""
    table = pd.read_csv(path)
    H = np.full((n_frames, 3, 3), np.nan)
    err = np.full(n_frames, np.nan)
    valid = np.zeros(n_frames, bool)
    f = table["frame"].values.astype(int)
    keep = (f >= 0) & (f < n_frames)
    H[f[keep]] = table[MATRIX_COLUMNS].values[keep].reshape(-1, 3, 3)
    err[f[keep]] = table["reprojection_error"].values[keep]
    valid[f[keep]] = table["valid"].values[keep].astype(bool)
    return H, err, valid


def load_smart_tagging(path, fps=25):
    """[{code, start, end, tags}] with start/end as video frames (end exclusive)."""
    out = []
    for inst in ET.parse(path).iter("instance"):
        s = float(inst.findtext("start"))
        e = float(inst.findtext("end"))
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start": int(round(s * fps)),
            "end": int(round(e * fps)),
            "tags": {lab.findtext("group"): (lab.findtext("text") or "").strip()
                     for lab in inst.findall("label")},
        })
    return out


def tag_team_side(tag_team, team_names):
    """Map a Smart Tagging 'Team' value (club name) to HOME/AWAY, or None for N/A or unknown.

    Matching is by shared words, case-insensitively, because the tagging and the metadata spell
    club names differently ("Atletic Club" vs "Athletic Club"). Check the printed mapping once.
    """
    if not tag_team or tag_team.upper() == "N/A":
        return None
    words = set(tag_team.lower().replace(".", " ").split())
    scores = []
    for side, name in enumerate(team_names):
        other = set(name.lower().replace(".", " ").split())
        scores.append(len(words & other) + sum(1 for w in words for o in other
                                               if len(w) > 3 and (w[:4] == o[:4])))
    if scores[0] == scores[1]:
        return None
    return HOME if scores[0] > scores[1] else AWAY


def graded_blocks(repo_root, game="BAR-ATH"):
    """[(half, first_frame, last_frame)] published in the repository (games.half_video_blocks).

    These are the published half boundaries (DATA.md). They are used only to write X outside the
    graded halves, never as labels.
    """
    sys.path.insert(0, repo_root)
    import games  # noqa: E402
    return games.half_video_blocks(game)


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{lab}\n" for i, lab in enumerate(labels))
    return path
