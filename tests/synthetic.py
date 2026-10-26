"""A small synthetic match written in the SAME file formats as the real bundle.

It exists only to test that the pipeline runs end to end without the real data (which must not
leave your machine). Its numbers say nothing about real performance.

    python tests/synthetic.py --out /tmp/synth        # writes data/ and gt.csv
"""
import argparse
import os

import numpy as np

L, W, FPS = 105.3, 68.0, 25
SLOTS_PER_TEAM = 23


def simulate(T=30000, blocks=((1, 500, 14500), (2, 15500, 29500)), seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([[5, 34], [20, 10], [20, 26], [20, 42], [20, 58], [35, 15], [35, 34], [35, 53],
                     [48, 12], [48, 34], [48, 56]], float)
    pos = np.zeros((2, 11, 2))
    pos[0] = base
    pos[1] = np.c_[L - base[:, 0], base[:, 1]]
    ball = np.array([L / 2, W / 2])
    jog = np.zeros((2, 11, 2))          # persistent individual running, like real players
    px = np.full((T, 2, 11), np.nan)
    py = np.full((T, 2, 11), np.nan)
    bx = np.full(T, np.nan)
    by = np.full(T, np.nan)
    gt = np.full(T, "X", dtype="<U1")
    events = []

    for half, a, b in blocks:
        attack = 1 if half == 1 else -1        # home attacks +x in the first half
        team, carrier = 0, 9
        state = "dead"
        dead_left = 5 * FPS
        next_pass = 0
        flight = None
        alive_left = 0
        for t in range(a, b + 1):
            if state == "dead":
                dead_left -= 1
                for k in (0, 1):
                    jog[k] = 0.97 * jog[k] + rng.normal(0, 0.12, (11, 2))
                    pos[k] += jog[k] / FPS
                if dead_left <= 0:
                    state = "alive"
                    team = rng.integers(2)
                    carrier = rng.integers(1, 11)
                    ball = pos[team, carrier].copy()
                    alive_left = int(rng.uniform(40, 90) * FPS)
                    next_pass = t + int(rng.uniform(1.5, 5) * FPS)
                    events.append(("SET PIECES", t / FPS - 1, t / FPS + 3, team, half))
            else:
                alive_left -= 1
                if flight is not None:
                    tgt_team, tgt, steps = flight
                    target = pos[tgt_team, tgt]
                    d = target - ball
                    step = 15.0 / FPS
                    if np.hypot(*d) <= step:
                        ball = target.copy()
                        team, carrier, flight = tgt_team, tgt, None
                        next_pass = t + int(rng.uniform(1.5, 5) * FPS)
                    else:
                        ball = ball + d / np.hypot(*d) * step
                else:
                    sgn = attack if team == 0 else -attack
                    move = np.array([sgn * 3.0, rng.normal(0, 2)]) / FPS
                    pos[team, carrier] += move
                    ball = pos[team, carrier] + np.array([0.5 * sgn, 0.0])
                    if t >= next_pass:
                        if rng.random() < 0.8:
                            choices = [i for i in range(1, 11) if i != carrier]
                            flight = (team, rng.choice(choices), 0)
                        else:
                            flight = (1 - team, rng.integers(1, 11), 0)
                # team movement: attackers spread and advance, defenders compress to the ball
                for k in (0, 1):
                    sgn = attack if k == 0 else -attack
                    home_side = base if (k == 0) == (attack == 1) else np.c_[L - base[:, 0], base[:, 1]]
                    if k == team:
                        target = home_side + np.array([sgn * 15, 0]) + (ball - [L / 2, W / 2]) * 0.3
                    else:
                        target = home_side * 0.6 + ball * 0.4
                    jog[k] = 0.97 * jog[k] + rng.normal(0, 0.45, (11, 2))
                    vel = np.clip((target - pos[k]) * 0.6 + jog[k], -7, 7)
                    if k == team and flight is None:
                        vel[carrier] = 0
                    pos[k] += vel / FPS
                if alive_left <= 0 and flight is None:
                    state = "dead"
                    dead_left = int(rng.uniform(8, 20) * FPS)
                    ball = np.array([ball[0], -1.5 if ball[1] < W / 2 else W + 1.5])
            pos[:, :, 0] = np.clip(pos[:, :, 0], 0, L)
            pos[:, :, 1] = np.clip(pos[:, :, 1], 0, W)
            px[t], py[t] = pos[:, :, 0], pos[:, :, 1]
            bx[t], by[t] = ball
            gt[t] = "D" if state == "dead" else ("H" if team == 0 else "A")

    # observation noise: ball missing in bursts (~40%), far players sometimes missing
    seen = np.ones(T, bool)
    t = 0
    while t < T:
        if rng.random() < 0.012:
            k = int(rng.uniform(0.2, 3.0) * FPS)
            seen[t:t + k] = False
            t += k
        t += 1
    bx[~seen] = np.nan
    by[~seen] = np.nan
    aim = np.c_[np.convolve(np.nan_to_num(bx, nan=L / 2), np.ones(50) / 50, "same"),
                np.convolve(np.nan_to_num(by, nan=W / 2), np.ones(50) / 50, "same")]
    far = np.hypot(px - aim[:, 0, None, None], py - aim[:, 1, None, None]) > 45
    drop = far & (rng.random(far.shape) < 0.5)
    px[drop] = np.nan
    py[drop] = np.nan
    return px, py, bx, by, gt, aim, events


def write_bundle(out, T=30000, blocks=((1, 500, 14500), (2, 15500, 29500)), seed=0):
    px, py, bx, by, gt, aim, events = simulate(T, blocks, seed)
    folder = os.path.join(out, "data", "tactical")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, "BAR-ATH_tactical")

    with open(stem + "_atd.xml", "w") as fh:
        fh.write('<?xml version="1.0"?>\n<Root><Teams>'
                 '<Team id="T1"><Name>FC Barcelona</Name></Team>'
                 '<Team id="T2"><Name>Athletic Club</Name></Team></Teams><Players>')
        for k, tid in enumerate(("T1", "T2")):
            for s in range(SLOTS_PER_TEAM):
                fh.write(f'<Player id="P{k}_{s}" teamId="{tid}"><Name>Track {k * SLOTS_PER_TEAM + s + 1}</Name></Player>')
        fh.write("</Players></Root>\n")

    with open(stem + "_atd.txt", "w") as fh:
        for t in range(T):
            groups = []
            for k in (0, 1):
                for s in range(SLOTS_PER_TEAM):
                    if s < 11 and not np.isnan(px[t, k, s]):
                        groups.append(f"{px[t, k, s] / L:.5f},{py[t, k, s] / W:.5f},0.0")
                    else:
                        groups.append("NaN,NaN,NaN")
            ball = "NaN,NaN,NaN" if np.isnan(bx[t]) else f"{bx[t] / L:.5f},{by[t] / W:.5f},NaN"
            fh.write(f"{t + 1}:{';'.join(groups)}:{ball}\n")

    with open(stem + "_homography.csv", "w") as fh:
        fh.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
        sx = 1920 / 60.0
        for t in range(T):
            ax, ay = aim[t]
            fh.write(f"{t},{sx * L},0,{960 - sx * ax},0,{sx * W},{540 - sx * ay},0,0,1,0.2,1\n")

    names = ("Barcelona", "Atletic Club")
    with open(stem + "_smart_tagging.xml", "w") as fh:
        fh.write("<file><ALL_INSTANCES>")
        for i, (code, s, e, team, half) in enumerate(events):
            fh.write(f"<instance><ID>{i + 1}</ID><start>{s:.2f}</start><end>{e:.2f}</end><code>{code}</code>"
                     f"<label><group>Team</group><text>{names[team]}</text></label>"
                     f"<label><group>Half</group><text>{'1st' if half == 1 else '2nd'} Half</text></label></instance>")
        fh.write("</ALL_INSTANCES></file>\n")

    with open(os.path.join(out, "gt.csv"), "w") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{t},{lab}\n" for t, lab in enumerate(gt))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    write_bundle(ap.parse_args().out)
