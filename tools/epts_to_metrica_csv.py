"""Metrica の Sample_Game_3（FIFA EPTS 形式: tracking.txt + metadata.xml + events.json）を、
Sample_Game_1 / 2 と同じ CSV 形式に変換する。tools/make_eval_data.py で EVAL-G3 を作る前に使う。
Convert Metrica Sample_Game_3 (FIFA EPTS) to the CSV layout of Sample_Game_1/2.

    python tools/epts_to_metrica_csv.py ../sample-data/data/Sample_Game_3 ../sample-data/data/Sample_Game_3_csv
    python tools/make_eval_data.py --src ../sample-data/data/Sample_Game_3_csv --out external/EVAL-G3 --seed 2 --stem EVAL-G3
"""
import sys, os, json, xml.etree.ElementTree as ET
import numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
meta = ET.parse(os.path.join(src, "Sample_Game_3_metadata.xml")).getroot()
team_of = {p.get("id"): p.get("teamId") for p in meta.iter("Player")}
name_of = {p.get("id"): p.find("Name").text for p in meta.iter("Player")}
ch_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
gp = {pp.find("Name").text: (pp.find("Value").text if pp.find("Value") is not None else None)
      for pp in meta.iter("ProviderParameter")}
h1 = (int(gp["first_half_start"]), int(gp["first_half_end"])); h2 = (int(gp["second_half_start"]), int(gp["second_half_end"]))
specs = []
for ds in meta.iter("DataFormatSpecification"):
    order = []
    for ref in ds.iter("PlayerChannelRef"):
        cid = ref.get("playerChannelId")
        if cid.endswith("_x"): order.append(ch_player[cid])
    specs.append((int(ds.get("startFrame")), int(ds.get("endFrame")), order))
home_ids = sorted({p for p, t in team_of.items() if t == "FIFATMA"}, key=lambda p: int(name_of[p].split()[-1]))
away_ids = sorted({p for p, t in team_of.items() if t == "FIFATMB"}, key=lambda p: int(name_of[p].split()[-1]))
lines = open(os.path.join(src, "Sample_Game_3_tracking.txt")).read().splitlines()
T = len(lines)
col = {p: i for i, p in enumerate(home_ids + away_ids)}
XY = np.full((T, len(col), 2), np.nan); B = np.full((T, 2), np.nan); F = np.zeros(T, int)
si = 0
for r, ln in enumerate(lines):
    fr, pl, ba = ln.split(":")
    fr = int(fr); F[r] = fr
    while not (specs[si][0] <= fr <= specs[si][1]): si = (si + 1) % len(specs)
    order = specs[si][2]
    for pid, xy in zip(order, pl.split(";")):
        x, y = xy.split(",")
        if x != "NaN": XY[r, col[pid]] = (float(x), float(y))
    bx, byy = ba.split(",")[:2]
    if bx != "NaN": B[r] = (float(bx), float(byy))
period = np.where(F <= h1[1], 1, 2)
def write(ids, fname, with_ball):
    hdr1 = ["", "", ""]; hdr2 = ["", "", ""]; hdr3 = ["Period", "Frame", "Time [s]"]
    data = {"Period": period, "Frame": F, "Time [s]": np.round(F / 25.0, 2)}
    cols = [period, F, np.round(F / 25.0, 2)]
    for p in ids:
        nm = name_of[p].replace(" ", "")
        hdr1 += ["Home" if team_of[p] == "FIFATMA" else "Away", ""]; hdr2 += [nm.replace("Player", ""), ""]; hdr3 += [nm, ""]
        cols += [XY[:, col[p], 0], XY[:, col[p], 1]]
    hdr1 += ["", ""]; hdr2 += ["", ""]; hdr3 += ["Ball", ""]
    cols += [B[:, 0], B[:, 1]]
    M = pd.DataFrame(np.column_stack(cols))
    with open(os.path.join(out, fname), "w", newline="") as fh:
        for h in (hdr1, hdr2, hdr3): fh.write(",".join(h) + "\n")
        M[0] = M[0].astype(int); M[1] = M[1].astype(int)
        M.to_csv(fh, header=False, index=False, na_rep="NaN", float_format="%.5f")
write(home_ids, "Sample_Game_3_RawTrackingData_Home_Team.csv", True)
write(away_ids, "Sample_Game_3_RawTrackingData_Away_Team.csv", True)
ev = json.load(open(os.path.join(src, "Sample_Game_3_events.json")))["data"]
rows = []
for e in ev:
    s = e.get("subtypes")
    sub = "" if s is None else (s["name"] if isinstance(s, dict) else "-".join(x["name"] for x in s))
    rows.append({"Team": "Home" if e["team"]["id"] == "FIFATMA" else "Away", "Type": e["type"]["name"], "Subtype": sub or np.nan,
                 "Period": e["period"], "Start Frame": e["start"]["frame"], "Start Time [s]": e["start"]["time"],
                 "End Frame": e["end"]["frame"], "End Time [s]": e["end"]["time"],
                 "From": (e.get("from") or {}).get("name", "").replace(" ", "") if e.get("from") else "",
                 "To": (e.get("to") or {}).get("name", "").replace(" ", "") if e.get("to") else "",
                 "Start X": e["start"]["x"], "Start Y": e["start"]["y"], "End X": e["end"]["x"], "End Y": e["end"]["y"]})
pd.DataFrame(rows).to_csv(os.path.join(out, "Sample_Game_3_RawEventsData.csv"), index=False)
print("frames", T, "home players", len(home_ids), "away", len(away_ids), "events", len(rows),
      "ball seen", np.isfinite(B[:, 0]).mean().round(3), "players per frame", np.isfinite(XY[:, :, 0]).sum(1).mean().round(1))
