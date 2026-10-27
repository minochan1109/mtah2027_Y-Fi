"""Read the Smart Tagging XML keeping EVERY label (group, text), for the debug tools.

Each instance becomes
    {"id", "start", "end" (video frames, end exclusive), "start_s", "end_s", "code",
     "labels": [(group, text), ...], "tags": {group: "text1 / text2"}, "name"}
"name" is what the debug tools show: the Type text when there is one (e.g. "Kick Off (Start)"),
otherwise the code; texts of other groups (not Team/Half) are added in brackets,
e.g. "ATTACKING TRANSITION (Diagonal)".
"""
import xml.etree.ElementTree as ET

FPS = 25
NOT_NAME = ("Team", "Half")


def load_tags(path, fps=FPS):
    out = []
    for inst in ET.parse(path).iter("instance"):
        s = float(inst.findtext("start"))
        e = float(inst.findtext("end"))
        labels = [((lab.findtext("group") or "").strip(), (lab.findtext("text") or "").strip())
                  for lab in inst.findall("label")]
        tags = {}
        for g, t in labels:
            tags[g] = f"{tags[g]} / {t}" if g in tags else t
        code = (inst.findtext("code") or "").strip()
        types = [t for g, t in labels if g == "Type" and t]
        others = [t for g, t in labels if g not in NOT_NAME + ("Type",) and t]
        name = " / ".join(types) if types else code
        if others:
            name += " (" + ", ".join(others) + ")"
        out.append({"id": (inst.findtext("ID") or "").strip(), "start": int(round(s * fps)),
                    "end": int(round(e * fps)), "start_s": s, "end_s": e, "code": code,
                    "labels": labels, "tags": tags, "name": name})
    return out
