"""Count what is in the Smart Tagging file: codes, label groups, label texts and their combinations.

    python tools/tag_stats.py --repo ../more-than-a-hack-2027 --out results/tag_stats.txt
    python tools/tag_stats.py --xml path/to/BAR-ATH_tactical_smart_tagging.xml

Prints to the screen and (with --out) writes the same text to a file.
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tag_io import FPS, NOT_NAME, load_tags  # noqa: E402


def tstr(sec):
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


def report(tags):
    L = []
    add = L.append
    add(f"SMART TAGGING STATS: {len(tags)} instances")
    if not tags:
        return L
    add(f"time range: {tstr(min(g['start_s'] for g in tags))} - {tstr(max(g['end_s'] for g in tags))} (video time)")
    add("")

    add("1) BY CODE  (count, total seconds, mean seconds)")
    by_code = defaultdict(list)
    for g in tags:
        by_code[g["code"]].append(g["end_s"] - g["start_s"])
    for c, d in sorted(by_code.items(), key=lambda kv: -len(kv[1])):
        add(f"   {c:<40} {len(d):5d}  {sum(d):8.1f} s  {sum(d) / len(d):6.2f} s")
    add("")

    add("2) LABEL GROUPS  (in how many instances the group appears)")
    groups = Counter(gr for g in tags for gr in {gr for gr, _ in g["labels"]})
    for gr, n in groups.most_common():
        add(f"   {gr:<40} {n:5d}")
    add("")

    add("3) TEXTS PER GROUP  (count)")
    per = defaultdict(Counter)
    for g in tags:
        for gr, t in g["labels"]:
            per[gr][t] += 1
    for gr in [gr for gr, _ in groups.most_common()]:
        add(f"   [{gr}]  {len(per[gr])} different texts")
        for t, n in per[gr].most_common():
            add(f"      {t:<45} {n:5d}")
    add("")

    add("4) CODE x LABEL TEXT  (texts other than Team/Half; count, mean seconds)")
    combo = defaultdict(list)
    for g in tags:
        texts = [f"{gr}={t}" for gr, t in g["labels"] if gr not in NOT_NAME] or ["(no other label)"]
        for t in texts:
            combo[(g["code"], t)].append(g["end_s"] - g["start_s"])
    for code in sorted(by_code, key=lambda c: -len(by_code[c])):
        add(f"   {code}")
        rows = sorted([(t, d) for (c, t), d in combo.items() if c == code], key=lambda kv: -len(kv[1]))
        for t, d in rows:
            add(f"      {t:<55} {len(d):5d}  {sum(d) / len(d):6.2f} s")
    add("")

    add("5) TEAM PER CODE  (count)")
    for code in sorted(by_code, key=lambda c: -len(by_code[c])):
        cnt = Counter(g["tags"].get("Team", "(none)") for g in tags if g["code"] == code)
        add(f"   {code:<40} " + ", ".join(f"{k}: {v}" for k, v in cnt.most_common()))
    add("")

    add("6) HALF  (count)")
    for t, n in Counter(g["tags"].get("Half", "(none)") for g in tags).most_common():
        add(f"   {t:<20} {n:5d}")
    add("")

    add("7) INSTANCES WITH THE SAME GROUP TWICE OR MORE (first 20)")
    dup = [g for g in tags if len({gr for gr, _ in g["labels"]}) < len(g["labels"])]
    add(f"   {len(dup)} instances")
    for g in dup[:20]:
        add(f"   ID {g['id']}  {tstr(g['start_s'])}  {g['code']}  " + "; ".join(f"{a}={b}" for a, b in g["labels"]))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="more-than-a-hack-2027 clone (finds the XML in its data folder)")
    ap.add_argument("--xml", help="path to the smart tagging XML (instead of --repo)")
    ap.add_argument("--game", default="BAR-ATH")
    ap.add_argument("--data-dir")
    ap.add_argument("--out", help="also write the report to this text file")
    a = ap.parse_args(argv)
    path = a.xml
    if not path:
        if not a.repo:
            raise SystemExit("give --repo or --xml")
        repo = os.path.abspath(a.repo)
        sys.path.insert(0, repo)
        import games  # noqa: E402
        data_dir = a.data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(repo, "data")
        path = games.asset_paths(a.game, data_dir)["smart_tagging"]
    if not os.path.isfile(path):
        raise SystemExit(f"not found: {path}")
    text = "\n".join(report(load_tags(path, FPS))) + "\n"
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
