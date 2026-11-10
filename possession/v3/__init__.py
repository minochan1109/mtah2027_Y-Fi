"""Version 3 = version 2 + the decided play-stopped detector (MT-11) and H/A decoding with D fixed (MT-24).

Version 2 is NOT changed: v3 imports the v2 modules that it uses as they are ([A] clean_players,
[B] ball_quality, [C] tag_policy, [D] features_v2 / pseudo_v2 / model_v2, dead_debug), so improvements
pushed to those v2 files are picked up by v3 automatically. Only the new or changed parts live here:

    config_v3.py           Params3 = Params2 + the v3 settings (defaults = the decided version)
    pipeline_v3.py         wiring: [A] -> [B] -> [C] -> [D] -> [D2] play stopped -> [E] H/A -> CSV
    dead_situation.py      [D2] the hybrid play-stopped detector (core -> start / end -> situations
                           -> play signs -> 3-second rule), settings in params/dead_hybrid.txt
    situation_features.py  per-frame player / ball-position / tag columns used by the detector
    expr.py                the small expression language of params/dead_hybrid.txt
    hmm_v3.py              [E] H/A only, inside every play stretch, with D fixed
    soccer_rules.py        optional H/A corrections from football facts (off by default)
    (MT-26 clues: possession/v2/clue_features.py is used as is)

Run (the submission):
    python -m possession.v3.pipeline_v3 --repo ../more-than-a-hack-2027 --out results/final/submission.csv
"""
