"""Version 2 (sample implementation): base = no-ball evidence, the ball is ADDED when it is trusted.

One file per function, so each person can own and improve one part:

    config_v2.py      every number of version 2, grouped by the file that uses it
    clean_players.py  [A] remove false player detections, fix the goalkeeper's team
    ball_quality.py   [B] judge every ball detection (confidence 0-1), interpolate only trusted ones,
                          flag balls in the air
    tag_policy.py     [C] which Smart Tagging texts are used, how (restart / attack / defend / ignore)
                          and how strongly
    features_v2.py    [D] base features (shape, containment, recovery runs) + ball add-on features
    pseudo_v2.py      [D] pseudo labels with the cleaned data ("possession stays until the other
                          team controls the ball")
    model_v2.py       [D] team model (single LightGBM with ball columns blank when untrusted,
                          or base model + ball add-on in log-odds)
    hmm_v2.py         [E] Viterbi whose switch cost drops when a transition is signalled
    pipeline_v2.py        wiring, --external scoring on other games, --report diagnostics

Run:
    python -m possession.v2.pipeline_v2 --repo ../more-than-a-hack-2027 --out results/v2.csv --report results/v2_report.txt
"""
