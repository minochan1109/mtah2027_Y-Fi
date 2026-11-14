"""[D] Team model of version 2.

"additive" (default) base model on base columns (every frame) + ball model on ball columns
           (trusted frames); logit(P) = logit(P_base) + ball_weight * conf * logit(P_ball)
"single"   one LightGBM on base + ball columns; ball columns are NaN where the ball is not trusted,
           and LightGBM learns what to do when they are missing (falls back to the base columns)
"""
import numpy as np

from .. import models
from . import features_v2


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_predict(Xh, Xa, y_team, conf, P, log=print):
    rows = np.arange(len(Xh))
    if P.team_model == "single":
        cols = features_v2.all_columns(Xh)
        m, _ = models.self_train(models.TeamModel, P, cols, Xh, Xa, y_team, rows)
        log("team model (single) top features: " + ", ".join(n for n, _ in m.importance(8)))
        return m.predict(Xh, Xa), {"single": m}
    base_cols = features_v2.base_columns(Xh)
    ball_cols = features_v2.ball_columns(Xh)
    mb, y2 = models.self_train(models.TeamModel, P, base_cols, Xh, Xa, y_team, rows)
    p_base = mb.predict(Xh, Xa)
    trusted = np.flatnonzero(conf >= P.ball_conf_min)
    mball = models.TeamModel(P, ball_cols).fit(Xh, Xa, y2, trusted[y2[trusted] >= 0])
    p_ball = np.full(len(Xh), 0.5)
    p_ball[trusted] = mball.predict(Xh, Xa, trusted)
    z = _logit(p_base) + P.ball_weight * conf * _logit(p_ball)
    log("base model top features: " + ", ".join(n for n, _ in mb.importance(8)))
    log("ball model top features: " + ", ".join(n for n, _ in mball.importance(6)))
    return 1 / (1 + np.exp(-z)), {"base": mb, "ball": mball}
