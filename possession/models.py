"""Module 1 (logistic regression) and modules 2/3 (LightGBM), trained on pseudo labels.

Cross-validation uses contiguous time blocks, never random frames: neighbouring frames are almost
identical, so a random split leaks the answer into training and inflates every score.
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


def time_blocks(n, P):
    return np.arange(n) // P.frames(P.cv_block_s)


def _subsample(idx, step):
    return idx[::step] if step > 1 else idx


# --------------------------------------------------------------------------- module 1
class DeadModel:
    def __init__(self, P, step=5):
        self.P, self.step = P, step
        self.model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                   StandardScaler(),
                                   LogisticRegression(C=0.5, max_iter=2000))

    def fit(self, X, y):
        idx = _subsample(np.flatnonzero(y >= 0), self.step)
        if len(np.unique(y[idx])) < 2:
            raise ValueError("dead pseudo labels contain one class only -- check the thresholds")
        self.cols = list(X.columns)
        self.model.fit(X.iloc[idx].values, y[idx])
        return self

    def predict(self, X):
        return self.model.predict_proba(X[self.cols].values)[:, 1]

    def coefficients(self):
        lr = self.model[-1]
        names = self.cols + [f"missing_{c}" for c in np.array(self.cols)[self.model[0].indicator_.features_]]
        return sorted(zip(names, lr.coef_[0]), key=lambda t: -abs(t[1]))


# --------------------------------------------------------------------------- modules 2 and 3
class TeamModel:
    """P(home has the ball) from symmetric own/opp features (home view + away view)."""

    def __init__(self, P, cols, step=3):
        if lgb is None:
            raise ImportError("pip install lightgbm")
        self.P, self.cols, self.step = P, cols, step

    def _stack(self, Xh, Xa, y, idx):
        X = np.vstack([Xh.iloc[idx][self.cols].values, Xa.iloc[idx][self.cols].values])
        t = np.r_[(y[idx] == 1).astype(int), (y[idx] == 0).astype(int)]  # own team has the ball
        return X, t

    def fit(self, Xh, Xa, y, idx=None):
        if idx is None:
            idx = np.flatnonzero(y >= 0)
        idx = _subsample(idx, self.step)
        X, t = self._stack(Xh, Xa, y, idx)
        self.model = lgb.LGBMClassifier(**self.P.lgbm).fit(X, t)
        return self

    def predict(self, Xh, Xa, rows=None):
        rows = np.arange(len(Xh)) if rows is None else rows
        ph = self.model.predict_proba(Xh.iloc[rows][self.cols].values)[:, 1]
        pa = self.model.predict_proba(Xa.iloc[rows][self.cols].values)[:, 1]
        return 0.5 * (ph + (1.0 - pa))          # symmetric estimate of P(home)

    def oof(self, Xh, Xa, y, rows):
        """Out-of-fold P(home) on `rows`, time-blocked; used for self-training and diagnostics."""
        groups = time_blocks(len(Xh), self.P)
        out = np.full(len(Xh), np.nan)
        lab = np.flatnonzero(y >= 0)
        gkf = GroupKFold(n_splits=self.P.n_folds)
        for tr_g, te_g in gkf.split(rows, groups=groups[rows]):
            test_rows = rows[te_g]
            test_groups = np.unique(groups[test_rows])
            train = lab[~np.isin(groups[lab], test_groups)]
            m = TeamModel(self.P, self.cols, self.step).fit(Xh, Xa, y, train)
            out[test_rows] = m.predict(Xh, Xa, test_rows)
        return out

    def importance(self, top=15):
        imp = self.model.booster_.feature_importance("gain")
        return sorted(zip(self.cols, imp), key=lambda t: -t[1])[:top]


def self_train(model_cls, P, cols, Xh, Xa, y, rows):
    """Add confident out-of-fold predictions as extra pseudo labels, P.self_training_rounds times."""
    y = y.copy()
    for _ in range(P.self_training_rounds):
        m = model_cls(P, cols)
        p = m.oof(Xh, Xa, y, rows)
        new = rows[(y[rows] < 0) & ((p[rows] > P.self_training_conf) | (p[rows] < 1 - P.self_training_conf))]
        y[new] = (p[new] > 0.5).astype(np.int8)
    return model_cls(P, cols).fit(Xh, Xa, y), y
