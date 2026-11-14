"""[E] Viterbi with a switch cost that changes over time.

The H<->A switch cost is lowered where a transition is signalled (players of one team suddenly
running back towards their own goal), and stays high elsewhere. Same D chain as version 1.
"""
import numpy as np

from ..hmm import A, D, H  # noqa: F401


def switch_signal(Xh, P):
    """0..1 per frame: how strongly the movement says 'possession just changed'."""
    onset = np.maximum(Xh["r_own_recover_onset"].values, Xh["r_opp_recover_onset"].values)
    z = P.switch_signal_slope * (np.nan_to_num(onset) - P.switch_signal_mid)
    return 1.0 / (1.0 + np.exp(-z))


def viterbi_tv(logE, P, switch_cost):
    """logE (T, 3); switch_cost (T,) cost of H<->A INTO frame t. Returns 0/1/2 per frame."""
    T = len(logE)
    m = P.frames(P.min_dead_s)
    K = 2 + m
    INF = 1e18
    trans = np.full((K, K), INF)
    trans[H, H] = trans[A, A] = 0.0
    trans[H, 2] = trans[A, 2] = P.cost_enter_dead
    for i in range(2, K - 1):
        trans[i, i + 1] = 0.0
    trans[K - 1, K - 1] = 0.0
    trans[K - 1, H] = trans[K - 1, A] = P.cost_leave_dead
    state_emis = np.r_[0, 1, np.full(m, 2)]
    cost = -logE[0][state_emis].copy()
    back = np.zeros((T, K), np.int16)
    cols = np.arange(K)
    for t in range(1, T):
        trans[H, A] = trans[A, H] = switch_cost[t]
        tot = cost[:, None] + trans
        back[t] = tot.argmin(axis=0)
        cost = tot[back[t], cols] - logE[t][state_emis]
    s = int(cost.argmin())
    path = np.empty(T, np.int16)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    return state_emis[path]


def viterbi_ha(p_home, switch_cost, floor=1e-4):
    """Two states only (0 = H, 1 = A) for one play stretch whose D is already decided.
    p_home (T,), switch_cost (T,) = cost of H<->A INTO frame t. Returns 0/1 per frame."""
    ph = np.clip(np.nan_to_num(p_home, nan=0.5), floor, 1 - floor)
    lh, la = np.log(ph), np.log(1 - ph)
    T = len(ph)
    back = np.zeros((T, 2), np.int8)
    c0, c1 = -lh[0], -la[0]
    for t in range(1, T):
        sc = switch_cost[t]
        b0 = 0 if c0 <= c1 + sc else 1
        b1 = 1 if c1 <= c0 + sc else 0
        n0 = (c0 if b0 == 0 else c1 + sc) - lh[t]
        n1 = (c1 if b1 == 1 else c0 + sc) - la[t]
        back[t, 0], back[t, 1] = b0, b1
        c0, c1 = n0, n1
    s = 0 if c0 <= c1 else 1
    path = np.empty(T, np.int8)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    return path


def decode_with_dead(dead, p_home, switch_cost, blocks, n_frames, floor=1e-4):
    """Final labels when D is fixed: 'D' where dead, H/A from viterbi_ha in every play stretch,
    'X' outside the graded halves."""
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, a, b in blocks:
        seg = dead[a:b + 1]
        labels[a:b + 1] = np.where(seg, "D", "")
        m = np.r_[False, ~seg, False].astype(np.int8)
        d = np.diff(m)
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            path = viterbi_ha(p_home[a + s:a + e], switch_cost[a + s:a + e], floor)
            labels[a + s:a + e] = np.where(path == 0, "H", "A")
    return labels
