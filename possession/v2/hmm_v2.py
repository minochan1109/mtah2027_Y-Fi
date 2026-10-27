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
