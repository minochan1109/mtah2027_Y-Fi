"""Module 4: three-state HMM (H / A / D) decoded with Viterbi.

The dead state is expanded into a chain D1 -> D2 -> ... -> Dm so that a stoppage, once entered,
lasts at least m frames (a simple semi-Markov constraint). Costs are negative log probabilities.
"""
import numpy as np

H, A, D = 0, 1, 2


def emissions(p_dead, p_home, P):
    """Per-frame log probabilities for H, A, D. NaN p_home means 'no evidence' (0.5)."""
    f = P.prob_floor
    pd_ = np.clip(np.nan_to_num(p_dead, nan=0.5), f, 1 - f)
    ph = np.clip(np.nan_to_num(p_home, nan=0.5), f, 1 - f)
    return np.log(np.stack([(1 - pd_) * ph, (1 - pd_) * (1 - ph), pd_], axis=1))


def tagging_prior(tags, side_of, n_frames, P):
    """P(home) suggested by Smart Tagging phases, NaN where no team tag applies.

    Attacking phases and set pieces point to the tagged team; defending phases to the other one.
    """
    attack = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
              "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL", "SET PIECES"}
    defend = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
              "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
    votes = np.zeros(n_frames)
    count = np.zeros(n_frames)
    for e in tags or []:
        side = side_of(e["tags"].get("Team"))
        if side is None or e["code"] not in attack | defend:
            continue
        home = (side == 0) == (e["code"] in attack)
        s, t = max(0, e["start"]), min(n_frames, e["end"])
        votes[s:t] += 1.0 if home else 0.0
        count[s:t] += 1.0
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, votes / np.maximum(count, 1), np.nan)


def combine(p_home, prior, P):
    w = P.tagging_weight
    return np.where(np.isnan(prior), p_home, (1 - w) * np.nan_to_num(p_home, nan=0.5) + w * prior)


def viterbi(logE, P):
    """Most likely H/A/D sequence. logE: (T, 3). Returns int array of 0/1/2."""
    T = len(logE)
    m = P.frames(P.min_dead_s)
    K = 2 + m                                     # H, A, D1..Dm
    INF = 1e18
    trans = np.full((K, K), INF)                  # trans[i, j] = cost of i -> j
    trans[H, H] = trans[A, A] = 0.0
    trans[H, A] = trans[A, H] = P.cost_switch_team
    trans[H, 2] = trans[A, 2] = P.cost_enter_dead
    for i in range(2, K - 1):
        trans[i, i + 1] = 0.0
    trans[K - 1, K - 1] = 0.0
    trans[K - 1, H] = trans[K - 1, A] = P.cost_leave_dead

    state_emis = np.r_[0, 1, np.full(m, 2)]
    cost = -logE[0][state_emis].copy()
    back = np.zeros((T, K), np.int16)
    for t in range(1, T):
        tot = cost[:, None] + trans
        back[t] = tot.argmin(axis=0)
        cost = tot[back[t], np.arange(K)] - logE[t][state_emis]
    s = int(cost.argmin())
    path = np.empty(T, np.int16)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    return state_emis[path]
