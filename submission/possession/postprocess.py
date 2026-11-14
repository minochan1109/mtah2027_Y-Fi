"""Turn decoded states into submission labels, and describe the result."""
import numpy as np

LETTERS = np.array(["H", "A", "D"])


def to_labels(states_by_half, n_frames, blocks):
    """Graded halves get H/A/D (never X: X only costs macroF1 there); everything else gets X."""
    labels = np.full(n_frames, "X", dtype="<U1")
    for (_, a, b), states in zip(blocks, states_by_half):
        labels[a:b + 1] = LETTERS[states]
    return labels


def shift_transitions(labels, k):
    """Delay (k>0) or advance (k<0) every label change by k frames. Default 0.

    Only change k from reasoning or from other games -- never from the public 5 minutes.
    """
    if k == 0:
        return labels
    out = labels.copy()
    if k > 0:
        out[k:] = labels[:-k]
    else:
        out[:k] = labels[-k:]
    out[labels == "X"] = "X"
    return out


def summary(labels, blocks, fps):
    """Possession count and durations -- compare with football common sense, not with the GT."""
    spells, deads = [], []
    for _, a, b in blocks:
        seg = labels[a:b + 1]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.r_[0, change]
        stops = np.r_[change, len(seg)]
        for s, e in zip(starts, stops):
            (spells if seg[s] in "HA" else deads).append((e - s) / fps)
    spells = np.array(spells) if spells else np.zeros(1)
    deads = np.array(deads) if deads else np.zeros(1)
    graded = np.concatenate([labels[a:b + 1] for _, a, b in blocks])
    return {
        "possessions": int(len(spells)),
        "possession_mean_s": float(spells.mean()),
        "possession_median_s": float(np.median(spells)),
        "short_possessions_lt1s": int((spells < 1).sum()),
        "dead_stretches": int(len(deads)),
        "dead_mean_s": float(deads.mean()),
        "share_H": float((graded == "H").mean()),
        "share_A": float((graded == "A").mean()),
        "share_D": float((graded == "D").mean()),
    }
