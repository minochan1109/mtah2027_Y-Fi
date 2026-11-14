"""Ball possession for More than a Hack 2027, Phase 1.

Modules (one per function of the method):
    data           read ATD / homography / Smart Tagging, write the submission
    preprocess     metres, velocities, short ball gaps, camera aim point, attacking direction
    features       (1) dead-ball features, (2) ball-based and (3) shape-based team features
    pseudo_labels  algorithmic labels used for training (no human labels on the game)
    models         (1) logistic regression, (2)/(3) LightGBM, time-blocked CV, self-training
    hmm            (4) three-state HMM + Viterbi with a minimum stoppage length
    postprocess    labels, X outside the halves, summary statistics
    pipeline       command line entry point
"""
