"""Tools for evaluation of results of dynamical systems."""

import numpy as np


def k_step_extrapolation_error(target, estimate):
    """Computes k-step extrapolation error.

    Computes k-step l2-error of two sets of trajectories.
    N: number of trajectories.
    T: time steps in the trajectories
    D: dimensionality of the state space.
    M: nummber os samples of estimate trajectory distribution.

    params:
    -------
    target: np.ndarray
        Shape is (N, T, D)
    estimate: np.ndarray
        Shape is (N, M, T, D)

    returns:
    --------
    Tuple of np.array, each of size T.
    """
    if not(isinstance(
        target, np.ndarray) and isinstance(estimate, np.ndarray)):
        raise ValueError("target and estimate should be np.ndarray.")
    msg = "target must have shape (N, T, D)"
    msg += " and estimate must have shape (N, M, T, D)"
    # expand target to match estimate.
    tshape = target.shape
    eshape = estimate.shape
    if not(len(tshape) == 3):
        raise ValueError(msg)
    if not(len(eshape) == 4):
        raise ValueError(msg)
    if not tshape == eshape[:1] + eshape[2:]:
        raise ValueError(msg)

    target = target[:, None]
    # Sum square diff over dimensions
    diff = np.sqrt(np.square(target - estimate).sum(axis=-1))
    # Get mean accross samples
    diff_std = diff.mean(axis=1).mean(0)
    diff = diff.mean(axis=1).mean(0)

    return diff, diff_std
