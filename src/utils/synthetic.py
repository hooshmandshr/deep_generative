"""Tools for simulating synthetic data sets using dynamical systems."""


import numpy as np


def check_input_shape(x, expected_dim):
    """Checks input shape to be of the expected dimension

    params:
    -------
    x: np.ndarray
    expected_dim: int

    returns:
    --------
    bool
    """
    if not len(x.shape) == 2:
        raise ValueError("x must have shape (N, D).")
    if not x.shape[1] == expected_dim:
        raise ValueError("x must be of shape (N, {}).".format(expected_dim))

def fhn(x, a=.7, b=.8, tau=12.5, i_ext=1.42):
    """Rate of change of ODE defined by Fitzhugh-Nagumo system."""
    check_input_shape(x, expected_dim=2)
    v, w = x[:, 0], x[:, 1]
    dx = np.zeros_like(x)

    dx[:, 0] = v - np.power(v, 3) / 3. - w + i_ext
    dx[:, 1] = (v + a - b * w) / tau
    return dx


def rossler(x, a=.1, b=.1, c=14.):
    """Rate of change of ODE defined by Rossler system."""
    check_input_shape(x, expected_dim=3)
    x_, y_, z_ = x[:, 0], x[:, 1], x[:, 2]
    dx = np.zeros_like(x)

    dx[:, 0] = - y_ - z_
    dx[:, 1] = x_ + a * y_
    dx[:, 2] = b + z_ * (x_ - c)
    return dx


def lorenz(x, sigma=10., beta=8/3., rho=28.):
    """Rate of change of ODE defined by Lorenz system."""
    check_input_shape(x, expected_dim=3)
    x_, y_, z_ = x[:, 0], x[:, 1], x[:, 2]
    dx = np.zeros_like(x)

    dx[:, 0] = sigma * (y_ - x_)
    dx[:, 1] = x_ * (rho - z_) - y_
    dx[:, 2] = x_ * y_ - beta * z_
    return dx


def euler_solve_forward(x, f, time, d_time=0.05, noise_covar=None, **kwargs):
    """Euler discretization of ODE solve forward given inital points.

    params:
    -------
    x: np.ndarray
        Initial points, shape must be (N, D) where N is # points and D is
        dimensionality of the state space.
    f: function
        Signature function for the rate of change given a point is state space.
    d_time: float
        Time discretization value.
    noise_covar: None or ndarray.
        If None, the system evovles deterministically. Otherwise, noise_covar
        is the time invariant covariance of the Gaussian white noise.
    """
    trajectory = np.zeros((time,) + x.shape)

    n_points, dim = x.shape

    if noise_covar is not None:
        for t in range(time):
            trajectory[t] = x
            x = x + d_time * f(x, **kwargs) + np.random.multivariate_normal(
                np.zeros(dim), noise_covar, n_points)
    else:
        for t in range(time):
            trajectory[t] = x
            x = x + d_time * f(x, **kwargs)
    return trajectory.transpose([1, 0, 2])
