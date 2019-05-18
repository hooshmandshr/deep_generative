"""Testing dynamics.py classes."""


import numpy as np
import tensorflow as tf

from dynamics import KalmanFilter, MarkovDynamicsDiagnostics, MLPDynamics
from transform import MultiLayerPerceptron


def test_kalman_filter(full_covar=False):
    """Tester for Kalman Filter and dynamics diagnostics class."""

    obs_dim = 7
    lat_dim = 2
    time = 29
    n_examples = 13
    order = 1

    trans_matrix = np.append(
        np.eye(2) + 0.5 * np.array([[ 0.4, -1.6], [0.4, -0.8]]),
        np.zeros([1, 2]),
        axis=0)

    x = np.arange(-10, 10)
    y = np.arange(-10, 10)
    u, v = np.meshgrid(x, y)
    grid = np.array([u.ravel(), v.ravel()]).T
    grid_size = len(grid)

    with tf.Graph().as_default():

        kf = KalmanFilter(
                lat_dim=lat_dim, obs_dim=obs_dim,
                time_steps=time,
                init_transition_matrix_bias=trans_matrix,
                full_covariance=full_covar, order=order)
        # sample 10 additional steps in time
        samples = kf.sample(n_examples, time_steps=time)

        n_samples = 17
        time_forward = 11
        diag = MarkovDynamicsDiagnostics(
                dynamics=kf, n_samples=n_samples,
                grid_size=grid_size, time_forward=time_forward)

        # Multiple initial states with single smaples each
        init_states = np.random.normal(3, 1, [n_samples, order, lat_dim])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            data = sess.run(samples)
            extrapolation1 = diag.run_extrapolate(
                    session=sess, states=init_states)

        # Single initial states with multiple samples
        init_states = np.random.normal(3, 1, lat_dim)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            extrapolation2 = diag.run_extrapolate(
                    session=sess, states=init_states)
            grid_out = diag.run_extrapolate(
                    session=sess, states=grid, name="grid")

        assert data[0].shape == (n_examples, time, lat_dim)
        assert data[1].shape == (n_examples, time, obs_dim)
        assert extrapolation1[0].shape == (n_samples, time_forward, lat_dim)
        assert extrapolation1[1].shape == (n_samples, time_forward, obs_dim)
        assert extrapolation2[0].shape == (n_samples, time_forward, lat_dim)
        assert extrapolation2[1].shape == (n_samples, time_forward, obs_dim)
        # Check grid size.
        assert grid_out[0].shape == (grid_size, 2, lat_dim)


def test_mlp_dynamics():
    """Tester for MLPDynamics and MarkovLatentDynamics in general."""
    n_ex, dim, dim_o, time, n_sample = 2, 3, 4, 5, 6
    with tf.Graph().as_default():
        model = MLPDynamics(
            lat_dim=dim, obs_dim=dim_o, time_steps=time,
            transition_layers=[4],
            emission_transform=MultiLayerPerceptron,
            full_covariance=False,
            hidden_units=[5])

        x = np.ones([n_ex, time, dim_o])
        y = np.ones([n_sample, n_ex, time, dim])

        x_ = tf.constant(x)
        y_ = tf.constant(y)
        p = model.log_prob(x_, y_)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(p)
    assert out.shape == (n_sample, n_ex)

    with tf.Graph().as_default():
        model = MLPDynamics(
            lat_dim=dim, obs_dim=dim_o, time_steps=time,
            transition_layers=[4],
            emission_transform=MultiLayerPerceptron,
            full_covariance=False,
            hidden_units=[5])

        x = np.ones([2, time, dim_o])
        y = np.ones([2, time, dim])

        x_ = tf.constant(x)
        y_ = tf.constant(y)
        p = model.log_prob(x_, y_)

        x[1, 1] += 10
        x_ = tf.constant(x)
        p2 = model.log_prob(x_, y_)

        y[0, 1] += 10
        y_ = tf.constant(y)
        p3 = model.log_prob(x_, y_)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out1 = sess.run(p)
            out2 = sess.run(p2)
            out3 = sess.run(p3)

    assert out1[0] == out1[1]
    assert out1[0] == out2[0]
    assert not out2[0] == out2[1]
    assert out2[1] == out3[1]
    assert not out3[0] == out3[1]

if __name__ == "__main__":
    test_kalman_filter(full_covar=False)
    test_kalman_filter(full_covar=True)
    test_mlp_dynamics()
