"""Testing norm_flow.py classes."""


import numpy as np
import tensorflow as tf

from distribution import StateSpaceNormalDiag
from model import ReparameterizedDistribution
from transform import LinearTransform, MultiLayerPerceptron


def test_reparam_gaussian_diag_covar():
    """Tester for correctness of linear transformation."""

    in_dim = 3
    out_dim = 2
    n_sample = 1000

    tot_dist = 2
    input_ = np.random.rand(tot_dist, in_dim)
    dist = tf.contrib.distributions.MultivariateNormalDiag

    with tf.Graph().as_default():
        input_tensor = tf.constant(input_)
        d_1 = ReparameterizedDistribution(
                out_dim=out_dim, in_dim=in_dim,
                distribution=dist, transform=LinearTransform,
                reparam_scale=True)
        d_2 = ReparameterizedDistribution(
                out_dim=out_dim, in_dim=in_dim,
                distribution=dist, transform=LinearTransform,
                reparam_scale=False)
        s_1 = d_1.sample(n_samples=n_sample, y=input_tensor)
        # Repeat for testing sampling multiple times.
        s_1 = d_1.sample(n_samples=n_sample, y=input_tensor)
        s_2 = d_2.sample(n_samples=n_sample, y=input_tensor)
        s_2 = d_2.sample(n_samples=n_sample, y=input_tensor)

        cov_2 = d_2.get_distribution(y=input_tensor).covariance()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf_res = sess.run([s_1, s_2])

    expected_shape = (n_sample, tot_dist, out_dim)
    print("Testing the shape of the samples.")
    for i in range(2):
        assert tf_res[i].shape == expected_shape, "sample shape not correct."


def test_reparam_gaussian_full_covar():
    """Tester for correctness of linear transformation."""

    in_dim = 3
    out_dim = 2
    n_sample = 1000

    tot_dist = 2
    input_ = np.random.rand(tot_dist, in_dim)
    dist = tf.contrib.distributions.MultivariateNormalTriL

    with tf.Graph().as_default():
        input_tensor = tf.constant(input_)
        d_1 = ReparameterizedDistribution(
                out_dim=out_dim, in_dim=in_dim,
                distribution=dist, transform=LinearTransform,
                reparam_scale=True)
        d_2 = ReparameterizedDistribution(
                out_dim=out_dim, in_dim=in_dim,
                distribution=dist, transform=LinearTransform,
                reparam_scale=False)
        s_1 = d_1.sample(n_samples=n_sample, y=input_tensor)
        # Repeat for testing sampling multiple times.
        s_1 = d_1.sample(n_samples=n_sample, y=input_tensor)
        s_2 = d_2.sample(n_samples=n_sample, y=input_tensor)
        s_2 = d_2.sample(n_samples=n_sample, y=input_tensor)

        cov_2 = d_2.get_distribution(y=input_tensor).covariance()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf_res = sess.run([s_1, s_2])
            tf_cov = sess.run([cov_2])

    expected_shape = (n_sample, tot_dist, out_dim)
    print("Testing the shape of the samples.")
    for i in range(2):
        assert tf_res[i].shape == expected_shape, "sample shape not correct."


def test_state_space_normal_diag():
    """Tester for correctness of StateSpaceNormalDiag transformation."""

    time = 5
    in_dim = 3
    out_dim = 2
    n_sample = 100

    tot_dist = 1
    input_ = np.random.rand(tot_dist, time, in_dim)

    with tf.Graph().as_default():
        input_tensor = tf.constant(input_)
        d_1 = ReparameterizedDistribution(
                out_dim=(time, out_dim), in_dim=(time, in_dim),
                distribution=StateSpaceNormalDiag,
                transform=MultiLayerPerceptron,
                reparam_scale=True, hidden_units=[20, 20])
        s_1 = d_1.sample(n_samples=n_sample, y=input_tensor)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf_res = sess.run(s_1)

    expected_shape = (n_sample, tot_dist, time, out_dim)
    print("Testing the shape of the StateSpaceNormalDiag samples.")
    assert tf_res.shape == expected_shape, "sample shape not correct."


if __name__ == "__main__":
    test_reparam_gaussian_diag_covar()
    test_reparam_gaussian_full_covar()
    test_state_space_normal_diag()
