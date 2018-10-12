"""Testing norm_flow.py classes."""


import numpy as np
import tensorflow as tf

from model import ReparameterizedDistribution
from transform import LinearTransform


def test_reparam_gaussian_full_covar():
    """Tester for correctness of linear transformation."""

    in_dim = 3
    out_dim = 2
    n_sample = 1000

    tot_dist = 2
    input_ = np.random.rand(tot_dist, in_dim)
    dist = tf.contrib.distributions.MultivariateNormalTriL

    with tf.Graph().as_default():
        d_1 = ReparameterizedDistribution(
                out_dim=out_dim, in_dim=in_dim,
                distribution=dist, transform=LinearTransform,
                reparam_scale=True)
        d_2 = ReparameterizedDistribution(
                out_dim=out_dim, in_dim=in_dim,
                distribution=dist, transform=LinearTransform,
                reparam_scale=False)
        d_3 = ReparameterizedDistribution(
                out_dim=out_dim, in_dim=in_dim,
                distribution=dist, transform=LinearTransform,
                reparam_scale=tf.constant(np.eye(2)))
        input_tensor = tf.constant(input_)
        s_1 = d_1.sample(n_samples=n_sample, y=input_tensor)
        s_2 = d_2.sample(n_samples=n_sample, y=input_tensor)
        s_3 = d_3.sample(n_samples=n_sample, y=input_tensor)

        cov_2 = d_2.get_distribution(y=input_tensor).covariance()
        cov_3 = d_3.get_distribution(y=input_tensor).covariance()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf_res = sess.run([s_1, s_2, s_3])
            tf_cov = sess.run([cov_2, cov_3])

    expected_shape = (n_sample, tot_dist, out_dim)
    print "Testing the shape of the samples."
    for i in range(3):
        assert tf_res[i].shape == expected_shape, "sample shape not correct."
    print "Testing that the covariance of the distributions are all the same."
    assert np.allclose(np.array(np.eye(2)), tf_cov[1])
    assert np.allclose(tf_cov[0][0], tf_cov[0][1])

if __name__ == "__main__":
    test_reparam_gaussian_full_covar()

