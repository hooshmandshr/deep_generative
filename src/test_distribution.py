"""Testing correctness of the implemented distributions."""

import numpy as np
import scipy.stats
import tensorflow as tf

from distribution import *
from utils.test_block_matrix import dense_matrix


def positive_definite_matrix(shape):
    """Gets a positive definite matrix of given shape."""
    mat = np.tril(np.random.normal(0, 1, shape))
    # Transpose last two dimensions.
    t_perm = [i for i in range(len(shape))]
    t_perm = t_perm[:-2] + t_perm[-2:][::-1]
    return np.matmul(mat, mat.transpose(t_perm))

def test_block_diagonal_normal():
    """Tests the correctness of BlockTriDiagonalNormal distribution."""
    time, dim = 3, 2
    dtype = np.float32
    loc_ = np.random.rand(time * dim).astype(dtype)
    # diagonal part of the inverse covariance.
    inv_cov_d = np.random.rand(time, dim, dim)
    inv_cov_d += inv_cov_d.transpose([0, 2, 1])
    inv_cov_d /= 2.
    inv_cov_d += np.eye(dim)[None, :, :] * dim * time
    inv_cov_d = inv_cov_d.astype(dtype)
    # Off-diagonal block part of the inverse-covariance.
    inv_cov_o = np.random.rand(time - 1, dim, dim).astype(dtype)
    # Get the dense form of the matrix.
    dense_inv_cov = dense_matrix(inv_cov_d, inv_cov_o, tridiagonal=True)

    # Number of samples.
    n_ex = 10000
    with tf.Graph().as_default():
        btm = BlockTriDiagonalMatrix(
            diag_block=tf.constant(inv_cov_d),
            offdiag_block=tf.constant(inv_cov_o))
        dist = BlockTriDiagonalNormal(loc=loc_, inv_cov=btm)
        entropy_tensor = dist.entropy()
        samples_tensor = dist.sample(n_ex)
        with tf.Session() as sess:
            samples = sess.run(samples_tensor)
            entropy = sess.run(entropy_tensor)
    print "Checking correctness of entropy computation."
    cov = np.linalg.inv(dense_inv_cov)
    assert np.allclose(
            scipy.stats.multivariate_normal(loc_, cov).entropy(),
            entropy), "Correctness of entorpy."

    print "Checking the shape of samples"
    assert samples.shape == (n_ex, time * dim), "Shape of samples."

def test_multiplicative_normal():
    """Tests the correctness of BlockTriDiagonalNormal distribution."""
    time, dim = 4, 3
    # Number of samples.
    q1 = positive_definite_matrix([dim, dim])
    q = positive_definite_matrix([dim, dim])
    a = np.random.normal(0, 1, [dim, dim])
    # c is the precision matrix of the potential distributions.
    c_inv = positive_definite_matrix([time, dim, dim])
    m = np.random.normal(0, 1, [time, dim])
    # Compute intermediate results.
    diag = np.array([np.eye(dim) for i in range(time)])
    offdiag = np.array([-a for i in range(time - 1)])
    dense_eye_minus_a = dense_matrix(diag, offdiag)
    dense_q = dense_matrix(np.array([q1] + [q for i in range(time - 1)]))
    dense_c_inv = dense_matrix(c_inv)
    cov_inv = np.matmul(
            np.matmul(dense_eye_minus_a.T, np.linalg.inv(dense_q)),
            dense_eye_minus_a) + dense_c_inv
    cov = np.linalg.inv(cov_inv)

    mu = np.matmul(np.matmul(cov, dense_c_inv), m.reshape([dim * time, 1]))

    with tf.Graph().as_default():
        dist = MultiplicativeNormal(
                q_init=tf.constant(q1),
                q_matrix=tf.constant(q),
                a_matrix=tf.constant(a),
                c_matrix=tf.constant(c_inv),
                m_matrix=tf.constant(m))
        sigma = [dist.inv_cov.diag_block, dist.inv_cov.offdiag_block]
        loc = dist.loc
        entropy = dist.entropy()
        with tf.Session() as sess:
            tf_sigma = sess.run(sigma)
            tf_mu = sess.run(loc)
            entropy_result = sess.run(entropy)

    print "Checking correctness of mean and covariance computaiotn."
    dense_tf_sigma = dense_matrix(tf_sigma[0], tf_sigma[1], tridiagonal=True)
    assert np.allclose(dense_tf_sigma, cov_inv), "Sigma inverse Incorrect."
    assert np.allclose(mu.ravel(), tf_mu.ravel()), "Mu computation incorrect."
    assert np.allclose(
            scipy.stats.multivariate_normal(m.ravel(), cov).entropy(),
            entropy_result), "Entropy of Gaussian incorrect."

    print "Cheking broadcasting samples with multiple distribution parameters."
    n_dist, n_samples = 5, 10
    # Number of samples.
    q1 = positive_definite_matrix([n_dist, dim, dim])
    q = positive_definite_matrix([n_dist, dim, dim])
    a = np.random.normal(0, 1, [n_dist, dim, dim])
    c_inv = positive_definite_matrix([n_dist, time, dim, dim])
    m = np.random.normal(0, 1, [n_dist, time, dim])
 
    with tf.Graph().as_default():
        mlt_gaussian = MultiplicativeNormal(
                q_init=tf.constant(q1),
                q_matrix=tf.constant(q),
                a_matrix=tf.constant(a),
                c_matrix=tf.constant(c_inv),
                m_matrix=tf.constant(m))
        samples = mlt_gaussian.sample(n_samples)
        entropy = mlt_gaussian.entropy()
        with tf.Session() as sess:
            sample_results = sess.run(samples)
            entropy_result = sess.run(entropy)
    assert sample_results.shape == (n_samples, n_dist, time, dim)
    assert entropy_result.shape == (n_dist,)

if __name__ == "__main__":
    test_block_diagonal_normal()
    test_multiplicative_normal()
 
