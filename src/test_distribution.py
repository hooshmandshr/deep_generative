"""Testing correctness of the implemented distributions."""

import numpy as np
import scipy.stats
import tensorflow as tf

from distribution import *
from utils.test_block_matrix import dense_matrix

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
    dtype = np.float32
    # Number of samples.
    def invable_mat(num=1):
        if num > 1:
            a = np.tril(np.random.normal(0, 1, [num, dim, dim]).astype(dtype))
            return np.matmul(a, a.transpose([0, 2, 1]))
        a = np.tril(np.random.normal(0, 1, [dim, dim]).astype(dtype))
        return np.matmul(a, a.T)

    q1 = invable_mat()
    q = invable_mat()
    a = (np.random.normal(0, 1, [dim, dim]) + np.eye(dim)).astype(dtype)
    c = invable_mat(num=time)
    m = np.random.normal(0, 1, [1, dim * time]).astype(dtype)
    # Compute intermediate results.
    diag = np.array([np.eye(dim, dtype=dtype) for i in range(time)])
    offdiag = np.array([-a for i in range(time -1)])
    dense_eye_minus_a = dense_matrix(diag, offdiag)
    dense_q = dense_matrix(np.array([q1] + [q for i in range(time - 1)]))
    dense_c = dense_matrix(c)
    cov_inv = np.matmul(
            np.matmul(dense_eye_minus_a, np.linalg.inv(dense_q)),
            dense_eye_minus_a.T) + dense_c

    cov = np.linalg.inv(cov_inv)
    mu = np.matmul(np.matmul(cov, dense_c), m.T)

    with tf.Graph().as_default():
        dist = MultiplicativeNormal(
                q_init=tf.constant(q1),
                q_matrix=tf.constant(q),
                a_matrix=tf.constant(a),
                c_matrix=tf.constant(c),
                m_matrix=tf.constant(m.reshape([time, dim])))
        sigma = [dist.inv_cov.diag_block, dist.inv_cov.offdiag_block]
        loc = dist.loc
        with tf.Session() as sess:
           tf_sigma = sess.run(sigma)
           tf_mu = sess.run(loc)

    print "Cecking correctness of mean and covariance computaiotn."
    err = 1e-03
    dense_tf_sigma = dense_matrix(tf_sigma[0], tf_sigma[1], tridiagonal=True)
    assert np.allclose(dense_tf_sigma, cov_inv, err), "Sigma inverse Incorrect."
    assert np.allclose(mu.ravel(), tf_mu.ravel(), err), "Mu computation incorrect."


if __name__ == "__main__":
    test_block_diagonal_normal()
    test_multiplicative_normal()
 
