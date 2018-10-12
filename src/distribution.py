"""Provides extra dsitributions beside available tensorflow distributions."""


import numpy as np
import tensorflow as tf

from utils.block_matrix import *


class LogitNormal(tf.distributions.Normal):
    """Class that extends Normal to a logit-normal distribution."""

    def __init__(self, loc, scale):
        super(LogitNormal, self).__init__(loc=loc, scale=scale)

    def log_prob(self, value, name='log_prob'):
        logit_value = tf.log(value / (1 - value))
        log_prob = super(LogitNormal, self).log_prob(logit_value, name=name)
        return log_prob - tf.log(value) - tf.log(1 - value)

    def sample(self, sample_shape=(), seed=None, name='sample'):
            sample = super(LogitNormal, self).sample(
                sample_shape=sample_shape, seed=seed, name=name)
            return tf.sigmoid(sample)


class LogitNormalDiag(LogitNormal):
    """Class that extends LogitNormal to logit multi-variate diagonal.""" 

    def __init__(self, loc, scale_diag):
        super(LogitNormalDiag, self).__init__(loc=loc, scale=scale_diag)

    def log_prob(self, value, name='log_prob'):
        """Sum of log-prob of independent logit-normal in the last axis."""
        log_prob = super(LogitNormalDiag, self).log_prob(
                value=value, name=name)
        return tf.reduce_sum(log_prob, axis=-1)


class MultiPoisson(tf.contrib.distributions.Poisson):
    """Class that extends poisson to independent multivarate Poissons."""

    def log_prob(self, value, name='log_prob'):
        """Sum of log-prob of independent logit-normal in the last axis."""
        log_prob = super(MultiPoisson, self).log_prob(
                value=value, name=name)
        return tf.reduce_sum(log_prob, axis=-1)


class BlockTriDiagonalNormal(tf.distributions.Distribution):
    """Class for a multi-variate normal with block-tri-diagonal covariance."""

    def __init__(self, loc, inv_cov, inv_cov_chol_factor=None):
        """Sets up the variables for the block-tri-diagonal covariance normal.

        params:
        -------
        loc: tf.Tensor:
            Mean of the multi-variate normal distribution.
        inv_cov: BlockTriDiagonalMatrix
            Inverse of the covariance of the multi-variate normal distribution.
            The shape must be compatible with loc.
        inv_cov_chol_factor: None or BlockBiDiagonalMatrix
            Lower triangular Cholesky factor of the inverse covariance. If
            None, it will be computed in the constructor.
        """
        # Noise variable that will be re-parameterized to produce
        # samples from the desired distribution with log density.
        if not isinstance(inv_cov, BlockTriDiagonalMatrix):
            raise ValueError(
                    "inv_cov should be of type BlockTriDiagonalMatrix.")

        self.base_dist = tf.distributions.Normal(
                loc=tf.zeros(loc.shape), scale=tf.ones(loc.shape))
        self.loc = loc
        self.inv_cov = inv_cov
        self.chol_factor = inv_cov_chol_factor
        if inv_cov_chol_factor is None:
            self.chol_factor = self.inv_cov.cholesky()
        self.chol_factor_transpose = self.chol_factor.transpose(in_place=False)

    def sample(self, n_samples):
        """Samples n_samples times from the mutli-variate normal."""
        samples = self.base_dist.sample(n_samples)
        return self.loc + self.chol_factor_transpose.solve(samples)

    def entropy(self):
        """Closed form entropy using det of cholesky factor of inv-cov."""
        tot_dim = self.chol_factor.num_block * self.chol_factor.block_dim
        const = (tf.log(2. * np.pi) + 1.) * tot_dim / 2.
        return - tf.reduce_sum(tf.reduce_sum(
                tf.log(self.chol_factor.get_diag_part()), axis=-1), axis=-1) + const


class MultiplicativeNormal(BlockTriDiagonalNormal):
    """Parameterization of time-correlated approximate posterior for LDS.

    This parameterization is the result of multiplication of T Guassian
    distribution of dimension D.
    """

    def __init__(self, q_init, q_matrix, a_matrix, c_matrix, m_matrix): 
        """Sets up the variables for the block-tri-diagonal covariance normal.

        params:
        -------
        q_init: tf.Tensor with shape (..., D, D)
            Covariance of the prior gaussian.
        q_matrix: tf.Tensor with shape (..., D, D)
            Covariance of the transition Gaussian distribution. 
        a_matrix: tf.Tensor with shape (..., D, D)
            Linear transformation matrix for the transition function.
        c_matrix: tf.Tensor with shape (..., T, D, D)
            Block-diagonal matrix.
        m_matrix: tf.Tensor with shape (..., T, D)
        """
        # Dimension of space.
        self.dim = m_matrix.shape[-1].value
        # Number of time steps.
        self.time = m_matrix.shape[-2].value
        self.outer_dim = q_init.shape[:-2]
        if not(q_init.shape[-2:] == (self.dim, self.dim) and\
                q_matrix.shape[-2:] == (self.dim, self.dim) and\
                a_matrix.shape[-2:] == (self.dim, self.dim) and\
                c_matrix.shape[-3:] == (self.time, self.dim, self.dim)):
            msg = "Dimensions and time-steps not consistent across variables."
            raise ValueError(msg)
        elif not(q_matrix.shape[:-2] == self.outer_dim and\
                a_matrix.shape[:-2]  == self.outer_dim and\
                c_matrix.shape[:-3] == self.outer_dim and\
                m_matrix.shape[:-2] == self.outer_dim):
            msg = "Outer dimension of variables do not match."
            raise ValueError(msg)
        # Construct the block-tri-diagonal matrix that expresses the inverse
        # covariance of the resulting Gaussian.
        q_1_inv = tf.linalg.inv(q_init)
        q_inv = tf.linalg.inv(q_matrix)
        aqat_1 = tf.matmul(
                tf.matmul(a_matrix, q_1_inv), a_matrix, transpose_b=True)
        aqat = tf.matmul(
                tf.matmul(a_matrix, q_inv), a_matrix, transpose_b=True)
        # construct the diagonal and off-diagnoal blocks of the inverse
        # covariance.
        diag_blocks = [q_1_inv, aqat_1 + q_inv]
        offdiag_blocks = [-tf.matmul(a_matrix, q_1_inv)]
        for i in range(2, self.time):
            diag_blocks.append(aqat + q_inv)
            offdiag_blocks.append(-tf.matmul(a_matrix, q_inv))
        # Expand and concat the list into a tensor.
        diag_blocks = tf.concat(
                [tf.expand_dims(t, -3) for t in diag_blocks], axis=-3)
        offdiag_blocks = tf.concat(
                [tf.expand_dims(t, -3) for t in offdiag_blocks], axis=-3)
        diag_blocks += c_matrix

        inv_cov = BlockTriDiagonalMatrix(
                diag_block=diag_blocks, offdiag_block=offdiag_blocks)
        # Compute the mean of the Gaussian.
        mult_res = tf.expand_dims(tf.squeeze(
                tf.matmul(c_matrix, tf.expand_dims(m_matrix, -1)), axis=-1),
                axis=-3)
        # Cholesky factor of the inverse covariance.
        inv_cov_chol_factor = inv_cov.cholesky()

        mean = tf.squeeze(inv_cov_chol_factor.solve(
                inv_cov_chol_factor.solve(mult_res), transpose=True), axis=-3)

        super(MultiplicativeNormal, self).__init__(loc=mean, inv_cov=inv_cov)
        # Now we have a block-tri-diagonal normal that we can sample from
        # and compute entropy of.

