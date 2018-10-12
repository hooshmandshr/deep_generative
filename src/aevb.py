"""Class for implementing the auto encoding variational bayes learning."""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from model import Model


class AutoEncodingVariationalBayes(object):
    """Class that implements the AEVB algorithm."""

    def __init__(self, data, generative_model, recognition_model, prior=None,
            optimizer=None, n_monte_carlo_samples=1, batch_size=1):
        """
        params:
        -------
        data: numpy.ndarray
        prior: tf.distributions.Distribution
            If None, the generative model is interpreted as full-joint
            distribution over observations and latent codes.
        generative_model: model.Model
        recognition_model: model.Model
        optimizer: tf.train.Optimizer
            If None, by default AdamOptimizer with learning rate 0.001 will be
            used.
        n_monte_carlo_smaples: int
            Number of monte-carlo examples to be used for estimation of ELBO.
        batch_size: int
            Batch size for stochastic estimation of ELBO.
        """
        self.data = data
        self.prior = prior
        self.gen_model = generative_model
        self.rec_model = recognition_model
        self.sample_size = n_monte_carlo_samples
        self.batch_size = batch_size
        # Data properties.
        self.data_size = data.shape[0]
        self.data_dim = data.shape[-1]
        self.opt = optimizer
        if self.opt is None:
            self.opt = tf.train.AdamOptimizer(learning_rate=0.001)
        # Setting up tensorflow Tensors for input and ELBO
        self.batch = tf.placeholder(
                shape=[self.batch_size, self.data_dim],
                dtype=tf.float64, name="input")
        # Set up ELBO computation graph.
        self.elbo = None
        self.get_elbo()
        self.train_op = self.opt.minimize(-self.elbo)
        # Indicate that there is no open session.
        self.sess = None

    def get_elbo(self):
        """Computes the evidence lower bound of the model."""
        if self.elbo is not None:
            return self.elbo

        # Get monte-carlo samples
        mc_samples = self.rec_model.sample(
                n_samples=self.sample_size, y=self.batch)

        likelihood = self.gen_model.log_prob(x=self.batch, y=mc_samples)
        if self.prior is not None:
            likelihood += self.prior.log_prob(mc_samples)
        expected_likelihood = tf.reduce_mean(likelihood, axis=0)
        entropy = self.rec_model.entropy(y=self.batch)
        self.elbo = tf.reduce_sum(expected_likelihood + entropy)
        # Reconstruction of data
        self.recon = tf.reduce_mean(
            self.gen_model.sample(n_samples=self.sample_size, y=mc_samples),
            axis=0)

        return self.elbo

    def make_session(self):
        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def save_session(self):
        pass

    def load_session(self):
        pass

    def close_session(self):
        if self.sess is not None:
            self.sess.close()

    def set_data(self):
        pass

    def train(self, steps, fetch_losses=True):

        self.make_session()
        losses = []
        for i in tqdm(range(steps)):
            idx = i % (self.data_size - self.batch_size)
            if fetch_losses:
                iteration_elbo, _ = self.sess.run(
                        [self.elbo, self.train_op],
                        feed_dict={"input:0": self.data[idx:idx + self.batch_size]})
                losses.append(iteration_elbo)
            else:
                self.sess.run(
                        self.train_op,
                        feed_dict={"input:0": self.data[idx:idx + self.batch_size]})
        return losses

    def get_reconstructions(self, idxs):
        recs = []
        for i in idxs:
            input_ = np.concatenate(
                    [self.data[i:i+1] for j in range(self.batch_size)],
                    axis=0)
            rec = self.sess.run(
                    self.recon,
                    feed_dict={"input:0": input_})
            recs.append(rec[:, 0])
        return recs
