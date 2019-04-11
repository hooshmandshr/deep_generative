"""Class for implementing the auto encoding variational bayes learning."""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from distribution import MultiplicativeNormal
from dynamics import FLDS
from model import Model, ReparameterizedDistribution
from norm_flow_model import NormalizingFlowModel
from transform import MultiLayerPerceptron as MLP


class AutoEncodingVariationalBayes(object):
    """Class that implements the AEVB algorithm."""

    def __init__(self, data, generative_model, recognition_model, prior=None,
            optimizer=None, n_monte_carlo_samples=1, batch_size=1,
            reg_coeff=0.01):
        """
        params:
        -------
        data: numpy.ndarray
            Shape of input is assumed to be (N, ...) where first dimensions
            corresponds to each example and the rest is the shape of each input
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
        # Shape of each example
        self.data_dim = data.shape[1:]
        self.opt = optimizer
        if self.opt is None:
            self.opt = tf.train.AdamOptimizer(learning_rate=0.001)
        # Setting up tensorflow Tensors for input and ELBO
        self.batch = tf.placeholder(
                shape=(self.batch_size,) + self.data_dim,
                dtype=tf.float64, name="input")
        # Set up ELBO computation graph.
        self.elbo = None
        self.get_elbo()
        regularizer = self.gen_model.get_regularizer()
        regularizer += self.rec_model.get_regularizer()
        self.train_op = self.opt.minimize(-self.elbo + reg_coeff * regularizer)
        # Indicate that there is no open session.
        self.sess = None
        # shuffled indices
        self.idx = np.array([]).astype(np.int)

    def get_elbo(self):
        """Computes the evidence lower bound of the model."""
        if self.elbo is not None:
            return self.elbo

        # Get monte-carlo samples
        mc_samples = self.rec_model.sample(
                n_samples=self.sample_size, y=self.batch)
        if isinstance(self.rec_model, NormalizingFlowModel):
            # Sampling method for norm flow models returns both samples
            # and thir log-probabilities.
            mc_samples, mc_samples_log_prob = mc_samples

        likelihood = self.gen_model.log_prob(x=self.batch, y=mc_samples)
        if self.prior is not None:
            likelihood += self.prior.log_prob(mc_samples)
        self.expected_likelihood = tf.reduce_mean(likelihood, axis=0)

        # Computing the entropy of the recognition model based on type. 
        if self.rec_model.has_entropy():
            entropy = self.rec_model.entropy(y=self.batch)
        else:
            # We have to compute MC estimate of the entropy term.
            if not isinstance(self.rec_model, NormalizingFlowModel):
                # We have to compute the log-prob of each sample under the
                # recognition model.
                mc_samples_log_prob = self.rec_model.log_prob(
                        x=mc_samples, y=self.batch)

            entropy = -tf.reduce_mean(mc_samples_log_prob, axis=-1)
        self.entropy = entropy

        self.elbo = tf.reduce_mean(self.expected_likelihood + self.entropy)
        # Reconstruction of data
        self.codes = mc_samples
        self.recon = self.gen_model.sample(n_samples=self.sample_size, y=mc_samples)

        return self.elbo

    def make_session(self):
        """Sets up a new session if none has been initiated."""
        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def save_session(self):
        pass

    def load_session(self):
        pass

    def close_session(self):
        """Closes the existing session."""
        if self.sess is not None:
            self.sess.close()

    def set_data(self):
        pass

    def next_idx(self):
        """Returns the next idx of of the shuffled training data."""
        if len(self.idx) < self.batch_size:
            new_idxs = np.random.permutation(
                    np.random.permutation(self.data_size))
            self.idx = np.append(self.idx, new_idxs)

        out_idx = self.idx[:self.batch_size]
        self.idx = self.idx[self.batch_size:]
        return out_idx

    def train(self, steps, fetch_losses=True):
        """Trains the models for a fixed step size.

        Note: this trainer does not shuffle input data and iterates based on
        the order it receives the data.

        params:
        -------
        steps: int
            Number of iterations in the training.
        fetch_losses: bool
            Whether to retrieve losses for each iteration of the training.

        returns:
        --------
        list of losses per each iteration or emtpy list.
        """
        self.make_session()
        losses = []
        for i in tqdm(range(steps)):
            idx = self.next_idx()
            if fetch_losses:
                iteration_elbo, _ = self.sess.run(
                        [self.elbo, self.train_op],
                        feed_dict={"input:0": self.data[idx]})
                losses.append(iteration_elbo)
            else:
                self.sess.run(
                        self.train_op,
                        feed_dict={"input:0": self.data[idx]})
        return losses

    def check_examples_shape(self, examples):
        """Checks whether input matches the shape of the data."""
        assert isinstance(examples, np.ndarray)
        input_dim = examples.shape[1:]
        if not input_dim == self.data_dim:
            msg = "Input shape {} does not match training data shape {}."
            raise ValueError(msg.format(input_dim, self.data_dim))

    def get_codes(self, examples):
        """Get samples from the recognition model for particular examples.

        Both examples and idxs must not be None and examples has precedence
        over idxs.

        params:
        -------
        examples: numpy.ndarray
            Same shape as the data with the exception of size
            (i.e. first dimension).
        returns:
        numpy.ndarray of codes corresponding to the examples.
        """
        self.check_examples_shape(examples)
        input_size = examples.shape[0]
        codes = []
        for i in tqdm(range(input_size)):
            input_ = np.concatenate(
                    [examples[i:i+1] for j in range(self.batch_size)],
                    axis=0)
            rec = self.sess.run(
                    self.codes,
                    feed_dict={"input:0": input_})
            codes.append(rec[:, 0])
        return np.array(codes)

    def get_reconstructions(self, examples):
        """Get reconstructions for particular examples of the dataset.

        Both examples and idxs must not be None and examples has precedence
        over idxs.

        params:
        -------
        examples: numpy.ndarray
            Same shape as the data with the exception of size
            (i.e. first dimension).

        returns:
        numpy.ndarray of codes corresponding to the examples.
 
        """
        self.check_examples_shape(examples)
        input_size = examples.shape[0]
        recs = []
        for i in tqdm(range(input_size)):
            input_ = np.concatenate(
                    [examples[i:i+1] for j in range(self.batch_size)],
                    axis=0)
            rec = self.sess.run(
                    self.recon,
                    feed_dict={"input:0": input_})
            recs.append(rec[:, 0])
        return np.array(recs)


class FLDSVB(AutoEncodingVariationalBayes):
    """Class for FLDS learning model."""

    def __init__(self, data, lat_dim, nonlinear_transform, poisson=False,
            optimizer=None, n_monte_carlo_samples=1, batch_size=1,
            full_covariance=True, shared_params=True, **kwargs):
        """
        params:
        -------
        data: numpy.ndarray
            Shape of input is assumed to be (N, ...) where first dimensions
            corresponds to each example and the rest is the shape of each input
        nonlinear_transform: tf.Transform type
            Nonlinear transformation from latent space to parameters of the
            observation model.
        poisson: bool
            If False, the model is Gaussian. Otherwise, the emission model is
            Poisson.
        optimizer: tf.train.Optimizer
            If None, by default AdamOptimizer with learning rate 0.001 will be
            used.
        n_monte_carlo_smaples: int
            Number of monte-carlo examples to be used for estimation of ELBO.
        batch_size: int
            Batch size for stochastic estimation of ELBO.
        shared_params: bool
            If True, the recognition and generative models share parameters
            Q, Q1, A. Otherwise, the recognition model has it's own LDS
            parameters.
        **kwargs:
            Arguments for neural network function corresponding to the
            generative function.
        """
        self.lat_dim = lat_dim
        _, self.time, self.obs_dim = data.shape
        self.poisson = poisson
        gen_model = FLDS(
                lat_dim=lat_dim, obs_dim=self.obs_dim, time_steps=self.time,
                init_transition_matrix_bias=np.append(
                    np.eye(self.lat_dim), np.zeros([1, self.lat_dim]), axis=0),
                full_covariance=full_covariance,
                poisson=self.poisson,
                nonlinear_transform=nonlinear_transform, **kwargs)

        # (Q1, Q, A)
        lds_params = None
        if shared_params:
            # Transpose is necessary because our linear transformation class
            # does x^TA therefore a transpose is needed to go to canonical form
            # of vector transformation.
            lds_params = gen_model.get_linear_dynamics_params(transpose_a=True)

        recon_model = ReparameterizedDistribution(
                in_dim=(self.time, self.obs_dim),
                out_dim=(self.time, self.lat_dim),
                distribution=MultiplicativeNormal,
                mult_normal_vars=lds_params,
                transform=MLP, hidden_units=[self.obs_dim * 2, self.obs_dim])

        super(FLDSVB, self).__init__(
                data=data, generative_model=gen_model,
                recognition_model=recon_model,
                n_monte_carlo_samples=n_monte_carlo_samples,
                batch_size=batch_size, optimizer=optimizer)

