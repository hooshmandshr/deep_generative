"""Class for implementing the auto encoding variational bayes learning."""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from distribution import MultiplicativeNormal, StateSpaceNormalDiag
from dynamics import FLDS, DeepKalmanFilter, DeepKalmanDynamics, MLPDynamics, KalmanFilter
from model import Model, ReparameterizedDistribution
from norm_flow import TimeAutoRegressivePlanarFlow, MultiLayerKalmanFlow
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
    """Class for FLDS Auto-Encoding Variational Bayes."""

    def __init__(self, data, lat_dim, emission_transform,
            emission_transform_params, recognition_transform,
            recognition_transform_params, poisson=False, optimizer=None,
            n_monte_carlo_samples=1, batch_size=1, full_covariance=True,
            shared_params=True):
        """
        params:
        -------
        data: numpy.ndarray
            Shape of input is assumed to be (N, ...) where first dimensions
            corresponds to each example and the rest is the shape of each input
        lat_dim: int
            Dimensionality of the latent state space.
        emission_transform: tf.Transform type
            Nonlinear transformation from latent space to parameters of the
            observation model.
        emission_transform_params: dict
            Arguments to be passed to the emission transform.
        recognition_transform: tf.Transform type
            Nonlinear transformation from observations to parameters of the
            inference/recognition network.
        recognition_transform_params: dict
            Arguments to be passed to the recognition transform.
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
        """
        self.lat_dim = lat_dim
        _, self.time, self.obs_dim = data.shape
        self.poisson = poisson
        gen_model = FLDS(
                lat_dim=lat_dim, obs_dim=self.obs_dim, time_steps=self.time,
                full_covariance=full_covariance,
                poisson=self.poisson,
                nonlinear_transform=emission_transform,
                **emission_transform_params)

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
                transform=recognition_transform,
                **recognition_transform_params)

        super(FLDSVB, self).__init__(
                data=data, generative_model=gen_model,
                recognition_model=recon_model,
                n_monte_carlo_samples=n_monte_carlo_samples,
                batch_size=batch_size, optimizer=optimizer)


class PFLDSVB(FLDSVB):
    """Implements pflds class from count data."""

    def __init__(self, data, lat_dim, hdim=60, optimizer=None,
            n_monte_carlo_samples=1, batch_size=1, full_covariance=True,
            shared_params=True):
        """
        params:
        -------
        data: numpy.ndarray
            Shape of input is assumed to be (N, ...) where first dimensions
            corresponds to each example and the rest is the shape of each input
        lat_dim: int
            Dimensionality of the latent state space.
        hdim: int
            Number of hidden units per layer of the MLP networks in the
            generative model and inference model.
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
        """

        # Both generative models non-linearity and recognition networks models
        # are MLP.
        gen_mlp_params = {
                "activation": tf.nn.tanh, "output_activation": tf.exp,
                "hidden_units": [hdim, hdim]}
        rec_mlp_params = {
                "activation": tf.nn.tanh, "hidden_units": [hdim, hdim]}
        super(PFLDSVB, self).__init__(
                data=data, lat_dim=lat_dim,
                emission_transform=MLP,
                emission_transform_params=gen_mlp_params,
                recognition_transform=MLP,
                recognition_transform_params=rec_mlp_params, poisson=True,
                optimizer=optimizer, n_monte_carlo_samples=1, batch_size=1,
                full_covariance=full_covariance, shared_params=True)


class DeepKalmanVB(AutoEncodingVariationalBayes):
    """Class for Deep Kalman Filter Auto-Encoding Variational Bayes."""

    def __init__(self, data, lat_dim, transition_hidden_dim,
            emission_layers, recon_hidden_dim,
            mean_field=False, backward=True,
            optimizer=None, n_monte_carlo_samples=1, batch_size=1):
        """
        params:
        -------
        data: numpy.ndarray
            Shape of input is assumed to be (N, ...) where first dimensions
            corresponds to each example and the rest is the shape of each input
        lat_dim: int
        transition_hidden_dim: int
            Dimensionality of the hidden unit of the gated transition transform
            in the generative model.
        emission_layers: list of int
            Hidden layers of the MLP emission transform.
        recon_hidden_dim: int
            Dimensionality of the hidden state of the RNNs in the recognition
            model.
        mean_field: bool
        backward: bool
        optimizer: tf.train.Optimizer
            If None, by default AdamOptimizer with learning rate 0.001 will be
            used.
        n_monte_carlo_smaples: int
            Number of monte-carlo examples to be used for estimation of ELBO.
        batch_size: int
            Batch size for stochastic estimation of ELBO.
        """
        self.lat_dim = lat_dim
        _, self.time, self.obs_dim = data.shape
        self.gen_hid_dim = transition_hidden_dim
        self.rec_hid_dim = recon_hidden_dim

        gen_model = DeepKalmanDynamics(
            lat_dim=self.lat_dim, obs_dim=self.obs_dim, time_steps=self.time,
            transition_units=self.gen_hid_dim, emission_layers=emission_layers)

        recon_model = DeepKalmanFilter(
            in_dim=self.obs_dim, out_dim=self.lat_dim, time_steps=self.time,
            rnn_hdim=self.rec_hid_dim, mean_field=mean_field, backward=backward)

        super(DeepKalmanVB, self).__init__(
                data=data, generative_model=gen_model,
                recognition_model=recon_model,
                n_monte_carlo_samples=n_monte_carlo_samples,
                batch_size=batch_size, optimizer=optimizer)


class FilteringNormalizingFlowVB(AutoEncodingVariationalBayes):
    """Class for Filtering Normalizing Flow Auto-Encoding Variational Bayes."""

    def __init__(self, data, lat_dim, transition_layers, emission_layers,
            recognition_layers, n_flow_layers, residual=False, backward=False,
            poisson=False, condition_shared_units=0,
            optimizer=None, n_monte_carlo_samples=1, batch_size=1,
            full_covariance=True, order=1):
        """
        params:
        -------
        data: numpy.ndarray
            Shape of input is assumed to be (N, ...) where first dimensions
            corresponds to each example and the rest is the shape of each input
        transition_layers: list of int
        emission_layers: list of int
            If [], the transition is linear.
        recognition_layers: list of int
        n_flow_layers: int
        residual: bool
        backward: bool
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
        """
        self.lat_dim = lat_dim
        _, self.time, self.obs_dim = data.shape
        self.poisson = poisson

        out_activation = None
        if self.poisson:
            out_activation = tf.exp

        gen_model = None
        if len(transition_layers) == 0:
            # Linear model.
            gen_model = FLDS(
                    lat_dim=lat_dim, obs_dim=self.obs_dim,
                    time_steps=self.time, full_covariance=full_covariance,
                    order=order, poisson=self.poisson,
                    nonlinear_transform=MLP, hidden_units=emission_layers)
        else:
            gen_model = MLPDynamics(
                    lat_dim=self.lat_dim, obs_dim=self.obs_dim,
                    time_steps=self.time, transition_layers=transition_layers,
                    residual=residual, poisson=self.poisson,
                    full_covariance=full_covariance, emission_transform=MLP,
                    order=order,
                    hidden_units=emission_layers,
                    output_activation=out_activation)

        #TODO: have a single recognition network for MF params and NF params.

        extra_dim = condition_shared_units
        base_model = ReparameterizedDistribution(
            out_dim=(self.time, self.lat_dim), in_dim=(self.time, self.obs_dim),
            distribution=StateSpaceNormalDiag, transform=MLP,
            reparam_scale=True, extra_dim=extra_dim,
            hidden_units=recognition_layers)

        recon_model = NormalizingFlowModel(
            in_dim=(self.time, self.obs_dim), base_model=base_model,
            norm_flow_type=TimeAutoRegressivePlanarFlow,
            norm_flow_params={"num_layer": n_flow_layers, "backward": backward},
            transform_type=MLP,
            transform_params={"hidden_units": recognition_layers}, share=extra_dim)

        super(FilteringNormalizingFlowVB, self).__init__(
                data=data, generative_model=gen_model,
                recognition_model=recon_model,
                n_monte_carlo_samples=n_monte_carlo_samples,
                batch_size=batch_size, optimizer=optimizer)


class KalmanNormalizingFlowVB(AutoEncodingVariationalBayes):
    """Class for Filtering Normalizing Flow Auto-Encoding Variational Bayes."""

    def __init__(self, data, lat_dim, transition_layers, emission_layers,
            recognition_layers, n_flow_layers, residual=False,
            poisson=False, condition_shared_units=0,
            optimizer=None, n_monte_carlo_samples=1, batch_size=1,
            full_covariance=True, order=1):
        """
        params:
        -------
        data: numpy.ndarray
            Shape of input is assumed to be (N, ...) where first dimensions
            corresponds to each example and the rest is the shape of each input
        transition_layers: list of int
        emission_layers: list of int
            If [], the transition is linear.
        recognition_layers: list of int
        n_flow_layers: int
        residual: bool
        backward: bool
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
        """
        self.lat_dim = lat_dim
        _, self.time, self.obs_dim = data.shape
        self.poisson = poisson

        gen_model = None
        if len(transition_layers) == 0:
            # Linear model.
            gen_model = FLDS(
                    lat_dim=lat_dim, obs_dim=self.obs_dim,
                    time_steps=self.time, full_covariance=full_covariance,
                    order=order, poisson=self.poisson,
                    nonlinear_transform=MLP, hidden_units=emission_layers)
        else:
            gen_model = MLPDynamics(
                    lat_dim=self.lat_dim, obs_dim=self.obs_dim,
                    time_steps=self.time, transition_layers=transition_layers,
                    residual=residual, poisson=self.poisson,
                    full_covariance=full_covariance, emission_transform=MLP,
                    hidden_units=emission_layers)

        extra_dim = condition_shared_units
        base_model = ReparameterizedDistribution(
            out_dim=(self.time, self.lat_dim), in_dim=(self.time, self.obs_dim),
            distribution=StateSpaceNormalDiag, transform=MLP,
            reparam_scale=True, extra_dim=extra_dim,
            hidden_units=recognition_layers)

        recon_model = NormalizingFlowModel(
            in_dim=(self.time, self.obs_dim), base_model=base_model,
            norm_flow_type=MultiLayerKalmanFlow,
            norm_flow_params={'n_layer': n_flow_layers},
            transform_type=MLP,
            transform_params={"hidden_units": recognition_layers}, share=extra_dim)

        super(KalmanNormalizingFlowVB, self).__init__(
                data=data, generative_model=gen_model,
                recognition_model=recon_model,
                n_monte_carlo_samples=n_monte_carlo_samples,
                batch_size=batch_size, optimizer=optimizer)
