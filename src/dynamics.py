"""Class for implementing dynamical systems models according to Model design.

The goal of this library is to provide necessary dynamical systems models
to easily sample from, compute density, etc.
"""

import numpy as np
import tensorflow as tf

from model import Model, ReparameterizedDistribution
from transform import LinearTransform


class MarkovDynamics(Model):

    def __init__(self, init_model, transition_model, time_steps, order=1):
        """Sets up the necessary networks for the markov dynamics.

        params:
        -------
        init_state: model.Model with in_dim=0 or tf.distributions.Distribution
        transition_model: model.Model
            in_dim and out_dim of this model should be the same.
        time_steps: int
            How many time steps exists in the process.
        order: int
            Order of the markov process that expresses our dynamics.
        """
        self.time_steps = time_steps
        # In effect, the prior of the dynamical system.
        self.init_model = init_model
        self.transition_model = transition_model
        # dimension of each state
        self.state_dim = self.transition_model.in_dim
        if isinstance(self.init_model, Model):
            if self.state_dim == self.init_model.out_dim:
                msg = "Dimensions of prior and transition model mismatch: {}, {}."
                raise ValueError(msg.format(
                    self.state_dim, self.init_model.out_dim))

        super(MarkovDynamics, self).__init__(out_dim=self.state_dim * time_steps)

    def log_prob(self, x):
        """Log probability of the dynamics model p(x) = p(x1, x2, ...).

        params:
        -------
        x: tf.Tensor
            Shape of x should match the shape of model. In other words,
            (?, ..., T, D)
        """
        log_prob = self.init_model.log_prob(x[:, 0, :])
        for i in range(1, self.time_steps):
            log_prob += self.transition_model.log_prob(
                    x=x[:, i, :], y=x[:, i - 1, :])
        return log_prob

    def sample(self, n_samples, time_steps=None):
        """Samples from the markov dynamics model.

        params:
        -------
        n_samples: int
        time_steps: int or None
        If None the numbe of states in the samples will be equal to that
        of the model.

        Returns:
        --------
        tf.Tensor
        """
        if time_steps is None:
            time_steps = self.time_steps
        # Sample from the prior distribution for initial state.
        states = []
        states.append(self.init_model.sample((1, n_samples)))
        for i in range(time_steps - 1):
            states.append(self.transition_model.sample(
                y=states[-1][0], n_samples=1))
        return tf.transpose(tf.concat(states, axis=0), perm=[1, 0, 2])


class MarkovLatentDynamics(MarkovDynamics):
    """Class for expressing Laten Markov dynamics."""

    def __init__(self, init_model, transition_model, emission_model,
            time_steps, order=1):
        """Sets up the necessary networks for the markov dynamics.

        params:
        -------
        init_state: model.Model with in_dim=0 or tf.distributions.Distribution
        transition_model: model.Model
            in_dim and out_dim of this model should be the same.
        time_steps: int
            How many time steps exists in the process.
        order: int
            Order of the markov process that expresses our dynamics.
        """
        super(MarkovLatentDynamics, self).__init__(
                init_model=init_model, transition_model=transition_model,
                time_steps=time_steps, order=order)
        self.emission_model = emission_model
        if not self.emission_model.in_dim == self.state_dim:
            msg = "Dimensions of transition and emission models: {}, {}."
            raise ValueError(msg.format(
                self.emission_model.in_dim, self.state_dim)) 
        # Dimension of the observation space.
        self.obs_dim = self.emission_model.out_dim

    def log_prob(self, x, y):
        """Samples from the full-joint model p(x, y).

        x and y should both have the same inner dimension. If not, broadcasting
        is done to make the shapes match. For instance, if input shapes are
        (M, N, T, D_x) and (N, T, D_Y), y is duplicated M times to correspond
        to the shape of x.

        params:
        -------
        x: tf.Tensor with shape (N, T, D_x)
            Tensor corresponding to the paths in observations space.
        y: tf.Tensor with shape (N, T, D_y)
            Tensor corresponding to the paths in latent space.

        returns:
        --------
        tf.Tensor with shape (?), corresponding to the log-probability of
        joint-distribution given the input.
        """
        out_shape = None
        if not x.shape[:-2] == y.shape[:-2]:
            if len(x.shape) + 1 == len(y.shape):
                out_shape = y.shape[0].value
                x = tf.reshape(tf.concat(
                        [tf.expand_dims(x, axis=0) for i in range(out_shape)],
                        axis=0), [-1] + x.shape[-2:].as_list())
                y = tf.reshape(y, [-1] + y.shape[-2:].as_list())
            else:
                raise ValueError("Shape of inputs x, y does not match.")

        prior_log_prob = super(MarkovLatentDynamics, self).log_prob(x=y)
        log_p = tf.reduce_sum(self.emission_model.log_prob(
                x=x, y=y)) + prior_log_prob
        if out_shape is not None:
            return tf.reshape(log_p, [out_shape, -1])
        return log_p

    def sample(self, n_samples, y=None, time_steps=None):
        """Samples from the full-joint model p(x, y).

        params:
        -------
        n_samples: int
        y: tf.Tensor or None
            if None, a sample is drawn from the joint distribution p(x, y).
            Otherwise, samples are drawn from conditional p(x|y).
        time_teps: int

        returns:
        --------
        tuple of tf.Tensor. Respectively, samples from the states in the
        latent space and states from the observation space.
        """
        if time_steps is None:
            time_steps = self.time_steps
        latent_path = y
        if latent_path is None:
            latent_path = super(MarkovLatentDynamics, self).sample(
                    n_samples=n_samples)
        observation_path = self.emission_model.sample(
            n_samples=(), y=latent_path)

        if y is not None:
            return observation_path
        return latent_path, observation_path


class LatentLinearDynamicalSystem(MarkovLatentDynamics):
    """Class for implementation of Kalman-Filter generative model."""

    def __init__(self, lat_dim, obs_dim, time_steps, emission_model,
            init_transition_matrix_bias=None, full_covariance=True):
        """Sets up the parameters of the Kalman filter sets up super class.

        params:
        -------
        lat_dim: int
        obs_dim: int
        time_steps: int
        emission_model: model
            Describing the relation of the model.
        init_transition_matrix_bias: np.ndarray shape (lat_dim + 1, lat_dim)
            Initial value for the transition matrix.
        full_covariance: True
        """
        self.lat_dim = lat_dim
        self.obs_dim = obs_dim

        self.full_covariance = full_covariance
        mean_0 = np.random.normal(0, 1, lat_dim)
        if full_covariance:
            dist = tf.contrib.distributions.MultivariateNormalTriL
            cov_0 = tf.Variable(np.eye(lat_dim))
        else:
            dist = tf.contrib.distributions.MultivariateNormalDiag
            cov_0 = tf.nn.softplus(tf.Variable(np.ones(lat_dim)))

        # Transition matrix for the linear function.
        if init_transition_matrix_bias is None:
            self.transition_matrix = tf.Variable(
                    np.random.normal(0, 1, [lat_dim + 1, lat_dim]))
        else:
            self.transition_matrix = tf.Variable(init_transition_matrix_bias)
        prior = dist(mean_0, cov_0)
        # Transition model
        transition_model = ReparameterizedDistribution(
                out_dim=lat_dim, in_dim=lat_dim, transform=LinearTransform,
                distribution=dist, reparam_scale=False,
                gov_param=self.transition_matrix)

        super(LatentLinearDynamicalSystem, self).__init__(
                init_model=prior, transition_model=transition_model,
                emission_model=emission_model, time_steps=time_steps)


class KalmanFilter(LatentLinearDynamicalSystem):
    """Class for implementation of Kalman-Filter generative model."""

    def __init__(self, lat_dim, obs_dim, time_steps,
            init_transition_matrix_bias=None, full_covariance=True):
        """Sets up the parameters of the Kalman filter sets up super class.

        params:
        -------
        lat_dim: int
        obs_dim: int
        time_steps: int
        init_transition_matrix_bias: np.ndarray shape (lat_dim + 1, lat_dim)
            Initial value for the transition matrix.
        full_covariance: bool
            Covariance matrices are full if True, otherwise, diagonal.
        """
        mean_0 = np.random.normal(0, 1, lat_dim)
        if full_covariance:
            dist = tf.contrib.distributions.MultivariateNormalTriL
        else:
            dist = tf.contrib.distributions.MultivariateNormalDiag

        self.emission_matrix = tf.Variable(
                np.random.normal(0, 1, [lat_dim + 1, obs_dim]))
        # Emission model is reparameterized Gaussian with linear
        # transformation.
        emission_model = ReparameterizedDistribution(
                out_dim=obs_dim, in_dim=lat_dim, transform=LinearTransform,
                distribution=dist, reparam_scale=False,
                gov_param=self.emission_matrix)

        super(KalmanFilter, self).__init__(
                lat_dim=lat_dim, obs_dim=obs_dim, time_steps=time_steps,
                emission_model=emission_model,
                init_transition_matrix_bias=init_transition_matrix_bias,
                full_covariance=full_covariance)


class FLDS(LatentLinearDynamicalSystem):
    """Class for implementation of Kalman-Filter generative model."""

    def __init__(self, lat_dim, obs_dim, time_steps, nonlinear_transform,
            init_transition_matrix_bias=None, full_covariance=True, **kwargs):
        """Sets up the parameters of the Kalman filter sets up super class.

        params:
        -------
        lat_dim: int
        obs_dim: int
        time_steps: int
        nonlinear_transform: transform.Transform type
        init_transition_matrix_bias: np.ndarray shape (lat_dim + 1, lat_dim)
            Initial value for the transition matrix.
        full_covariance: bool
            Covariance matrices are full if True, otherwise, diagonal.
        """
        mean_0 = np.random.normal(0, 1, lat_dim)
        if full_covariance:
            dist = tf.contrib.distributions.MultivariateNormalTriL
        else:
            dist = tf.contrib.distributions.MultivariateNormalDiag

        # Emission model is reparameterized Gaussian with linear
        # transformation.
        emission_model = ReparameterizedDistribution(
                out_dim=obs_dim, in_dim=lat_dim, transform=nonlinear_transform,
                distribution=dist, reparam_scale=False, **kwargs)

        super(FLDS, self).__init__(
                lat_dim=lat_dim, obs_dim=obs_dim, time_steps=time_steps,
                emission_model=emission_model,
                init_transition_matrix_bias=init_transition_matrix_bias,
                full_covariance=full_covariance)

