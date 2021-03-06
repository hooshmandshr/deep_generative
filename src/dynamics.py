"""Class for implementing dynamical systems models according to Model design.

The goal of this library is to provide necessary dynamical systems models
to easily sample from, compute density, etc.
"""

import numpy as np
import tensorflow as tf

from distribution import MultiPoisson, MultiBernoulli, StateSpaceNormalDiag
from model import Model, ReparameterizedDistribution
from transform import LinearTransform, LSTMcell, GatedTransition
from transform import MultiLayerPerceptron as MLP


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
        self.order = order
        self.time_steps = time_steps
        # In effect, the prior of the dynamical system.
        self.init_model = init_model
        self.transition_model = transition_model
        # dimension of each state
        self.state_dim = self.transition_model.out_dim
        # Check whether there is enough history for the order of the
        # Markov process
        if not time_steps > order:
            raise ValueError("'time_steps' must be more that 'order'.")
        if isinstance(self.init_model, Model):
            if not self.state_dim == self.order * self.init_model.out_dim:
                msg = "Dimensions of prior and transition model mismatch: {}, {}."
                raise ValueError(msg.format(
                    self.state_dim, self.init_model.out_dim))
        # Check that the input dim of transition matches the order of Markov
        # model.
        if not(self.transition_model.in_dim == order * self.state_dim):
            msg = "Input to transition model should have dim {}"
            raise ValueError(msg.format(order * self.state_dim))
        self.out_shape = (time_steps, self.state_dim)
        super(MarkovDynamics, self).__init__(out_dim=self.state_dim * time_steps)

    def log_prob(self, x):
        """Log probability of the dynamics model p(x) = p(x1, x2, ...).

        params:
        -------
        x: tf.Tensor
            Shape of x should match the shape of model. In other words,
            (?, T, D)
        """
        n_sample = x.shape[0].value
        x_init = [x[..., i, :] for i in range(self.order)]
        x_pre = [x[..., (0 + i):(-self.order + i), :] for i in range(self.order)]
        if self.order == 1:
            x_init = x_init[0]
            x_pre = x_pre[0]
        else:
            x_init = tf.concat(x_init, axis=-1)
            x_pre = tf.concat(x_init, axis=-1)

        log_prob = self.init_model.log_prob(x_init)
        log_prob += tf.reduce_sum(
                self.transition_model.log_prob(
                    x=x[..., self.order:, :],
                    y=x_pre),
                axis=-1)
        return log_prob

    def sample(self, n_samples, init_states=None, time_steps=None,
            stochastic=True):
        """Samples from the markov dynamics model.

        params:
        -------
        n_samples: int
        time_steps: int or None
            If None the numbe of states in the samples will be equal to that
            of the model.
        stochastic: bool
            If True, you samples are return from the model. If Flase, given
            initial states, the system is evolved deterministically.

        Returns:
        --------
        tf.Tensor
        """
        if time_steps is None:
            time_steps = self.time_steps
        # Sample from the prior distribution for initial state.
        states = []

        if init_states is not None:
            assert isinstance(init_states, tf.Tensor)
            assert init_states.shape.as_list() == [n_samples, self.order, self.state_dim]
        else:
            init_states = self.init_model.sample((1, n_samples))
            init_states = tf.reshape(
                    init_states, [n_samples, self.order, self.state_dim])
        # Transpose time dimension and example dimension so that
        # states can line up as input to transition function.
        init_states = tf.transpose(init_states, perm=[1, 0, 2])

        for i in range(init_states.shape[0].value):
            states.append(init_states[i:i+1])
        # draw from the transition model.
        for i in range(time_steps - self.order):
            y = tf.reshape(
                    tf.transpose(tf.concat(states[-self.order:], axis=0), [1, 0, 2]),
                    [n_samples, -1])
            if not stochastic and isinstance(self.transition_model,
                    ReparameterizedDistribution):
                # If deterministic sampling (i.e. evolution is required)
                det_trans = self.transition_model.transforms[0].operator(y)
                if isinstance(
                        self.transition_model.transforms[0], GatedTransition):
                    # If trans is gated transform then the transform outputs
                    # a tuple of mu and sigma and we want mu in deterministic
                    # transition.
                    det_trans = det_trans[0]
                states.append(tf.expand_dims(det_trans, axis=0))
            else:
                states.append(self.transition_model.sample(
                    y=y, n_samples=1))

        return tf.transpose(tf.concat(states, axis=0), perm=[1, 0, 2])


class TimeVariantDynamics(Model):

    def __init__(self, state_dim, trans_hidden_units, time_steps):
        """Sets up the necessary networks for the markov dynamics.

        params:
        -------
        state_dim: int
        obs_dim: int
        trans_hidden_units: list of int
        """
        self.time_steps = time_steps
        # dimension of each state space
        self.state_dim = state_dim
        self.dist_type = tf.contrib.distributions.MultivariateNormalDiag

        self.init_model = self.dist_type(
                tf.Variable(np.zeros(self.state_dim)),
                tf.nn.softplus(tf.Variable(np.ones(self.state_dim))))
        # List of transformation per each step
        self.time_transition_model = []
        for time in range(self.time_steps - 1):
            self.time_transition_model.append(
                    ReparameterizedDistribution(
                        out_dim=self.state_dim, in_dim=self.state_dim,
                        transform=MLP,
                        distribution=self.dist_type,
                        hidden_units=trans_hidden_units))
        super(TimeVariantDynamics, self).__init__(
                out_dim=self.state_dim * time_steps)

    def sample(self, n_samples):
        """Log probability of the dynamics model p(x) = p(x1, x2, ...).

        params:
        -------
        n_samples: int
            Number of samples (paths).
        """
        samples = []
        samples.append(
                self.init_model.sample([1, n_samples]))
        for time_model in self.time_transition_model:
            samples.append(time_model.sample(n_samples=1, y=samples[-1][0]))

        return tf.transpose(tf.concat(samples, axis=0), perm=[1, 0, 2])

    def log_prob(self, x):
        """Log probability of the dynamics model p(x) = p(x1, x2, ...).

        params:
        -------
        x: tf.Tensor
            Shape of x should match the shape of model. In other words,
            (?, T, D)
        """
        log_prob = self.init_model.log_prob(x[:, 0])
        for t, model in enumerate(self.time_transition_model):
            log_prob += model.log_prob(x=x[:, t + 1], y=x[:, t])
        return log_prob


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
        log_p = None
        if not(x.shape[-3:-1] == y.shape[-3:-1]):
            raise ValueError("Shape of inputs x, y does not match.")
        if len(x.shape) == 3 and len(y.shape) == 3:
            log_p = tf.reduce_sum(self.emission_model.log_prob(
                x=x, y=y), axis=-1)
        elif len(x.shape) == 3 and len(y.shape) == 4:
            log_p = []
            for i in range(x.shape[0].value):
                log_p.append(tf.reduce_sum(self.emission_model.log_prob(
                    x=x[i:i+1], y=y[:, i:i+1]), axis=-1))
            log_p = tf.concat(log_p, axis=1)
        else:
            raise ValueError("Shape of inputs x, y does not match.")

        # Add prior
        return log_p + super(MarkovLatentDynamics, self).log_prob(x=y)

    def sample(self, n_samples, init_states=None, y=None, time_steps=None,
            stochastic=True, noisy_obs=True):
        """Samples from the full-joint model p(x, y).

        params:
        -------
        n_samples: int
        init_states: tf.Tensor
        y: tf.Tensor or None
            if None, a sample is drawn from the joint distribution p(x, y).
            Otherwise, samples are drawn from conditional p(x|y).
        time_teps: int
        stochastic: bool
            If True, the dynamics is evovled stochastically.
        noisy_obs: bool
            If True, the observations are samples. Else, mean observation
            is used.

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
                    n_samples=n_samples, init_states=init_states,
                    time_steps=time_steps, stochastic=stochastic)
        observation_path = None
        if not noisy_obs:
            observation_path = self.emission_model.mean(y=latent_path)
        else:
            observation_path = self.emission_model.sample(
                n_samples=(), y=latent_path)

        if y is not None:
            return observation_path
        return latent_path, observation_path

    def get_dynamics_model(self):
        """Returns the dynamics model only (prior)."""
        return MarkovDynamics(
                init_model=self.init_model, transition_model=self.transition_model,
                time_steps=self.time_steps, order=self.order)

    def mean(self, y):
        """Mean of the conditional p(x|y).

        params:
        -------
        y: tf.Tensor or None
            if None, a sample is drawn from the joint distribution p(x, y).
            Otherwise, samples are drawn from conditional p(x|y).

        returns:
        --------
        tuple of tf.Tensor. Respectively, samples from the states in the
        latent space and states from the observation space.
        """
        return self.emission_model.mean(y=y)

    def get_regularizer(self):
        """Regularizer for the dynamical system."""
        regul = self.transition_model.get_regularizer()
        regul += self.emission_model.get_regularizer()
        return regul


class LatentLinearDynamicalSystem(MarkovLatentDynamics):
    """Class for implementation of Kalman-Filter generative model."""

    def __init__(self, lat_dim, obs_dim, time_steps, emission_model,
            init_transition_matrix_bias=None, full_covariance=True, order=1):
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
        # TODO: determine what the initial point is.
        # mean_0 = np.random.normal(0, 1, lat_dim * order)
        mean_0 = np.zeros(lat_dim * order)
        if full_covariance:
            dist = tf.contrib.distributions.MultivariateNormalTriL
            cov_0 = tf.Variable(np.eye(lat_dim * order))
        else:
            dist = tf.contrib.distributions.MultivariateNormalDiag
            cov_0 = tf.nn.softplus(tf.Variable(np.ones(lat_dim * order)))

        # Transition matrix for the linear function.
        if init_transition_matrix_bias is None:
            init_ = np.concatenate([np.eye(lat_dim) for i in range(order)],
                    axis=0)
            init_ = np.concatenate([init_, np.zeros([1, lat_dim])])
            self.transition_matrix = tf.Variable(init_)
        else:
            self.transition_matrix = tf.Variable(init_transition_matrix_bias)
        prior = dist(mean_0, cov_0)

        # Covariance Matrix cholesky factor of evolution.
        cov_t = None
        if full_covariance:
            cov_t = tf.Variable(np.eye(lat_dim))
        else:
            cov_t = tf.Variable(np.zeros(lat_dim))
        # Transition model
        transition_model = ReparameterizedDistribution(
                out_dim=lat_dim, in_dim=lat_dim * order,
                transform=LinearTransform,
                distribution=dist, reparam_scale=cov_t,
                has_bias=False, gov_param=self.transition_matrix)

        # Covariance parameters are cholesky factors, so multiply to get the
        # covariances.
        q_factor = transition_model.scale_param
        self.q_init = None
        self.q_matrix = cov_t
        if not full_covariance:
            self.q_init = tf.diag(cov_0)
            self.q_matrix = tf.diag(tf.nn.softplus(cov_t))
        else:
            self.q_init = tf.matmul(cov_0, cov_0, transpose_b=True)
            self.q_matrix = tf.matmul(cov_t, cov_t, transpose_b=True)
        self.a_matrix = self.transition_matrix[:lat_dim]


        super(LatentLinearDynamicalSystem, self).__init__(
                init_model=prior, transition_model=transition_model,
                emission_model=emission_model, time_steps=time_steps,
                order=order)

    def get_linear_dynamics_params(self, transpose_a=False):
        """Gets parameter of the latent LDS i.e. Q1, Q, A.

        params:
        -------
        transpose_a: bool
            If True return transposed a_matrix. This is due to the fact that
            linear transformation class instead of Ax applies x^TA.
        """
        if transpose_a:
            return self.q_init, self.q_matrix, tf.transpose(self.a_matrix)
        return self.q_init, self.q_matrix, self.a_matrix


class KalmanFilter(LatentLinearDynamicalSystem):
    """Class for implementation of Kalman-Filter generative model."""

    def __init__(self, lat_dim, obs_dim, time_steps,
            init_transition_matrix_bias=None, full_covariance=True, order=1):
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
                full_covariance=full_covariance, order=order)


class FLDS(LatentLinearDynamicalSystem):
    """Class for implementation of fLDS/pfLDS generative model."""

    def __init__(self, lat_dim, obs_dim, time_steps, nonlinear_transform,
            init_transition_matrix_bias=None, poisson=False, binary=False,
            full_covariance=True, order=1, **kwargs):
        """Sets up the parameters of the Kalman filter sets up super class.

        params:
        -------
        lat_dim: int
        obs_dim: int
        time_steps: int
        nonlinear_transform: transform.Transform type
        init_transition_matrix_bias: np.ndarray shape (lat_dim + 1, lat_dim)
            Initial value for the transition matrix.
        poisson: bool
            If False the imission distribution is Gaussian, if not the emission
            distribution is poisson.
        full_covariance: bool
            Covariance matrices are full if True, otherwise, diagonal.
        """

        if poisson:
            dist = MultiPoisson
        elif binary:
            em_dist = MultiBernoulli

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
                full_covariance=full_covariance, order=order)

class MLPDynamics(MarkovLatentDynamics):
    """Class for implementation of ffDS/pffDS generative model."""

    def __init__(self, lat_dim, obs_dim, time_steps, transition_layers,
            emission_transform, poisson=False, binary=False, residual=False,
            full_covariance=False, order=1, **kwargs):
        """Sets up the parameters of the Kalman filter sets up super class.

        params:
        -------
        lat_dim: int
        obs_dim: int
        time_steps: int
        transition_layers: list of int
        nonlinear_transform: transform.Transform type
        poisson: bool
            True if observation is count data. Otherwise, observation is
            continuous.
        residual: bool
            If True MLP transition function is of residual form.
        full_covariance: bool
            Covariance matrices of noise processes are full if True. otherwise,
            diagonal covariance.
        """
        self.full_covariance = full_covariance
        if full_covariance:
            q_init = tf.Variable(np.eye(lat_dim * order))
            dist = tf.contrib.distributions.MultivariateNormalTriL
        else:
            q_init = tf.Variable(np.ones(lat_dim * order))
            dist = tf.contrib.distributions.MultivariateNormalDiag

        em_dist = dist
        if poisson:
            em_dist = MultiPoisson
        elif binary:
            em_dist = MultiBernoulli

        # Prior distribution for initial point
        prior = dist(np.zeros(lat_dim * order), q_init)
        # Transition model.
        trans_model = ReparameterizedDistribution(
                out_dim=lat_dim, in_dim=lat_dim * order,
                transform=MLP,
                distribution=dist, reparam_scale=False,
                hidden_units=transition_layers, residual=residual)
        # Emission model is reparameterized Gaussian with linear
        # transformation.
        emission_model = ReparameterizedDistribution(
                out_dim=obs_dim, in_dim=lat_dim, transform=emission_transform,
                distribution=em_dist, reparam_scale=False, **kwargs)

        super(MLPDynamics, self).__init__(
                init_model=prior, transition_model=trans_model,
                emission_model=emission_model, time_steps=time_steps,
                order=order)

    def get_regularizer(self):
        """Regularizer for the dynamical system."""
        base = super(MLPDynamics, self).get_regularizer()
        return base
        noise_scale = self.transition_model.scale_param
        noise_regul = 0.
        if self.full_covariance:
            noise_regul = tf.linalg.trace(tf.matmul(
                noise_scale, noise_scale, transpose_b=True))
        else:
            noise_regul = tf.reduce_sum(tf.square(noise_scale))
        return base + 100. * noise_regul



class DeepKalmanDynamics(MarkovLatentDynamics):
    """Class for implementation of gated transition generative model."""

    def __init__(self, lat_dim, obs_dim, time_steps, transition_units,
            emission_layers, poisson=False, binary=False):
        """Sets up the parameters of the Kalman filter sets up super class.

        params:
        -------
        lat_dim: int
        obs_dim: int
        time_steps: int
        transition_units: int
            Number of hidden units for the gated transition transform in DKF.
        poisson: bool
            True if observation is count data. Otherwise, observation is
            continuous.
        """
        q_init = tf.Variable(np.ones(lat_dim))
        dist = tf.contrib.distributions.MultivariateNormalDiag

        em_dist = dist
        if poisson:
            em_dist = MultiPoisson
        elif binary:
            em_dist = MultiBernoulli

        # Prior distribution for initial point
        prior = dist(np.zeros(lat_dim), q_init)
        # Transition model.
        trans_model = ReparameterizedDistribution(
                out_dim=lat_dim, in_dim=lat_dim,
                transform=GatedTransition,
                distribution=dist, reparam_scale=True,
                hidden_units=transition_units)
        # Emission model is reparameterized Gaussian with linear
        # transformation.
        emission_model = ReparameterizedDistribution(
                out_dim=obs_dim, in_dim=lat_dim, transform=MLP,
                distribution=em_dist, reparam_scale=False,
                hidden_units=emission_layers)

        super(DeepKalmanDynamics, self).__init__(
                init_model=prior, transition_model=trans_model,
                emission_model=emission_model, time_steps=time_steps)


class MarkovDynamicsDiagnostics(object):
    """Class for general diagnostics of dynamical systems."""

    def __init__(self, dynamics, n_samples, grid_size, time_forward):
        """Sets up the necessary operations like extrapolation.

        params:
        -------
        """
        if not isinstance(dynamics, MarkovDynamics):
            raise ValueError("dynamics should be dynamics.MarkovDynamics")
        self.dynamics = dynamics
        self.n_samples = n_samples
        self.time_forward = time_forward
        self.grid_size = grid_size

        # Dictionary that keeps placeholders and tensors that keep their
        # Extrapolated versions.
        self.init_tensor = {}
        self.extr_tensor = {}
        in_tensor, ex_tensor = self.init_extrapolate(
                n_init_points=self.n_samples, time_forward=self.time_forward)
        self.init_tensor["default"] = in_tensor
        self.extr_tensor["default"] = ex_tensor

        # Make the tensors for the grids.
        self.grid_size = grid_size
        if self.dynamics.order == 1:
            # Set up a tensor for all the initial states.
            in_tensor, ex_tensor = self.init_extrapolate(
                n_init_points=self.grid_size, time_forward=2, stochastic=False)
            self.init_tensor["grid"] = in_tensor
            self.extr_tensor["grid"] = ex_tensor

    def init_extrapolate(self, n_init_points, time_forward, stochastic=True):
        """Sets up tensors and operations for stochastic extrapolations.

        params:
        -------
        n_init_points: int
            The number of initial points that the dynamics extrapolates from.
        time_forward: int
            The number of total time points to extrapolate to.
        stochastic: bool
            Whether to evolve the states stochastically. If reparameterization,
            use the analytic mean of the model.

        returns:
        --------
        tuple of tf.Tensor, first is the place holder for the initial points
        and the second is for the extrapolation.
        """
        order = self.dynamics.order
        init_tensor = tf.placeholder(
                shape=[n_init_points, order, self.dynamics.state_dim],
                dtype=tf.float64, name="stateholder")
        extrapolate_tensor = self.dynamics.sample(
                n_samples=n_init_points,
                init_states=init_tensor, time_steps=time_forward,
                stochastic=stochastic, noisy_obs=False)
        return init_tensor, extrapolate_tensor

    def run_extrapolate(self, session, states, name="default"):
        """Runs and return sthe result of given extrapolation.

        params:
        -------
        session: tf.Session
            The open session of under which the dynamics system is open.
        states: np.ndarray
        name: string
            Key for the init and extrapolate tensor.
        """
        if not isinstance(states, np.ndarray):
            raise ValueError("States must be np.ndarray.")
        if len(states.shape) == 1:
            if not states.shape == (self.dynamics.state_dim,):
                raise ValueError("Input dimension should be {}".format(
                    self.dynamics.state_dim))
            # Put the singe state into correct shape for sampling from the
            # dynamics model.
            # Get the number of samples from the correspondent tensor.
            n_samples = self.init_tensor[name].shape[0].value
            states = states[None, None, :].repeat(n_samples, axis=0)
        else:
            if name == "grid":
                if not states.shape == (self.grid_size, self.dynamics.state_dim):
                    raise ValueError("Shape of grid must be {}.".format(
                        (self.grid_size, self.dynamics.state_dim)))
                # Reshape the states to match the init_tensor placeholder of
                # the grid extrapolation.
                states = states[:, None, :]

            elif not states.shape == self.init_tensor[name].shape:
                raise ValueError("Shape of states must be {}".format(
                    self.init_tensor.shape))
        stochastic = True
        if name == "grid":
            stochastic = False
        output = session.run(self.extr_tensor[name],
                feed_dict={self.init_tensor[name].name: states})
        return output


class DeepKalmanFilter(Model):

    def __init__(self, in_dim, out_dim, time_steps, rnn_hdim,
            mean_field=True, backward=True):

        super(DeepKalmanFilter, self).__init__(
                in_dim=in_dim, out_dim=out_dim * time_steps)

        self.state_dim = out_dim
        self.rnn_hdim = rnn_hdim
        self.time_steps = time_steps

        self.backward = backward
        self.mean_field = mean_field

        # LSTM RNN functions.
        self.rnn_fwd = LSTMcell(in_dim=self.in_dim, out_dim=rnn_hdim)
        self.rnn_bck = LSTMcell(in_dim=self.in_dim, out_dim=rnn_hdim)

        self.linear_mu_fwd = LinearTransform(
                in_dim=self.rnn_hdim, out_dim=self.state_dim)
        self.linear_sg_fwd = LinearTransform(
                in_dim=self.rnn_hdim, out_dim=self.state_dim)
        self.linear_mu_bck = None
        self.linear_sg_bck = None
        # Factorization transition function
        self.linear_transition = None

        if self.mean_field:
            if self.backward: 
                self.linear_mu_bck = LinearTransform(
                        in_dim=self.rnn_hdim, out_dim=self.state_dim)
                self.linear_sg_bck = LinearTransform(
                        in_dim=self.rnn_hdim, out_dim=self.state_dim)
        else:
            # Set up state transition function.
            self.linear_transition = LinearTransform(
                        in_dim=self.state_dim, out_dim=self.rnn_hdim)
            if self.backward:
                self.rnn_fwd = LSTMcell(in_dim=self.in_dim, out_dim=rnn_hdim)

        self.param_map = {}
        self.log_prob_map = {}
 
    def get_param(self, y):
        """Given observation, get parameteris of inference network."""

        if y in self.param_map:
            return self.param_map[y]

        mu = None
        sigma = None
        y_transpose = tf.transpose(y, perm=[1, 0, 2])

        # Get forward (and backward if necessary) states.
        hidden_fwd = self.rnn_fwd.operator(y_transpose)
        if self.backward:
            hidden_bck = self.rnn_bck.operator(y_transpose)

        if self.mean_field:
            mu_fwd = self.linear_mu_fwd.operator(hidden_fwd)
            sg_fwd = tf.nn.softplus(self.linear_sg_fwd.operator(hidden_fwd))

            if self.backward:
                mu_bck = self.linear_mu_bck.operator(hidden_bck)
                sg_bck = tf.nn.softplus(self.linear_sg_bck.operator(hidden_bck))

                sg_fwd_sqr = tf.square(sg_fwd)
                sg_bck_sqr = tf.square(sg_bck)
                sg_sqr_plus = sg_fwd_sqr + sg_bck_sqr
                mu = (mu_fwd * sg_bck_sqr + mu_bck * sg_fwd_sqr) / sg_sqr_plus
                sigma = sg_fwd_sqr * sg_bck_sqr / sg_sqr_plus
            else:
                mu = mu_fwd
                sigma = sg_fwd
            self.param_map[y] = (
                    tf.transpose(mu, perm=[1, 0, 2]),
                    tf.transpose(sigma, perm=[1, 0, 2]))

        else:
            rnn_hidden = hidden_fwd
            if self.backward:
                rnn_hidden += hidden_bck
            self.param_map[y] = rnn_hidden

        return self.param_map[y]

    def log_prob(self, x, y):
        """Log prob of samples given observations."""
        if x in self.log_prob_map:
            return self.log_prob_map[x]
        else:
            raise NotImplemented("General density function is not implemented yet.")

    def sample(self, n_samples, y):
        """Samples from the inference model given observation."""
        params = self.get_param(y)
        if self.mean_field:
            mu, sigma = params
            dist = StateSpaceNormalDiag(mu, sigma)
            result = dist.sample(n_samples)
            self.log_prob_map[result] = dist.log_prob(result)
            return result
        else:
            log_prob = 0.
            dist_ = tf.contrib.distributions.MultivariateNormalDiag
            hidden = params
            normalizer = 1 / 2.
            if self.backward:
                normalizer = 1 / 3.
            n_ex = y.shape[0].value
            particles = [tf.zeros([n_samples, n_ex, self.state_dim], dtype=y.dtype)]
            for time in range(self.time_steps):
                h_comb = normalizer * (tf.tanh(
                    self.linear_transition.operator(
                        particles[time])) + hidden[time])
                d = dist_(
                        self.linear_mu_fwd.operator(h_comb),
                        self.linear_sg_fwd.operator(h_comb))
                particles.append(d.sample(1)[0])
                log_prob += d.log_prob(particles[time + 1])
            # First Particle is just zero vector and has to be discarded.
            result = tf.concat(
                    [tf.expand_dims(p, axis=2) for p in particles[1:]], axis=2)
            # Store the computed log probability of samples.
            self.log_prob_map[result] = log_prob
            return result

    def get_regularizer(self):
        """Get regularization for all NN functions in the inference network."""
        return 0.
