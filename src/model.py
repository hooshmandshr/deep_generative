"""Provides the boiler plates and implementation for AEVB models.

The original AEVB paper:
https://arxiv.org/abs/1312.6114

Here, we provide abstract classes and implementation of certain models that are
used in our AEVB framework.
"""

import numpy as np
import tensorflow as tf

from distribution import LogitNormalDiag, MultiPoisson, MultiplicativeNormal,\
        MultiBernoulli, StateSpaceNormalDiag
from transform import GatedTransition

FULL_GAUSSIAN = tf.contrib.distributions.MultivariateNormalTriL
DIAG_GAUSSIAN = tf.contrib.distributions.MultivariateNormalDiag


class Model(object):
    """Abstract class for implementing models in form p(x|y) or p(x)."""

    def __init__(self, out_dim, in_dim=0):
        """Sets up the computational graphs for the model.

        params:
        -------
        in_dim: int
            Dimensionality of the input parameters (if model is conditional).
            If 0, the model is not conditional meaning there is p(x).
        out_dim: int
            Dimensionality of the random variable that is goverend by models.
        """
        self.in_dim = in_dim
        self.out_dim = out_dim

    def sample(self, n_samples, y=None):
        """Samples from the model conditional/or not."""
        if self.in_dim == 0 and y is not None:
            raise ValueError(
                    "Model is not conditional and does not take 'y'.")

        if not self.in_dim == 0 and y is None:
            raise ValueError(
                    "Model is conditional, 'y' can not be None.")

    def log_prob(self, x, y=None):
        """Computes log-prob of samples given the value of the distrib."""
        pass

    def entropy(self, y=None):
        """Computes the closed form entropy of the model if it exists."""
        pass

    def has_entropy(self):
        """If the model has explicit analytical entropy."""
        return False

    def get_regularizer(self):
        """Returns a regularization for the model if it exists."""
        return 0.

class ReparameterizedDistribution(Model):
    """Class that Implements reparameterized distributions p(x|y).

    p(x|y) is a known distribution q(x; f(y)) where the parameters of the
    distribution q() are governed by variable y through a transformation f().
    """

    def __init__(self, out_dim, in_dim, distribution, transform,
            reparam_scale=True, mult_normal_vars=None, **kwargs):
        """Initializes the model object and sets up necessary variables.

        params:
        -------
        out_dim: int
            Dimensionality of the output random variable x which is conditioned
            on the input variable.
        in_dim: int
            Dimensionality of the input (random) variable y.
        distribution: tensorflow.distributions.Distribution
            Known distribution that is reparameterized by a function f().
            distributions that are supported:
            -MultivariateNormalTril
            -MultivariateNormalDiag
            -LogitNormalDiag
            -MultiPoisson
            -MultiplicativeNormal:
                in_dim should be (time, dim)
                out_dim should be (time, Dim)
        transforma: Class that is extension of Transform
            Type of transformation f() to be applied to y in order to get
            the parameters of the know distributions.
        reparam_scale: Bool of tf.Tensor
            Whether to reparameterize the scale of the distribution (e.g.
            covariance of a Gaussian) if such a parameter exists for the
            distribution. If False, a tf.Variable will be randomly initialized.
            If a tf.Tensor, the given tensor will be the scale parameter of
            the distribution.
        mult_normal_vars: None or tuple
            If not None, it must be a list of tf.Tensor of length 3 that
            respectively are Q1, Q, A matrices where each have shape
            (out_dim, out_dim).
        **kwargs:
            parameters to be passed to the Transform constructors.
        """    
        super(ReparameterizedDistribution, self).__init__(
                out_dim=out_dim, in_dim=in_dim)
        
        self.dist_class = distribution
        self.transform_class = transform
        self.reparam_scale = reparam_scale
        self.scale_param = None
        # This list containts any concrete computation graph for this
        # reparametrized distribution given different inputs.
        self.dist_dict = {}
        self.trans_args = kwargs
        # already created reparametrization transforms for the model.
        # The logical order of these transformation should be handled
        # internally based on the distribution class.
        self.transforms = []
        self.mult_normal_vars = mult_normal_vars
        if self.mult_normal_vars is not None:
            msg = "mult_normal_vars must be None or a tuple of 3 tf.Tensor"
            msg += "with shape ({}, {}).".format(out_dim[-1], out_dim[-1])
            if not isinstance(self.mult_normal_vars, tuple):
                raise ValueError(msg)
            if not len(self.mult_normal_vars) == 3:
                raise ValueError(msg)
            for e in self.mult_normal_vars:
                if not isinstance(e, tf.Tensor):
                    raise ValueError(msg)
                if not e.shape == (out_dim[-1], out_dim[-1]):
                    raise ValueError(msg)
        if self.dist_class is StateSpaceNormalDiag:
            if not(len(in_dim) == 2 and len(out_dim) == 2):
                raise ValueError(
                        "in_dim and out_dim must be 2 dimensional for StateSpaceNormalDiag.")
            if not(in_dim[0] == out_dim[0]):
                raise ValueError(
                        "First dimension of out_dim and in_dim should be the same for StateSpaceNormalDiag.")

    def get_transforms(self):
        """Initializes or gets the transformations necessary for reparam.

        Each distribution type has different scale shape. This propmts this
        function to set up the correct shape for the scale if it is necessary
        for the scale to bre reparameterized.

        returns:
        --------
        transform.Transform
        """
        if len(self.transforms) > 0:
            return self.transforms

        if self.transform_class is GatedTransition:
            if self.dist_class is DIAG_GAUSSIAN:
                # Only one transform is needed
                self.transforms.append(self.transform_class(
                        in_dim=self.in_dim, **self.trans_args))
                return self.transforms

        # Regardless of distribution type, the location has the same
        # shape as the output.
        if self.dist_class is MultiplicativeNormal: 
            in_time, in_dim = self.in_dim
            time, out_dim = self.out_dim
            # Parameters of the C matrix and M matrix.
            self.transforms.append(self.transform_class(
                in_dim=in_dim,
                out_dim=out_dim * out_dim + out_dim,
                **self.trans_args))
            # Parameters of the M matrix.
            # self.transforms.append(self.transform_class(
            #    in_dim=in_dim,
            #    out_dim=out_dim,
            #    **self.trans_args))
        elif self.dist_class is StateSpaceNormalDiag:
            self.transforms.append(self.transform_class(
                in_dim=self.in_dim[1], out_dim=self.out_dim[1],
                **self.trans_args))
            self.transforms.append(self.transform_class(
                in_dim=self.in_dim[1], out_dim=self.out_dim[1],
                **self.trans_args))

        else: 
            self.transforms.append(self.transform_class(
                in_dim=self.in_dim, out_dim=self.out_dim,
                **self.trans_args))

        # Reparameteriz scale if needed.
        if self.reparam_scale == True:
            if self.dist_class is DIAG_GAUSSIAN or\
                    self.dist_class is LogitNormalDiag:
                # Diagonal Covariance.
                self.transforms.append(self.transform_class(
                    in_dim=self.in_dim, out_dim=self.out_dim,
                    **self.trans_args))
            # Multivariate Normal With full covariance.
            elif self.dist_class is FULL_GAUSSIAN:
                # Cholesky factor of the covariance matrix.
                self.transforms.append(self.transform_class(
                    in_dim=self.in_dim, out_dim=self.out_dim * self.out_dim,
                    **self.trans_args))

        return self.transforms
  
    def get_distribution(self, y):
        """Get the tf.Distribution given y as an input for p(x|y).

        params:
        -------
        y: tf.Tensor
            Input that governs the conditional distribution p(x|y).

        returns:
        --------
        tf.Distribution for p(x|y) if y is None 
        """
        if not self.dist_class is MultiplicativeNormal:
            if self.dist_class is StateSpaceNormalDiag:
                pass
            elif not y.shape[-1].value == self.in_dim:
                raise ValueError(
                        "Input must have dimension {}".format(self.in_dim))
        if y in self.dist_dict:
            # Necessary tf.Distribution has already been created for y.
            return self.dist_dict[y]

        # Get the necessary transformations for scale and location of the
        # distribution
        transforms = self.get_transforms()
        # Multivariate (Logit) normal with diagonal covariance.
        if self.dist_class is DIAG_GAUSSIAN or\
                self.dist_class is LogitNormalDiag:
            if self.transform_class is GatedTransition:
                # There is only one transformation that outputs two variabels.
                loc, scale = transforms[0].operator(y)
                dist = self.dist_class(loc, scale)
                self.dist_dict[y] = dist
                return dist
            # Rectify standard deviation so that it is a smooth
            # positive function
            loc_ = transforms[0].operator(y)
            scale_ = self.reparam_scale
            if self.reparam_scale is True:
                scale_ = tf.nn.softplus(transforms[1].operator(y))
            else:
                if self.reparam_scale is False:
                    if self.scale_param is None:
                        scale_ = tf.Variable(
                                np.zeros(self.out_dim), dtype=loc_.dtype)
                        self.scale_param = scale_
                    else:
                        scale_ = self.scale_param
                tot_unique_dist = 1
                for dim in loc_.shape[:-1]:
                    tot_unique_dist *= dim.value
                scale_ = [tf.expand_dims(
                    scale_, axis=0) for i in range(tot_unique_dist)]
                scale_ = tf.concat(scale_, axis=0)
                scale_shape = loc_.shape[:-1].as_list(
                        ) + [self.out_dim]
                scale_ = tf.nn.softplus(tf.reshape(scale_, scale_shape))

            dist = self.dist_class(loc=loc_, scale_diag=scale_)

        elif self.dist_class is StateSpaceNormalDiag:
            # Rectify standard deviation so that it is a smooth
            # positive function
            loc_ = transforms[0].operator(y)
            scale_ = self.reparam_scale
            if self.reparam_scale is True:
                scale_ = tf.nn.softplus(transforms[1].operator(y))
            else:
                raise NotImplemented(
                        "Use StateSpaceNormalDiag with reparam_scale=True")
            dist = self.dist_class(loc=loc_, scale=scale_)

        # Multivariate Poisson (independent variables).
        elif self.dist_class is MultiPoisson:
            rate_ = transforms[0].operator(y)
            dist = self.dist_class(rate_)

        # Multivariate Bernoulli (independent variables).
        elif self.dist_class is MultiBernoulli:
            probs_ = tf.nn.sigmoid(transforms[0].operator(y))
            dist = self.dist_class(probs_)

        # Multivariate Normal With full covariance.
        elif self.dist_class is FULL_GAUSSIAN:
            loc_ = transforms[0].operator(y)
            cov_ = self.reparam_scale
            if self.reparam_scale is True:
                cov_ = transforms[1].operator(y)
                cov_ = tf.reshape(cov_, [-1, self.out_dim, self.out_dim])
            else:
                if self.reparam_scale is False:
                    if self.scale_param is None:
                        cov_ = tf.Variable(
                                np.eye(self.out_dim), dtype=loc_.dtype)
                        self.scale_param = cov_
                    else:
                        cov_ = self.scale_param
                # The covariance should have the same shape as the mean
                tot_unique_dist = 1
                for dim in loc_.shape[:-1]:
                    tot_unique_dist *= dim.value
                cov_ = [tf.expand_dims(
                    cov_, axis=0) for i in range(tot_unique_dist)]
                cov_ = tf.concat(cov_, axis=0)
                cov_shape = loc_.shape[:-1].as_list(
                        ) + [self.out_dim, self.out_dim]
                cov_ = tf.reshape(cov_, cov_shape)

            dist = self.dist_class(loc=loc_, scale_tril=cov_)

        # MultiplicativeNormal for LDS models.
        elif self.dist_class is MultiplicativeNormal:
            in_time, in_dim = self.in_dim
            time, out_dim = self.out_dim
            if not in_time == time:
                raise ValueError("in_dim and out_dim mistmatch.")
            # How many examples/distributions in parallel.
            dtype = y.dtype
            n_ex = y.shape[0].value
            # Free parameters for covariances and transition matrix.
            # Shape of Q1, Q, A
            var_shape = [1, out_dim, out_dim]
            if n_ex == 0:
                var_shape = [out_dim, out_dim]
            def get_positive_definite_variable(index=0):
                """Gets cholesky factor variable serving as SPD matrix."""
                # Get a lower triangular variable
                var = tf.linalg.band_part(tf.Variable(
                        np.random.normal(0, 1, var_shape),
                        dtype=dtype), -1, 0)
                return tf.matmul(var, var, transpose_b=True)

            if self.mult_normal_vars is None:
                # Initialize the variables of the distribution.
                # These variables are Q1, Q, A
                q_init=get_positive_definite_variable()
                q_matrix=get_positive_definite_variable()
                # Shape of each parameter.
                a_matrix = tf.Variable(
                        np.random.normal(0, 1, var_shape), dtype=dtype)
            else:
                # Q1, Q, A are given as parameters to the initializer of the
                # CLass.
                q_init = self.mult_normal_vars[0]
                q_matrix = self.mult_normal_vars[1]
                a_matrix = self.mult_normal_vars[2]
                if n_ex > 0:
                    q_init = tf.expand_dims(q_init, axis=0)
                    q_matrix = tf.expand_dims(q_matrix, axis=0)
                    a_matrix = tf.expand_dims(a_matrix, axis=0)

            if n_ex > 0:
                q_init = tf.concat([q_init for i in range(n_ex)], axis=0)
                q_matrix = tf.concat([q_matrix for i in range(n_ex)], axis=0)
                a_matrix = tf.concat([a_matrix for i in range(n_ex)], axis=0)

            # c_matrix should be semi-positive definite
            trans_out = transforms[0].operator(y)
            shape_ = trans_out.shape.as_list()
            dim_ = len(shape_)
            slice_b = [0 for i in range(dim_ - 1)]
            slice_e = [-1 for i in range(dim_ - 1)]
            c_matrix = tf.slice(
                    trans_out, slice_b + [0], slice_e + [out_dim * out_dim])
            c_matrix = tf.linalg.band_part(
                    tf.reshape(c_matrix, shape_[:-1] + [out_dim, out_dim]),
                    -1, 0)
            # Ensure that c_matrix is a positive-definite matrix.
            c_matrix = tf.matmul(c_matrix, c_matrix, transpose_b=True)
            m_matrix = tf.slice(
                    trans_out, slice_b + [out_dim * out_dim], slice_e + [-1])

            dist = MultiplicativeNormal(
                    q_init=q_init,
                    q_matrix=q_matrix,
                    a_matrix=a_matrix,
                    c_matrix=c_matrix,
                    m_matrix=m_matrix)

        # Store the created distribution for tensor y in the dictionary.
        self.dist_dict[y] = dist
        return dist

    def log_prob(self, x, y):
        """Computes log probability of x under reparam distribution.

        Returns:
        --------
            tensorflow.Tensor that contains the log probability of input x.
        """
        dist = self.get_distribution(y)
        return dist.log_prob(x)

    def sample(self, y, n_samples):
        """Samples from the reparameterized distribution.

        Parameters:
        -----------
        n_samples: int
            Number of samples.
        Returns:
        --------
            tensorflow.Tensor.
        """
        dist = self.get_distribution(y)
        return dist.sample(n_samples)

    def entropy(self, y):
        """Samples from the reparameterized distribution.

        Parameters:
        -----------
        y: tf.Tensor
            Input variable that governs the distribution.
        Returns:
        --------
            tensorflow.Tensor.
        """
        dist = self.get_distribution(y)
        return dist.entropy()

    def has_entropy(self):
        """Based on type of base distribution the model can have entropy."""
        if self.dist_class is FULL_GAUSSIAN or\
                self.dist_class is DIAG_GAUSSIAN or\
                self.dist_class is MultiplicativeNormal:
            return True
        return False

    def get_regularizer(self, **kwargs):
        """Gets the regularizations for the transformations in the model."""
        reg = 0.
        for transform in self.get_transforms():
            reg += transform.get_regularizer(**kwargs)
        return reg

