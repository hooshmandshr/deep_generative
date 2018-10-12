"""Tools that implement models  expressed by a normalizing flow transformation.

Contains the boilerplate for general class of distributions (conditional or not)
that are expressed by a bijective transformation to a random variable described
by Model class.
"""

import numpy as np
import tensorflow as tf

from model import Model
from norm_flow import NormalizingFlow


class NormalizingFlowModel(Model):
    """Class for random variable that is transformed by a normalizing flow.

    This class is an extension of Model class with an exception. It does not
    necessary implement log_prob() method since the inverse of the normalizing
    flow function is not available necessarily.
    """

    def __init__(self, base_model, norm_flow):
        """Samples from the model and computes log probability of the samples.

        params:
        -------
        base_model:
            Instance of class Model.
        norm_flow:
            Instance of class NormalizingFlow.
        """
        if not isinstance(base_model, Model):
            raise KeyError("base_model must be of type Model.")
        if not isinstance(norm_flow, NormalizingFlow):
            raise KeyError("norm_flow must be of type NormalizingFlow.")
        self.has_density = False
        self.base_model = base_model
        self.norm_flow = norm_flow

    def sample(self, n_samples, y=None):
        """Samples from the model and computes log probability of the samples.

        params:
        -------
        y: tf.Tensor
            If model is conditional y must be None; Otherwise, a tf.Tensor.

        returns:
        --------
        tuple of tf.Tensor. First member of the tuple is samples from model.
        Second member is the log-density of the samples of the model.
        """
        samples = self.base_model.sample(n_samples=n_samples, y=y)
        log_prob = self.base_model.log_prob(x=samples, y=y)
        self.norm_flow.operator(samples)
        log_prob -= self.norm_flow.log_det_jacobian(samples)
        return samples, log_prob

