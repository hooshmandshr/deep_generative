"""Tools that implement models  expressed by a normalizing flow transformation.

Contains the boilerplate for general class of distributions (conditional or not)
that are expressed by a bijective transformation to a random variable described
by Model class.
"""

import numpy as np
import tensorflow as tf

from model import Model
from norm_flow import NormalizingFlow, TimeAutoRegressivePlanarFlow
from transform import Transform, MultiLayerPerceptron


class NormalizingFlowModel(Model):
    """Class for random variable that is transformed by a normalizing flow.

    This class is an extension of Model class with an exception. It does not
    necessary implement log_prob() method since the inverse of the normalizing
    flow function is not available necessarily.
    """

    def __init__(self, in_dim, base_model, norm_flow_type, norm_flow_params={},
            transform_type=None, transform_params={}):
        """Samples from the model and computes log probability of the samples.

        params:
        -------
        in_dim: int
            Dimensionality of the variable upon which the model is conditioned.
            If 0, the random variable is independent.
        base_model: model.Model
            Base model for the random variable that is transformed according
            to some normalizing flow.
        norm_flow_type: calss that inherits from NormalizingFlow
            Normalizing flow that transforms the random variable to produce
            a new random variable.
        norm_flow_params: dict
            Parameters to be passed to the constructor of the NormalizingFlow.
        transform_type: class that inherits from transform.Transform 
            This allows for conditioning the model on a secondary variable by
            letting a transformation of a secondary variable to govern the
            parameters of the normalizing flow. If None, the normalizing flow
            model is unconditional.
        transform_params: dict
            Parameters to tbe passed to the constructor of Transform.
        """
        if not (isinstance(base_model, Model) or\
                isinstance(base_model, tf.distributions.Distribution)):
            raise ValueError("base_model must be of type Model.")

        assert issubclass(norm_flow_type, NormalizingFlow)

        # Get in and out dimensionalities from the base model.
        if isinstance(base_model, Model):
            # Check to make sure the input dim of base model is the same.
            if not base_model.in_dim == in_dim:
                raise ValueError(
                        "in_dim {} and base_model.in_dim {} don't match".format(
                            in_dim, base_model.in_dim))
            out_dim = base_model.out_dim
        elif isinstance(base_model, tf.distributions.Distribution):
            # Base model is a tensorflow.Distribution sub-class
            sample_shape = tf.squeeze(base_model.sample(1)).shape
            if len(sample_shape) < 2:
                out_dim = sample_shape[0].value
            else:
                out_dim = tuple(sample_shape.as_list())
        else:
            # Input is not valid
            msg = "'base_model' must be model.Model or\
                    tf.distributions.Distribution."
            raise ValueError(msg)

        super(NormalizingFlowModel, self).__init__(
                in_dim=in_dim, out_dim=out_dim)

        self.has_density = False
        self.base_model = base_model
        self.norm_flow_type = norm_flow_type

        if in_dim == 0:
            assert transform_type is None
        if in_dim > 0:
            # The model is conditional and therefore a transformation
            # is needed.
            assert issubclass(transform_type, Transform)

        self.transform_type = transform_type

        # Dictionary that keeps the normalizing flow transformation
        # given a specific input tensor.
        self.norm_flow_dict = {}
        self.norm_flow = None
        self.norm_flow_params = norm_flow_params
        self.transform_params = transform_params

    def get_normalizing_flow(self, y):
        """Gets the normalizing flow for conditional variable y.

        parameters:
        -----------
        y: tf.Tensor
            Variable upon which the model is conditioned.

        returns:
        --------
        norm_flow.NormalizingFlow object corresponding to the input tensor.
        """
        if y is not None and self.transform_type is None:
            raise ValueError("Model not conditional 'y' must be None.")
        if y is None and self.transform_type is not None:
            raise ValueError("Model is conditional 'y' must be passed.")

        if y is None:
            if self.norm_flow is not None:
                return self.norm_flow
            self.norm_flow = self.norm_flow_type(
                    dim=self.out_dim, **self.norm_flow_params)
            return self.norm_flow
        else:
            num_flow = y.shape[0].value
            if y in self.norm_flow_dict:
                return self.norm_flow_dict[y]
            # Construct the NormalizingFlow object.
            # determintes how many normalizing flow transforms in parallel is
            # needed.

            param_shape = self.norm_flow_type.get_param_shape(
                    dim=self.out_dim, **self.norm_flow_params)

            if self.norm_flow_type is TimeAutoRegressivePlanarFlow:
                # the case of auto-regressive flow of has a specific treatment
                # of input mapping to parameters of the flow.
                trans_out_dim = param_shape[0] * param_shape[2] * param_shape[4]
                # TODO: Check whether the input dimension is correct which is
                # (# examples, # time steps, space dimensionality)
                time, space_dim = self.in_dim
                # Two consecutive times
                trans_in_dim = space_dim * 2 
                transform = self.transform_type(
                        in_dim=trans_in_dim, out_dim=trans_out_dim,
                        **self.transform_params)
                # reshape input into consecutive time points
                # turn y shape into time_consecutive tensor
                gov_params = transform.operator(
                        tf.concat([y[:, :-1], y[:, 1:]], axis=-1))
                gov_params = tf.reshape(
                        gov_params,
                        (num_flow, param_shape[1]) + param_shape[:1] + param_shape[2:])
                gov_params = tf.transpose(
                        gov_params, perm=[0, 2, 1, 3, 4, 5])
                # Construct each flow
                flows = []
                for i in range(num_flow):
                    flows.append(self.norm_flow_type(
                            dim=self.out_dim, gov_param=gov_params[i],
                            **self.norm_flow_params))
                self.norm_flow_dict[y] = flows
                return flows

            # The other straight forward cases
            # Compute the flattened parameter shape
            trans_out_dim = 1
            for dim in param_shape:
                trans_out_dim *= dim

            transform = self.transform_type(
                    in_dim=self.in_dim, out_dim=trans_out_dim,
                    **self.transform_params)
            flow_params = transform.operator(y)
            # List of flows for inputs respective to the input y.
            flows = []
            for i in range(num_flow):
                gov_param = tf.reshape(flow_params[i], param_shape)
                flows.append(self.norm_flow_type(
                        dim=self.out_dim, gov_param=gov_param,
                        **self.norm_flow_params))
            self.norm_flow_dict[y] = flows
            return flows

    def sample(self, n_samples, y=None):
        """Samples from the model and computes log probability of the samples.

        params:
        -------
        y: tf.Tensor
            If model is conditional y must be None; Otherwise, a tf.Tensor.

        returns:
        --------
        tuple of tf.Tensor. First member of the tuple is samples from model.
        The shape of the first model should be (n_samples, out_dim) or
        (n_samples, n_models/n_inputs, out_dim)
        Second member is the log-density of the samples of the model.
        """
        super(NormalizingFlowModel, self).sample(n_samples=n_samples, y=y)
        # How many parallel models/flows are needed.
        num_flows = 0
        if y is not None:
            num_flows = y.shape[0].value
        flows = self.get_normalizing_flow(y)

        if isinstance(self.base_model, Model):
            # Get sample and log prob according to the design of model.
            samples_not = self.base_model.sample(n_samples=n_samples, y=y)
            log_prob = self.base_model.log_prob(x=samples_not, y=y)

        else:
            # The base model is tf.distributions.Distribution
            if num_flows == 0:
                samples_not = self.base_model.sample(n_samples)
                log_prob = self.base_model.log_prob(samples_not)
            else:
                samples_not = self.base_model.sample([n_samples, num_flows])
                log_prob = self.base_model.log_prob(samples_not)

        if num_flows == 0:
            samples = flows.operator(samples_not)
            log_prob -= flows.log_det_jacobian(samples_not)

        else:
            # Multiple flows that have to be applied to samples.
            sample_list, log_p_list = [], []
            for i, flow in enumerate(flows):
                sample_list.append(
                        tf.expand_dims(flow.operator(samples_not[:, i]), axis=1))
                log_p_list.append(tf.expand_dims(
                        log_prob[:, i] - flow.log_det_jacobian(samples_not[:, i]),
                        axis=1))
            samples = tf.concat(sample_list, axis=1)
            log_prob_ = tf.concat(log_p_list, axis=1)

        return samples, log_prob

