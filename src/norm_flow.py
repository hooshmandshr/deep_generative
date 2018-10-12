import numpy as np
import tensorflow as tf

from transform import Transform


class NormalizingFlow(Transform):

    def __init__(self, dim, gov_param=None, initial_value=None, name=None):
        """Sets up the universal properties of any transformation function.

        Governing parameters of the transformation is set in the constructor.

        params:
        -------
        dim: int
            dimensionality of the input code/variable and output variable.
        gov_param: tf.Tensor
            In case that parameters of the transformation are governed by
            another tensor.
        initial_value: numpy.ndarray
            Initial value of the transformation variable.
        """
        super(NormalizingFlow, self).__init__(in_dim=dim, out_dim=dim,
                gov_param=gov_param, initial_value=initial_value, name=name)
        self.dim = dim

    def log_det_jacobian(self, x):
        """Given x the log determinent Jacobian of the matrix.

        returns:
        --------
        tf.Tensor for log-det-jacobian of the transformation given x.
        """
        pass


class PlanarFlow(NormalizingFlow):

    def __init__(self, dim, num=1, non_linearity=tf.tanh, gov_param=None,
            initial_value=None, name=None):
        """Sets up the universal properties of any transformation function.

        Governing parameters of the transformation is set in the constructor.

        params:
        -------
        dim: int
            dimensionality of the input code/variable and output variable.
        num: int
            Number of total different normalizing flows to be applied at once.
        gov_param: tf.Tensor
            In case that parameters of the transformation are governed by
            another tensor.
        initial_value: numpy.ndarray
            Initial value of the transformation variable.
        """
        super(PlanarFlow, self).__init__(dim=dim, gov_param=gov_param,
                initial_value=initial_value, name=name)
        # Set the rest of the attributes.
        self.non_linearity = non_linearity
        self.num = num
        # Make sure the shape of the parameters is correct.
        self.param_shape = (self.num, 2 * self.dim + 1)
        self.check_param_shape()

        # Partition the variable into variables of the planar flow.
        self.w = tf.slice(self.var, [0, 0], [-1, self.dim])
        self.u = tf.slice(self.var, [0, self.dim], [-1, self.dim])
        self.b = tf.slice(self.var, [0, 2 * self.dim], [-1, 1])
        # Guarantee invertibility of the forward transform.
        #self.enforce_invertiblity()
        self.u_bar = self.u
        # Tensor map for keeping track of computation redundancy.
        # If an operation on a tensor has been done before, do not redo it.
        self.tensor_map = {}

    def enforce_invertiblity(self):
        """Guarantee that planar flow does not have 0 determinant Jacobian."""
        if self.non_linearity is tf.tanh:
            dot = tf.reduce_sum(
                    (self.u * self.w), axis=1, keepdims=True)
            scalar = - 1 + tf.nn.softplus(dot) - dot
            norm_squared = tf.reduce_sum(
                    self.w * self.w, axis=1, keepdims=True)
            comp = scalar * self.w / norm_squared
            self.u_bar = self.u + comp

    def non_linearity_derivative(self, x):
        """Operation for the derivative of the non linearity function."""
        if self.non_linearity is tf.tanh:
            return 1. - tf.tanh(x) * tf.tanh(x)

    def inner_prod(self, x):
        """Computes the inner product part of the transformation."""
        if x in self.tensor_map:
            return self.tensor_map[x]

        if self.num == 1:
            result = tf.matmul(x, self.w, transpose_b=True) + self.b
        else:
            result = tf.matmul(x, tf.expand_dims(self.w, 2)) + tf.expand_dims(
                    self.b, 2)
        self.tensor_map[x] = result
        return result

    def operator(self, x):
        """Given x applies the Planar flow transformation to the input.

        params:
        -------
        x: tf.Tensor
            if self.num is (the number of total flows) 1 the input tensor must
            have shape (None, self.dim). If self.num > 1 the shape of input
            must have (self.num, None, self.dim).
        """
        dial = self.inner_prod(x)
        if self.num == 1:
            result = x + self.u_bar * self.non_linearity(dial)
        else:
            result = x + tf.expand_dims(self.u_bar, 1) * self.non_linearity(
                    dial)
        return result

    def log_det_jacobian(self, x):
        """Computes log-det-Jacobian for combination of inputs, flows.

        params:
        -------
        x: tensorflow.Tensor
            if self.num is (the number of total flows) 1 the input tensor must
            have shape (None, self.dim). If self.num > 1 the shape of input
            must have (self.num, None, self.dim).

        returns:
        --------
        tf.Tensor for log-det-jacobian of the transformation given x.
        """
        dial = self.inner_prod(x)
        if self.num == 1:
            psi = self.non_linearity_derivative(dial)
            det_jac = tf.matmul(self.u_bar, self.w, transpose_b=True) * psi
            result = tf.squeeze(tf.log(tf.abs(1 + det_jac)))
        else:
            psi = self.non_linearity_derivative(dial)
            dot = tf.reduce_sum(self.u_bar * self.w, axis=1, keepdims=True)
            dot = tf.expand_dims(dot, 1)
            det_jac = dot * psi
            result = tf.squeeze(tf.log(tf.abs(1 + det_jac)))
        return result


class TimeAutoRegressivePlanarFlow(NormalizingFlow):
    """Class that implements PlanarFlow auto-regressive transform."""

    def __init__(self, dim, time, num=1, num_sweep=1, num_layer=1,
            non_linearity=tf.tanh, gov_param=None, initial_value=None,
            name=None):
        """Sets up the universal properties of any transformation function.

        Governing parameters of the transformation is set in the constructor.

        params:
        -------
        dim: int
            dimensionality of the input code/variable and output variable.
        time: int
            Number of time steps in the process that produces a single
            trajectory/path of the random variable.
        num: int
            Number of total different normalizing flows to be applied at once.
        num_sweep: int
            Number of total sweeps of the entire time components to be done.
        num_layer: int
            Number of planar flow layer that is applied to time subsets.
        gov_param: tf.Tensor
            In case that parameters of the transformation are governed by
            another tensor.
        initial_value: numpy.ndarray
            Initial value of the transformation variable.
        """
        super(TimeAutoRegressivePlanarFlow, self).__init__(
                dim=dim, gov_param=gov_param, initial_value=initial_value,
                name=name)
        # Set the rest of the attributes.
        self.non_linearity = non_linearity
        self.num = num
        self.time = time
        self.num_layer = num_layer
        self.num_sweep = num_sweep
        # Make sure the shape of the parameters is correct.
        self.param_shape = (self.num_sweep, self.time - 1, self.num_layer,
                self.num, 2 * 2 * self.dim + 1)
        self.check_param_shape()
        # Storing all the individual flows in this structure.
        self.set_up_sweep_layer_flows()
        # Stores the tensor result of log det given inputs to avoid
        # redundancy of computation.
        self.log_det_jac_map = {}
        # TODO: Do the same for the result of the transformation.

    def set_up_sweep_layer_flows(self):
        """Sets up the necessary flow transformation for each time slice."""
        self.sweep_time_layer_flow = []
        for sweep in range(self.num_sweep):
            self.sweep_time_layer_flow.append([])
            for time in range(self.time - 1):
                self.sweep_time_layer_flow[-1].append([])
                for layer in range(self.num_layer):
                    gov_param = None
                    init_value = None
                    if self.var is not None:
                        gov_param = self.var[sweep, time, layer, :, :]
                    if self.initial_value is not None:
                        init_value = self.initial_value[sweep, time, layer, :, :]
                    flow = PlanarFlow(
                            dim=self.dim * 2, num=self.num,
                            non_linearity=self.non_linearity,
                            gov_param=gov_param, initial_value=init_value)
                    self.sweep_time_layer_flow[-1][-1].append(flow)

    def check_input_shape(self, x):
        """Checks that the input shape is compatible with the flow.

        params:
        -------
        x: tf.Tensor:
            Input tensor that the series of transformation will be applied to.
        """
        # Shape of the input tensor.
        s = x.shape
        # Last two dimensions of the expected shape.
        l_dim = (self.time, self.dim)
        if self.num == 1:
            if not (len(s) == 3 and s[1:] == l_dim):
                raise ValueError("Input shape must be (?, time, dim).")
        else:
            if not (len(s) == 4 and s[2:] == l_dim and s[0].value == self.num):
                raise ValueError("Input shape must be (num, ?, time, dim).")

    def time_slice(self, x):
        """Get time slice of input for a particular time.

        params:
        -------
        x: tf.Tensor
            Input tensor which is time ordered.
        """
        if self.num == 1:
            return [x[:, t, :] for t in range(self.time)]
        else:
            return [x[:, :, t, :] for t in range(self.time)]

    def stitch_time_slice(self, x):
        """Stitch time sliced input/output together into a single tensor.

        params:
        -------
        x: list of tf.Tensor
            Each member of the list has either shape (num, ?, dim) or (?, dim).
        """
        expand_axis = 1
        if self.num > 1:
            expand_axis = 2
        return tf.concat(
                [tf.expand_dims(y, axis=expand_axis) for y in x],
                axis=expand_axis)

    def operator(self, x):
        """Transforming x according to time autoregressive normalizing flow.

        params:
        -------
        x: tf.Tensor
            Must have shape either (num, ?, time, dim) or (?, time, dim).
        returns:
        -------
        tf.Tensor. Holds the transformed input that has the same shape as the
        input.
        """
        # holds the final result.
        self.check_input_shape(x)
        # Keeps intermediate transformation results of x sliced by time.
        # By the end this list holds the time sliced result.
        time_slice = self.time_slice(x)
        log_det_jac = 0.

        for sweep in range(self.num_sweep):
            for time in range(self.time - 1):
                # Grap two consecutive time dimensions
                time_subset = tf.concat(time_slice[time:time + 2], axis=-1)
                for layer in range(self.num_layer):
                    flow = self.sweep_time_layer_flow[sweep][time][layer]
                    time_subset = flow.operator(time_subset)
                    log_det_jac += flow.log_det_jacobian(time_subset )
                if self.num == 1:
                    time_slice[time] = time_subset[:, :self.dim]
                    time_slice[time + 1] = time_subset[:, self.dim:]
                else:
                    time_slice[time] = time_subset[:, :, :self.dim]
                    time_slice[time + 1] = time_subset[:, :, self.dim:]
        # Stitch the time slices back together into a single tensor.
        self.log_det_jac_map[x] = log_det_jac
        return self.stitch_time_slice(time_slice)

    def log_det_jacobian(self, x):
        """Computes log-det-jacobian of the transformation given an input.

        params:
        -------
        x: tf.Tensor
            Must have shape either (num, ?, time, dim) or (?, time, dim).
        returns:
        -------
        tf.Tensor. Holds the log-det-jacobian of the transformation given x.
        It has shape (num, ?) or (?,) depending on the input shape.
        """
        if x in self.log_det_jac_map:
            return self.log_det_jac_map[x]
        self.operator(x)
        return self.log_det_jac_map[x]

