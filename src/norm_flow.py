import numpy as np
import tensorflow as tf

from transform import Transform
from utils.block_matrix import BlockBiDiagonalMatrix


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

    @staticmethod
    def get_param_shape(**kwargs):
        """Gets the shape of the governing parameters of the transform."""
        raise NotImplementedError(
                "Abstract method that each sub-class has to implement.")

class PlanarFlow(NormalizingFlow):

    def __init__(self, dim, non_linearity=tf.tanh, gov_param=None,
            initial_value=None, enforce_inverse=True, name=None):
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
        enforce_inverse: bool
            If true, the parameters are changed slightly to guarantee
            invertibility.
        """
        super(PlanarFlow, self).__init__(dim=dim, gov_param=gov_param,
                initial_value=initial_value, name=name)
        # Set the rest of the attributes.
        self.non_linearity = non_linearity
        # Make sure the shape of the parameters is correct.
        self.param_shape = PlanarFlow.get_param_shape(dim=dim)
        self.check_param_shape()

        # Partition the variable into variables of the planar flow.
        self.w = tf.slice(self.var, [0, 0], [-1, self.dim])
        self.u = tf.slice(self.var, [0, self.dim], [-1, self.dim])
        self.b = tf.slice(self.var, [0, 2 * self.dim], [-1, 1])
        # Guarantee invertibility of the forward transform.
        self.u_bar = self.u
        if enforce_inverse:
            self.enforce_invertiblity()

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

        result = tf.matmul(x, self.w, transpose_b=True) + self.b

        self.tensor_map[x] = result
        return result

    def operator(self, x):
        """Given x applies the Planar flow transformation to the input.

        params:
        -------
        x: tf.Tensor
            Input tensor for which the transformation is computed.

        returns:
        --------
        tf.Tensor.
        """
        dial = self.inner_prod(x)
        result = x + self.u_bar * self.non_linearity(dial)
        return result

    def log_det_jacobian(self, x):
        """Computes log-det-Jacobian for combination of inputs, flows.

        params:
        -------
        x: tensorflow.Tensor
            Input tensor for which the log-det-jacobian is computed.

        returns:
        --------
        tf.Tensor for log-det-jacobian of the transformation given x.
        """
        dial = self.inner_prod(x)

        psi = self.non_linearity_derivative(dial)
        det_jac = tf.matmul(self.u_bar, self.w, transpose_b=True) * psi
        result = tf.squeeze(tf.log(tf.abs(1 + det_jac)))

        return result

    def initializer(self):
        """Default initializer of the transformation class."""
        init_val = np.zeros(self.param_shape)
        # Xaviar initializer unifrom for w parameter.
        lim = np.sqrt(6. / (1. + self.dim))
        init_val[:, :self.dim] = np.random.uniform(-lim, lim, self.dim)
        self.var = tf.Variable(init_val)

    @staticmethod
    def get_param_shape(**kwargs):
        """Gets the shape of the governing parameters of the transform."""
        dim = kwargs["dim"]
        return (1, 2 * dim + 1)


class MultiLayerPlanarFlow(NormalizingFlow):

    def __init__(self, dim, num_layer, non_linearity=tf.tanh,
            gov_param=None, initial_value=None, name=None):
        """Sets up the universal properties of any transformation function.

        Governing parameters of the transformation is set in the constructor.

        params:
        -------
        dim: int
            dimensionality of the input code/variable and output variable.
        num_layer: int
            Number of successive layers of palanr flow transformation.
        gov_param: tf.Tensor
            In case that parameters of the transformation are governed by
            another tensor.
        initial_value: numpy.ndarray
            Initial value of the transformation variable.
        enforce_inverse: bool
            If true, the parameters are changed slightly to guarantee
            invertibility.
        """
        super(MultiLayerPlanarFlow, self).__init__(dim=dim, gov_param=gov_param,
                initial_value=initial_value, name=name)
        # Set the rest of the attributes.
        self.non_linearity = non_linearity
        self.num_layer = num_layer
        # Make sure the shape of the parameters is correct.
        #self.param_shape = (self.num_layer, self.num, 2 * self.dim + 1)
        self.param_shape = MultiLayerPlanarFlow.get_param_shape(
                num_layer=self.num_layer, dim=self.dim)
        self.check_param_shape()
        # Create a flow transform for every layer.
        self.layers = []
        for i in range(self.num_layer):
            layer_gov_param = self.var[i]
            self.layers.append(PlanarFlow(
                dim=self.dim, non_linearity=non_linearity,
                gov_param=layer_gov_param, name=name))

    def operator(self, x):
        """Given x applies the Planar flow transformation to the input.

        params:
        -------
        x: tf.Tensor

        """
        result = x
        for layer in self.layers:
            result = layer.operator(result)
        return result

    def log_det_jacobian(self, x):
        """Computes log-det-Jacobian for combination of inputs, flows.

        params:
        -------
        x: tensorflow.Tensor

        returns:
        --------
        tf.Tensor for log-det-jacobian of the transformation given x.
        """
        result = 0.
        # Tensor for intermediate transformation results.
        inter_trans = x
        for layer in self.layers:
            result += layer.log_det_jacobian(inter_trans)
            inter_trans = layer.operator(inter_trans)
        return result

    def initializer(self):
        """Default initializer of the transformation class."""
        init_val = np.zeros(self.param_shape)
        # Xaviar initializer unifrom for w parameter.
        lim = np.sqrt(6. / (1. + self.dim))
        init_val[:, :, :self.dim] = np.random.uniform(
                -lim, lim, [self.num_layer, 1, self.dim])
        self.var = tf.Variable(init_val)

    @staticmethod
    def get_param_shape(**kwargs):
        """Gets the shape of the governing parameters of the transform.

        Parameters:
        -----------
        **kwargs:
            Paramters of the constructor that must include the following:
                dim,
                num,
                num_layer
        """
        num_layer, dim = kwargs["num_layer"], kwargs["dim"]

        return (num_layer, 1, 2 * dim + 1)


class TimeAutoRegressivePlanarFlow(NormalizingFlow):
    """Class that implements PlanarFlow auto-regressive transform."""

    def __init__(self, dim, num_sweep=1, num_layer=1,
            non_linearity=tf.tanh, gov_param=None, initial_value=None,
            backward=False, name=None):
        """Sets up the universal properties of any transformation function.

        Governing parameters of the transformation is set in the constructor.

        params:
        -------
        dim: tuple of int
            First, dimensionality of the input code/variable and output
            variable. The second member is the number of time steps in the
            process that produces a single trajectory/path of the random
            variable.
        num_sweep: int
            Number of total sweeps of the entire time components to be done.
        num_layer: int
            Number of planar flow layer that is applied to time subsets.
        gov_param: tf.Tensor
            In case that parameters of the transformation are governed by
            another tensor.
        initial_value: numpy.ndarray
            Initial value of the transformation variable.
        backward: bool
            If True, a backward pass is made on the latent variables.
        """
        super(TimeAutoRegressivePlanarFlow, self).__init__(
                dim=dim, gov_param=gov_param, initial_value=initial_value,
                name=name)
        # Set the rest of the attributes.
        self.non_linearity = non_linearity
        self.time, self.space_dim = self.dim
        self.num_layer = num_layer
        self.num_sweep = num_sweep
        self.backward = backward
        self.backward_loop = 1
        if backward:
            self.backward_loop = 2
        # Make sure the shape of the parameters is correct.
        self.param_shape = self.get_param_shape(
                dim=dim, num_layer=num_layer, num_sweep=num_sweep,
                backward=backward)
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

        for fb_idx in range(self.backward_loop):
            self.sweep_time_layer_flow.append([])
            for sweep in range(self.num_sweep):
                self.sweep_time_layer_flow[-1].append([])
                for time in range(self.time - 1):
                    self.sweep_time_layer_flow[-1][-1].append([])
                    for layer in range(self.num_layer):
                        gov_param = None
                        init_value = None
                        if self.var is not None:
                            gov_param = self.var[fb_idx, sweep, time, layer, :, :]
                        if self.initial_value is not None:
                            init_value = self.initial_value[fb_idx, sweep, time, layer, :, :]
                        flow = PlanarFlow(
                                dim=self.space_dim * 2,
                                non_linearity=self.non_linearity,
                                gov_param=gov_param, initial_value=init_value)
                        self.sweep_time_layer_flow[-1][-1][-1].append(flow)

    @staticmethod
    def get_param_shape(**kwargs):
        """Gets the shape of the governing parameters of the transform.

        Parameters:
        -----------
        **kwargs:
            Paramters of the constructor that must include the following:
                dim,
                num_layer
        """
        time, dim = kwargs["dim"]
        # Set with default values in case not given
        num_sweep = 1
        num_layer = 1
        num_fb = 1
        if "num_sweep" in kwargs:
            num_sweep = kwargs["num_sweep"]
        if "num_layer" in kwargs:
            num_layer = kwargs["num_layer"]
        if "backward" in kwargs:
            if kwargs["backward"]:
                num_fb = 2

        return (num_fb, num_sweep, time - 1, num_layer, 1, 2 * 2 * dim + 1)

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

        if not (len(s) == 3 and s[1:] == self.dim):
            raise ValueError("Input shape must be (?, time, space dim).")

    def time_slice(self, x):
        """Get time slice of input for a particular time.

        params:
        -------
        x: tf.Tensor
            Input tensor which is time ordered.
        """
        return [x[:, t, :] for t in range(self.time)]

    def stitch_time_slice(self, x):
        """Stitch time sliced input/output together into a single tensor.

        params:
        -------
        x: list of tf.Tensor
            Each member of the list has either shape (num, ?, dim) or (?, dim).
        """
        expand_axis = 1
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

        # For foward pass
        time_progression = [range(self.time - 1)]
        # For backward pass
        time_progression.append(range(self.time - 1)[::-1])

        for fb_idx in range(self.backward_loop):
            for sweep in range(self.num_sweep):
                for time in time_progression[fb_idx]:
                    # Grap two consecutive time dimensions
                    time_subset = tf.concat(time_slice[time:time + 2], axis=-1)
                    for layer in range(self.num_layer):
                        flow = self.sweep_time_layer_flow[fb_idx][sweep][time][layer]
                        time_subset = flow.operator(time_subset)
                        log_det_jac += flow.log_det_jacobian(time_subset )

                    time_slice[time] = time_subset[:, :self.space_dim]
                    time_slice[time + 1] = time_subset[:, self.space_dim:]
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


class AffineFlow(NormalizingFlow):

    def __init__(self, dim, non_linearity=tf.tanh, n_comp=1, gov_param=None,
            initial_value=None, enforce_inverse=True, lower=True, name=None):
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
        enforce_inverse: bool
            If true, the parameters are changed slightly to guarantee
            invertibility.
        lower: bool
            If True the a_matrix is lower triangular, otherwise it will be
            upper triangular.
        """
        super(AffineFlow, self).__init__(dim=dim, gov_param=gov_param,
                initial_value=initial_value, name=name)
        # Set the rest of the attributes.
        nl = non_linearity
        if not(nl is tf.tanh or nl is tf.nn.sigmoid or nl is tf.nn.softplus):
            raise NotImplemented(
                    "Only {}, {}, {} non-linearities are compatible.".format(
                        "tf.tanh", "tf.nn.sigmoid", "tf.nn.softplus"))

        self.non_linearity = non_linearity
        self.n_comp = n_comp
        # Make sure the shape of the parameters is correct.
        self.param_shape = AffineFlow.get_param_shape(dim=dim, n_comp=n_comp)
        self.check_param_shape()

        # Partition the variable into variables of the planar flow.
        self.lower = lower
        self.a_matrix = self.var[:, :dim]
        self.bias = self.var[:, dim:]

        if enforce_inverse:
            self.enforce_invertiblity()

        # Tensor map for keeping track of computation redundancy.
        # If an operation on a tensor has been done before, do not redo it.
        self.tensor_map = {}

    def enforce_invertiblity(self):
        """Guarantee that affine flow does not have 0 determinant Jacobian."""
        if self.lower:
            self.a_matrix = tf.matrix_band_part(self.a_matrix, 0, -1)
        else:
            self.a_matrix = tf.matrix_band_part(self.a_matrix, -1, 0)

        # Ensure that the diagonal elements of the matrix are all
        # bigger that -1/N where N is total parallel affine transformations
        threshold = 1.
        if self.non_linearity is tf.nn.sigmoid:
            threshold = 2.
        diag_part = tf.matrix_diag_part(self.a_matrix)
        self.a_matrix += - tf.matrix_diag(diag_part) + tf.matrix_diag(
                tf.nn.softplus(diag_part) - threshold / self.n_comp)

    def non_linearity_derivative(self, x):
        """Operation for the derivative of the non linearity function."""
        if self.non_linearity is tf.tanh:
            return 1. - tf.square(tf.tanh(x))
        elif self.non_linearity is tf.nn.sigmoid:
            return (1. - tf.square(tf.tanh(x))) / 2.
        elif self.non_linearity is tf.nn.softplus:
            return tf.nn.sigmoid(x)

    def matmul(self, x):
        """Computes the inner product part of the transformation."""
        if x in self.tensor_map:
            return self.tensor_map[x]

        x_ = tf.expand_dims(x, axis=0)
        x_ = tf.concat([x_ for i in range(self.n_comp)], axis=0)
        result = tf.matmul(x_, self.a_matrix) + self.bias

        self.tensor_map[x] = result
        return result

    def operator(self, x):
        """Given x applies the Planar flow transformation to the input.

        params:
        -------
        x: tf.Tensor
            Input tensor for which the transformation is computed.

        returns:
        --------
        tf.Tensor.
        """
        dial = self.matmul(x)
        result = x + tf.reduce_sum(self.non_linearity(dial), axis=0)
        return result

    def log_det_jacobian(self, x):
        """Computes log-det-Jacobian for combination of inputs, flows.

        params:
        -------
        x: tensorflow.Tensor
            Input tensor for which the log-det-jacobian is computed.

        returns:
        --------
        tf.Tensor for log-det-jacobian of the transformation given x.
        """
        dial = self.non_linearity_derivative(self.matmul(x))

        result = tf.reduce_sum(
                tf.log(
                    1. + tf.reduce_sum(tf.expand_dims(
                        tf.matrix_diag_part(self.a_matrix), axis=1) * dial, axis=0)),
                axis=1)
        return result

    def initializer(self):
        """Default initializer of the transformation class."""

        # Xaviar initializer unifrom for w parameter.
        std = np.sqrt(2. / (2 * self.dim))
        init_val = np.random.normal(0, std, self.param_shape)
        init_val[:, -1, :] *= 0.
        self.var = tf.Variable(init_val)

    @staticmethod
    def get_param_shape(**kwargs):
        """Gets the shape of the governing parameters of the transform."""
        dim = kwargs["dim"]
        n_comp = kwargs["n_comp"]
        return (n_comp, dim + 1, dim)


class KalmanFlow(NormalizingFlow):

    def __init__(self, time, dim, non_linearity=tf.tanh, gov_param=None,
            initial_value=None, enforce_inverse=True, lower=True, name=None):
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
        enforce_inverse: bool
            If true, the parameters are changed slightly to guarantee
            invertibility.
        lower: bool
            If True the a_matrix is lower triangular, otherwise it will be
            upper triangular.
        """
        self.time = time
        super(KalmanFlow, self).__init__(dim=dim, gov_param=gov_param,
                initial_value=initial_value, name=name)
        # Set the rest of the attributes.
        nl = non_linearity
        if not(nl is tf.tanh or nl is tf.nn.sigmoid or nl is tf.nn.softplus):
            raise NotImplemented(
                    "Only {}, {}, {} non-linearities are compatible.".format(
                        "tf.tanh", "tf.nn.sigmoid", "tf.nn.softplus"))

        self.non_linearity = non_linearity
        # Make sure the shape of the parameters is correct.
        self.param_shape = KalmanFlow.get_param_shape(time=time, dim=dim)
        self.check_param_shape()

        # Partition the variable into variables of the planar flow.

        self.lower = lower
        dsqr = dim * dim
        self.diag_ = tf.reshape(
                self.var[:, :dsqr], [time, dim, dim])
        if enforce_inverse:
            self.enforce_invertiblity()
        offdiag_ = tf.reshape(
                self.var[:(time - 1), dsqr:(2 * dsqr)],
                [time - 1, dim, dim])
        self.a_matrix = BlockBiDiagonalMatrix(
                diag_block=self.diag_, offdiag_block=offdiag_, lower=self.lower)
        self.bias = tf.expand_dims(
                self.var[:, (2 * dsqr):(2 * dsqr + dim)], axis=0)
        # Scale must be > -1 for the log-det-jac to be defined
        self.scale = tf.expand_dims(tf.nn.softplus(
            self.var[:, (2 * dsqr + dim):(2 * (dsqr + dim))]) - 1., axis=0)
        # Tensor map for keeping track of computation redundancy.
        # If an operation on a tensor has been done before, do not redo it.
        self.tensor_map = {}

    def enforce_invertiblity(self):
        """Guarantee that affine flow does not have 0 determinant Jacobian."""
        if self.lower:
            self.diag_ = tf.matrix_band_part(self.diag_, -1, 0)

        # Ensure that the diagonal elements of the matrix are all
        # bigger that -1/N where N is total parallel affine transformations

        diag_part = tf.matrix_diag_part(self.diag_)
        self.diag_ += - tf.matrix_diag(diag_part) + tf.matrix_diag(
                tf.nn.softplus(diag_part))

    def non_linearity_derivative(self, x):
        """Operation for the derivative of the non linearity function."""
        if self.non_linearity is tf.tanh:
            return 1. - tf.square(tf.tanh(x))
        elif self.non_linearity is tf.nn.sigmoid:
            return (1. - tf.square(tf.tanh(x))) / 2.
        elif self.non_linearity is tf.nn.softplus:
            return tf.nn.sigmoid(x)

    def matmul(self, x):
        """Computes the inner product part of the transformation."""
        if x in self.tensor_map:
            return self.tensor_map[x]
        result = self.a_matrix.solve(x) + self.bias
        return result

    def operator(self, x):
        """Given x applies the Planar flow transformation to the input.

        params:
        -------
        x: tf.Tensor
            Input tensor for which the transformation is computed.

        returns:
        --------
        tf.Tensor.
        """
        dial = self.matmul(x)
        result = x + self.scale * self.non_linearity(dial)
        return result

    def log_det_jacobian(self, x):
        """Computes log-det-Jacobian for combination of inputs, flows.

        params:
        -------
        x: tensorflow.Tensor
            Input tensor for which the log-det-jacobian is computed.

        returns:
        --------
        tf.Tensor for log-det-jacobian of the transformation given x.
        """
        dial = self.non_linearity_derivative(self.matmul(x))

        diag = tf.expand_dims(tf.matrix_diag_part(self.diag_), axis=0)
        result = tf.reduce_sum(tf.reduce_sum(
                tf.log(diag + dial * self.scale), axis=1), axis=1) - tf.reduce_sum(
                        tf.log(diag))
        return result

    def initializer(self):
        """Default initializer of the transformation class."""

        # Xaviar initializer unifrom for w parameter.
        std = np.sqrt(2. / (2 * self.time * self.dim))
        init_val = np.random.normal(0, std, self.param_shape)
        # Bias initialization
        dsqr = self.dim * self.dim
        init_val[:, (2 * dsqr):] *= 0.
        # Scale initilization
        init_val[:, (2 * dsqr + self.dim):] += 1.
        self.var = tf.Variable(init_val)

    @staticmethod
    def get_param_shape(**kwargs):
        """Gets the shape of the governing parameters of the transform."""
        dim = kwargs["dim"]
        time = kwargs["time"]
        return (time, 2 * (dim * dim + dim))


class MultiLayerKalmanFlow(NormalizingFlow):

    def __init__(self, time, dim, n_layer, non_linearity=tf.tanh,
            gov_param=None, lower=True, name=None):
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
        enforce_inverse: bool
            If true, the parameters are changed slightly to guarantee
            invertibility.
        lower: bool
            If True the a_matrix is lower triangular, otherwise it will be
            upper triangular.
        """
        self.time = time
        super(MultiLayerKalmanFlow, self).__init__(dim=dim,
                gov_param=gov_param, name=name)
        # Set the rest of the attributes.
        nl = non_linearity
        if not(nl is tf.tanh or nl is tf.nn.sigmoid or nl is tf.nn.softplus):
            raise NotImplemented(
                    "Only {}, {}, {} non-linearities are compatible.".format(
                        "tf.tanh", "tf.nn.sigmoid", "tf.nn.softplus"))

        self.non_linearity = non_linearity
        # Make sure the shape of the parameters is correct.
        self.param_shape = MultiLayerKalmanFlow.get_param_shape(
                time=time, dim=dim, n_layer=n_layer)
        self.check_param_shape()

        # Partition the variable into variables of the planar flow.
        self.n_layer = n_layer
        self.flows = []
        for i in range(self.n_layer):
            self.flows.append(
                    KalmanFlow(time=time, dim=dim, non_linearity=non_linearity,
                        gov_param=self.var[i], lower=lower, name=name))
        # Dictionary for keeping precomputed results
        self.mid_transform = {}

    def operator(self, x):
        """Given x applies the Planar flow transformation to the input.

        params:
        -------
        x: tf.Tensor
            Input tensor for which the transformation is computed.

        returns:
        --------
        tf.Tensor.
        """
        if x in self.mid_transform:
            # The result is at the end of the final level of already
            # computed transform.
            return self.mid_transform[x][-1]

        self.mid_transform[x] = [x]
        result = x
        for flow in self.flows:
            result = flow.operator(result)
            self.mid_transform[x].append(result)
        return result

    def log_det_jacobian(self, x):
        """Computes log-det-Jacobian for combination of inputs, flows.

        params:
        -------
        x: tensorflow.Tensor
            Input tensor for which the log-det-jacobian is computed.

        returns:
        --------
        tf.Tensor for log-det-jacobian of the transformation given x.
        """
        result = 0.
        if x not in self.mid_transform:
            # Compute the mid level transforms first.
            self.operator(x)
        for i, flow in enumerate(self.flows):
            result += flow.log_det_jacobian(self.mid_transform[x][i])
        return result

    def initializer(self):
        """Default initializer of the transformation class."""

        # Xaviar initializer unifrom for w parameter.
        std = np.sqrt(2. / (2 * self.time * self.dim))
        init_val = np.random.normal(0, std, self.param_shape)
        # Bias initialization
        dsqr = self.dim * self.dim
        init_val[..., (2 * dsqr):] *= 0.
        # Scale initilization
        init_val[..., (2 * dsqr + self.dim):] += 1.
        self.var = tf.Variable(init_val)

    @staticmethod
    def get_param_shape(**kwargs):
        """Gets the shape of the governing parameters of the transform."""
        dim = kwargs["dim"]
        time = kwargs["time"]
        n_layer = kwargs["n_layer"]
        return (n_layer, time, 2 * (dim * dim + dim))
