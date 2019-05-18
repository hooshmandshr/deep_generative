"""Class for implementation of major transformations."""


import numpy as np
import tensorflow as tf


class Transform(object):

    def __init__(self, in_dim, out_dim, gov_param=None, initial_value=None,
            name=None):
        """Sets up the universal properties of any transformation function.

        Governing parameters of the transformation is set in the constructor.

        params:
        -------
        in_dim: int
            dimensionality of the input code/variable.
        out_dim: int
            dimensionality of the output code/variable.
        gov_param: tf.Tensor
            In case that parameters of the transformation are governed by
            another tensor.
        initial_value: numpy.ndarray
            Initial value of the transformation variable.
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.initial_value = initial_value
        if gov_param is not None and not isinstance(gov_param, tf.Tensor) and\
                not isinstance(gov_param, tf.Variable):
            raise ValueError(
                    'Governing parameters of transformation should be Tensor.')
        self.var = gov_param
        # Check that parameters are tensors.
        # The following variable have to be set in the constructor of each
        # sub-class. Param shape is the correct shape of the parameters of the
        # Transformation. This has to be a tuple.
        self.param_shape = None
        self.name = name

    def initializer(self):
        """Default initializer of the transformation class."""
        self.var = tf.Variable(np.random.normal(0, 1, self.param_shape))

    def check_param_shape(self):
        """Checks the shape of the governing parameters or init values.

        It also initializes the variables (i.e. parameters) if necessary.
        """
        if self.var is not None:
            if not self.var.shape == self.param_shape:
                msg = "gov_param tensor's shape must be {}".format(
                        self.param_shape)
                msg += " instead it is {}".format(self.var.shape)
                raise ValueError(msg)
        elif self.initial_value is None:
            self.initializer()
        else:
            if not self.initial_value.shape == self.param_shape:
                raise ValueError("initial_value's shape must be {}.".format(
                    self.param_shape))
            self.var = tf.Variable(self.initial_value)

    def check_input_shape(self, x):
        """Checks whether the input has valid shape."""
        if not x.shape[-1] == self.in_dim:
            raise ValueError(
                    "Input must have dimension {}.".format(self.in_dim))

    def broadcast_operator(self, x):
        """Input of higher dimensions the operation will give same shape."""
        # Reshape the input array into 2 dimensions.
        input_ = x
        n_tot = 1
        for dim in x.shape[:-1]:
            n_tot *= dim.value
        input_ = tf.reshape(x, [n_tot, self.in_dim])
        output_ = tf.reshape(
                self.operator(input_), x.shape[:-1].as_list() + [self.out_dim])
        return output_

    def get_in_dim(self):
        return self.in_dim

    def get_out_dim(self):
        return self.out_dim

    def get_transformation_parameters(self):
        return self.var

    def get_name(self):
        return self.name

    def get_regularizer(self, scale=1.):
        """Computes the regularization of the parameter to be added to loss.

        returns:
        --------
        tensorflow.Tensor containing the regualization of the variables.
        """
        pass

    def operator(self, x):
        """Gives the tensorflow operation for transforming a tensor."""
        pass


class LinearTransform(Transform):

    def __init__(self, in_dim, out_dim, initial_value=None, gov_param=None,
            has_bias=True, non_linearity=None, name=None):
        """Sets up the linear transformation and bias variables.

        params:
        -------
        non_linearity: tf.Operation
            Type of non-linearity to be used for the output. In effect, turning
            LinearTransform to a fully connected layer with non-linear
            activations. E.g. tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh, etc. If
            None, the output is linear.
        """
        super(LinearTransform, self).__init__(
                in_dim, out_dim, gov_param=gov_param,
                initial_value=initial_value, name=name)
        # Make sure initialization and parameters are correct.
        self.param_shape = (self.in_dim + 1, out_dim)
        self.check_param_shape()

        # Partitioning the variable into respective variables of a linear
        # trnsformation.
        self.lin_trans = self.var[:-1]
        self.bias = self.var[-1]
        self.non_linearity = non_linearity
        self.has_bias = has_bias

    def initializer(self):
        """Overriden function to do Xavier initialization."""
        init_val = np.random.normal(
                0, np.sqrt(2. / (self.in_dim + self.out_dim)),
                self.param_shape)
        # Set bias to zero.
        init_val[-1] *= 0.
        self.var = tf.Variable(init_val)

    def get_regularizer(self, scale=1.):
        return scale * tf.reduce_sum(tf.reduce_sum(tf.square(self.lin_trans)))

    def operator(self, x):
        if len(x.shape) > 2:
            return self.broadcast_operator(x)
        self.check_input_shape(x)
        t_matrix = self.lin_trans
        bias = self.bias
        linear_output = tf.matmul(x, t_matrix)
        if self.has_bias:
            linear_output += bias
        if self.non_linearity is not None:
            return self.non_linearity(linear_output)
        return linear_output 


class LorenzTransform(Transform):

    def __init__(self, in_dim, out_dim,
            initial_value=None, gov_param=None, name=None,
            time_delta=0.03):
        """Sets up Euler discretization of Lorenz attractor."""
        super(LorenzTransform, self).__init__(
                in_dim=3, out_dim=3,
                initial_value=initial_value, gov_param=gov_param, name=name)

        self.param_shape = ((3,))
        self.check_param_shape()

        # Partition the variable into parameters of a Lorenz transfrom.
        self.sigma = self.var[0]
        self.rho = self.var[1]
        self.beta = self.var[2]
        self.time_delta = time_delta

    def operator(self, x):
        """Return a discretized Lorenz transformation."""
        if not(x.shape[-1].value == 3):
            raise ValueError('Dimension of variable should be 3')
        x_ = tf.slice(x, [0, 0], [-1, 1])
        y_ = tf.slice(x, [0, 1], [-1, 1])
        z_ = tf.slice(x, [0, 2], [-1, 1])
        return x + self.time_delta * tf.concat([
            self.sigma * (y_ - x_),
            x_ * (self.rho - z_) - y_,
            x_ * y_ - self.beta * z_], axis=1)


class RosslerTransform(Transform):

    def __init__(self, in_dim, out_dim,
            initial_value=None, gov_param=None, name=None,
            time_delta=0.03):
        """Sets up Euler discretization of Rossler attractor."""
        super(RosslerTransform, self).__init__(
                in_dim=3, out_dim=3,
                initial_value=initial_value, gov_param=gov_param, name=name)

        self.param_shape = ((3,))
        self.check_param_shape()

        # Partition the variable into parameters of a Lorenz transfrom.
        self.a = self.var[0]
        self.b = self.var[1]
        self.c = self.var[2]
        self.time_delta = time_delta

    def operator(self, x):
        """Return a discretized Lorenz transformation."""
        if not(x.shape[-1].value == 3):
            raise ValueError('Dimension of variable should be 3')
        x_ = x[:, 0:1]
        y_ = x[:, 1:2]
        z_ = x[:, 2:]
        return x + self.time_delta * tf.concat([
            y_ - z_,
            x_ + self.a * y_,
            self.b + z_ * (x_ - self.c)], axis=1)


class NagumoTransform(Transform):

    def __init__(self, in_dim, out_dim,
            initial_value=None, gov_param=None, name=None,
            time_delta=0.03):
        """Sets up Euler discretization of FitzHugh-Nagumo dynamics."""
        super(NagumoTransform, self).__init__(
                in_dim=2, out_dim=2,
                initial_value=initial_value, gov_param=gov_param, name=name)

        self.param_shape = ((4,))
        self.check_param_shape()

        # Partition the variable into parameters of a Lorenz transfrom.
        self.a = self.var[0]
        self.b = self.var[1]
        self.i_ext = self.var[2]
        self.tau = self.var[3]
        self.time_delta = time_delta

    def operator(self, x):
        """Return a discretized Lorenz transformation."""
        if not(x.shape[-1].value == 2):
            raise ValueError('Dimension of variable should be 2')
        v_ = x[:, 0:1]
        w_ = x[:, 1:2]
        return x + self.time_delta * tf.concat([
            v_ - tf.math.pow(v_, 3) / 3. - w_ + self.i_ext,
            (v_ + self.a - self.b * w_) / self.tau], axis=1)


class MultiLayerPerceptron(Transform):

    def __init__(self, in_dim, out_dim, hidden_units,
            activation=tf.nn.relu, residual=False, output_activation=None,
            drop_out_rate=None, name=None):
        """
        Sets up the layers of the MLP transformation.

        params:
        -------
        hidden_units: list of int
            Number of hidden units per hidden layer respectively.
        activation: tf.Operation
            Activation of each hidden layer in the network.
        output_activation: tf.Operation
            Output layer's non-linearity function.
        """
        super(MultiLayerPerceptron, self).__init__(
                in_dim=in_dim, out_dim=out_dim, name=name)

        if isinstance(activation, list):
            if not len(activation) == len(hidden_units):
                msg = "activation and hidden_units must be of same length."
                raise ValueError(msg)
        self.activation = activation
        self.out_activation = output_activation
        self.hidden_units = hidden_units
        # List that will containt the individual transformation for each layer.
        self.layers = []
        activ_ = self.activation
        for i, n_units in enumerate(hidden_units + [self.out_dim]):
            # Apply output layers non-linearity for if it is the last layer.
            if i == len(hidden_units):
                activ_ = self.out_activation
            elif isinstance(self.activation, list):
                activ_ = self.activation[i]
            layer_t = LinearTransform(
                    in_dim=in_dim, out_dim=n_units,
                    non_linearity=activ_)
            self.layers.append(layer_t)
            in_dim = n_units
        self.residual = residual
        self.drop_out_rate = drop_out_rate

    def get_regularizer(self, scale=1.):
        """Regularizer for the weights of the multilayer perceptron."""
        sum_all = 0.
        for layer in self.layers:
            sum_all += layer.get_regularizer()
        return sum_all

    def get_transformation_parameters(self):
        """Returns the list of variables of the layers of the MLP.

        returns:
        --------
        list of tensorflow.Tensor
        """
        if self.var is None:
            self.var = []
            for layer in self.layers:
                self.var.append(layer.var)
        return self.var

    def operator(self, x):
        if len(self.hidden_units) == 0:
            return x[..., :self.out_dim]
        output = x
        for i, layer in enumerate(self.layers):
            output = layer.operator(output)
            if self.drop_out_rate is not None and i <= len(self.hidden_units):
                # only for hidden_layers
                keep_prob = 1 - self.drop_out_rate
                output = tf.nn.dropout(output, keep_prob=keep_prob)
        if self.residual:
            return x + output
        return output


class LSTMcell(Transform):

    def __init__(self, in_dim, out_dim, gov_param=None, initial_value=None,
            name=None):
        """Sets up the layers of lstm.

        params:
        -------
        in_dim: int
            Dimensionality of input.
        out_dim: int
            Dimensionality of output.
        """
        super(LSTMcell, self).__init__(
                in_dim, out_dim, gov_param=gov_param,
                initial_value=initial_value, name=name)
        # Make sure initialization and parameters are correct.
        # W_f, U_f, b_f
        # W_i, U_i, b_i
        # W_o, U_o, b_o
        # W_c, U_c, b_c
        self.param_shape = (4*(in_dim + out_dim + 1), out_dim)
        self.check_param_shape()

        # Partitioning the variable into respective variables of a linear
        # trnsformation.
        idxs = np.array([0, self.in_dim, self.out_dim, 1]).cumsum()
        self.w = {}
        self.u = {}
        self.b = {}
        tot_dim = self.in_dim + self.out_dim + 1
        for node in ['f', 'i', 'o', 'c']:
            self.w[node] = self.var[idxs[0]:idxs[1]]
            self.u[node] = self.var[idxs[1]:idxs[2]]
            self.b[node] = self.var[idxs[2]:idxs[3]]
            idxs += tot_dim

    def initializer(self):
        """Overriden function to do Xavier initialization."""
        self.var = tf.Variable(np.random.normal(
                0, np.sqrt(2. / (self.in_dim + self.out_dim)),
                self.param_shape))

    def get_regularizer(self, scale=1.):
        return NotImplementedError("LSTM does not have.")

    def operator(self, x, h=None, c=None):
        """Operation that transforms the input and the previous state.

        params:
        -------
        x: tf.Tensor
            Shape is (?, in_dim). Current observation.
        h: tf.Tensor
            Shape is (?, out_dim). The previous state.

        returns:
        --------
        Tuple of tf.Tensor with shape (?, out_dim) which is the next state of the lstm.
        """
        if len(x.shape) == 3:
            # Ignores c and h and initializes to zero
            n_example = x.shape[1]
            n_time = x.shape[0]
            n_hidden_dim = self.out_dim
            # Input hidden state
            h = tf.constant(np.zeros([n_example, n_hidden_dim]))
            # Input error carousel state
            c = tf.constant(np.zeros([n_example, n_hidden_dim]))

            output_h = [h]
            output_c = [c]
            for t in range(n_time):
                hidden_t, carousel_t = self.operator(
                        x=x[t], h=output_h[t], c=output_c[t])
                output_h.append(hidden_t)
                output_c.append(carousel_t)

            return tf.concat(
                    [tf.expand_dims(h, axis=0) for h in output_h[1:]], axis=0)

        self.check_input_shape(x)
        comb = {}
        for node in ['f', 'i', 'o', 'c']:
            comb[node] = tf.matmul(
                    x,
                    self.w[node]) + tf.matmul(h, self.u[node]) + self.b[node]
            if node == 'c':
                comb[node] = tf.tanh(comb[node])
            else:
                comb[node] = tf.sigmoid(comb[node])

        c_star = comb['f'] * c + comb['i'] * comb['c']
        h_star = comb['o'] * tf.tanh(c_star)
        return h_star, c_star


class GatedTransition(Transform):

    def __init__(self, in_dim, hidden_units, name=None):
        """
        Sets up the layers of the MLP transformation.

        params:
        -------
        in_dim: int
        hidden_units: list of int
            Number of hidden units per hidden layer respectively.
        """
        super(GatedTransition, self).__init__(
                in_dim=in_dim, out_dim=in_dim, name=name)

        self.hidden_units = hidden_units

        # Initialize all the transformation that are needed.
        self.gate_transform = MultiLayerPerceptron(
                in_dim=self.in_dim, out_dim=self.in_dim,
                hidden_units=[hidden_units], output_activation=tf.nn.sigmoid)
        self.hidden_transform = MultiLayerPerceptron(
                in_dim=self.in_dim, out_dim=self.in_dim,
                hidden_units=[hidden_units])

        self.mean_transform = LinearTransform(
                in_dim=self.in_dim, out_dim=self.in_dim)
        self.std_transform = LinearTransform(
                in_dim=self.in_dim, out_dim=self.in_dim)

    def get_regularizer(self, scale=1.):
        """Regularizer for the weights of the multilayer perceptron."""
        sum_all = 0.
        sum_all += self.gate_transform.get_regularizer()
        sum_all += self.hidden_transform.get_regularizer()
        sum_all += self.mean_transform.get_regularizer()
        sum_all += self.std_transform.get_regularizer()

        return sum_all

    def get_transformation_parameters(self):
        """Returns the list of variables of the layers of the MLP.

        returns:
        --------
        list of tensorflow.Tensor
        """
        msg = "Not implemented for GatedTransition."
        raise NotImplemented(msg)

    def operator(self, x):
        gate = self.gate_transform.operator(x)
        hidden = self.hidden_transform.operator(x)
        mu = (1. - gate) * self.mean_transform.operator(x) + gate * hidden
        std = tf.nn.softplus(self.std_transform.operator(tf.nn.relu(hidden)))
        return mu, std
