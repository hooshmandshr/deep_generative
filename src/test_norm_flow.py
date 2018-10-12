"""Testing norm_flow.py classes."""


import numpy as np
import tensorflow as tf

from norm_flow import PlanarFlow, TimeAutoRegressivePlanarFlow

def planar_flow(params, x):
    """Apply multiple planar flow to input x.

    params:
    -------
    params: np.ndarray of shape (n_flow, 2 * n_dim + 1)
        Where n_flow is the number of multiple flow, n_ex is the number of
        examples and n_dim is the dimensionality of the space.
    x: np.ndarray of shape (n_flow, n_ex, n_dim)
        Where n_ex is the number of examples.

    returns:
    --------
    numpy.ndarray os shape (n_flow, n_ex, n_dim).
    """
    dim = x.shape[-1]
    w = params[:, :dim]
    u = params[:, dim:2*dim]
    b = params[:, 2*dim:2*dim+1]
    return x + u[:, None, :] * np.tanh(
            np.matmul(x, w[:, :, None]) + b[:, :, None])

def test_planar_flow():
    """Tester function for planar flow on tensor."""
    # Make sure the shapes are consistent.
    dim = 3
    n_example = 1000

    def run_tf_flow(input_val, dim, num=1, init_val=None, gov_param=None):
        with tf.Graph().as_default():
            g_param = None
            if gov_param is not None:
                g_param = tf.constant(gov_param)
            flow = PlanarFlow(
                    dim=dim, num=num, initial_value=init_val,
                    gov_param=g_param)
            x = tf.constant(input_val)
            y = flow.operator(x)
            z = flow.log_det_jacobian(x)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                output = sess.run(y)
                output_det_jacobian = sess.run(z)
        return output, output_det_jacobian

    input_ = np.random.rand(n_example, dim)

    output, output_det_jacobian = run_tf_flow(input_, dim=dim)

    print "Single flow transformation."
    assert output.shape == (n_example, dim), 'Testing output shape.'
    assert output_det_jacobian.shape == (n_example,), 'Testing log det jacobian.'

    # Make sure the shapes are consistent.
    num = 4

    input_ = np.random.rand(num, n_example, dim)

    output, output_det_jacobian = run_tf_flow(input_, dim=dim, num=num)

    print "Multiple flow transformation." 
    assert output.shape == (num, n_example, dim), 'Testing output shape.'
    assert output_det_jacobian.shape == (num, n_example), 'Testing log det jacobian.'

    # Making sure assigning governing parameters works.
    params = np.random.rand(num, 2 * dim + 1)

    output, output_det_jacobian = run_tf_flow(
            input_, dim=dim, num=num, gov_param=params)
    # Make sure the shapes are consistent.
    print "Multiple flow transformation correct computation, set parameters."
    assert np.allclose(output, planar_flow(params, input_))

    output, output_det_jacobian = run_tf_flow(
            input_, dim=dim, num=num, init_val=params)
    # Make sure the shapes are consistent.
    print "Multiple flow transformation result check, initial values."
    assert np.allclose(output, planar_flow(params, input_))


def test_autoregressive_flow():
    """Tester function for time auto-regressive planar flow on tensor."""
    # Make sure the shapes are consistent.
    dim = 3
    n_example = 1000
    time = 4

    n_sweep = 2
    n_layer = 3

    def run_tf_flow(input_val, dim, time, num=1, gov_param=None):
        with tf.Graph().as_default():
            g_param = None
            if gov_param is not None:
                g_param = tf.constant(gov_param)
            flow = TimeAutoRegressivePlanarFlow(dim=dim, time=time,
                    num=num, num_sweep=n_sweep, num_layer=n_layer,
                    gov_param=g_param)
            x = tf.constant(input_val)
            y = flow.operator(x)
            z = flow.log_det_jacobian(x)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                output = sess.run(y)
                output_det_jacobian = sess.run(z)
        return output, output_det_jacobian

    input_ = np.random.rand(n_example, time, dim)

    output, output_det_jacobian = run_tf_flow(input_, dim=dim, time=time)

    print "Single auto-regressive flow transformation."
    assert output.shape == (n_example, time, dim), 'Testing output shape.'
    assert output_det_jacobian.shape == (n_example,), 'Testing log det jacobian.'

    # Make sure the shapes are consistent.
    num = 4

    input_ = np.random.rand(num, n_example, time, dim)

    output, output_det_jacobian = run_tf_flow(
            input_, dim=dim, time=time, num=num)

    print "Multiple auto-regressive flow transformation with given parameters." 
    assert output.shape == (num, n_example, time, dim), 'Testing output shape.'
    assert output_det_jacobian.shape == (num, n_example), 'Testing log det jacobian.'

    param = np.random.rand(n_sweep, time - 1, n_layer, num, 2 * 2 * dim + 1)
    output, output_det_jacobian = run_tf_flow(
            input_, dim=dim, time=time, num=num, gov_param=param)

    print "Multiple auto-regressive flow transformation." 
    assert output.shape == (num, n_example, time, dim), 'Testing output shape.'
    assert output_det_jacobian.shape == (num, n_example), 'Testing log det jacobian.'


if __name__ == "__main__":
    test_planar_flow()
    test_autoregressive_flow()
