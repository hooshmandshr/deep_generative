"""Testing norm_flow.py classes."""


import numpy as np
import tensorflow as tf

from transform import LinearTransform, LSTMcell


def test_linear():
    """Tester for correctness of linear transformation."""

    in_dim = 2
    out_dim = 3

    input_ = np.random.rand(2, 100, 2)
    a_mat = np.random.rand(2, 3)
    bias = np.random.rand(1, 3)

    with tf.Graph().as_default():
        linear = LinearTransform(
                in_dim=in_dim, out_dim=out_dim,
                gov_param=tf.constant(np.append(a_mat, bias, axis=0)))
        in_tensor = tf.constant(input_)
        out_tensor = linear.operator(in_tensor)
        with tf.Session() as sess:
            tf_res = sess.run(out_tensor)

    expected = np.append(
            np.matmul(input_[0], a_mat)[None, :, :] + bias,
            np.matmul(input_[1], a_mat)[None, :, :] + bias,
            axis=0)
    print("Testing correct linear transform result")
    assert np.allclose(expected, tf_res), "Incorrect linear transform."


def test_lstm():
    """Tests lstm cell transformations."""
    n_example, n_time, n_dim = 3, 7, 10
    n_hidden_dim = 2

    with tf.Graph().as_default():
        # Input
        x = tf.constant(np.random.normal(0, 1, [n_time, n_example, n_dim]))
        # Input hidden state
        h = tf.constant(np.zeros([n_example, n_hidden_dim]))
        # Input error carousel state
        c = tf.constant(np.zeros([n_example, n_hidden_dim]))

        l = LSTMcell(in_dim=n_dim, out_dim=n_hidden_dim)

        output_h = [h]
        output_c = [c]
        for t in range(n_time):
            hidden_t, carousel_t = l.operator(
                    x=x[t], h=output_h[t], c=output_c[t])
            output_h.append(hidden_t)
            output_c.append(carousel_t)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(output_h)

    print("Testing LSTM output shape.")
    msg = "Shape of LSTM output not correct."
    assert np.array(output).shape == (n_time + 1, n_example, n_hidden_dim), msg

    with tf.Graph().as_default():
        # Input
        x = tf.constant(np.random.normal(0, 1, [n_time, n_example, n_dim]))
        # Input hidden state
        l = LSTMcell(in_dim=n_dim, out_dim=n_hidden_dim)
        out = l.operator(x=x)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(out)

    print("Testing LSTM output shape.")
    msg = "Shape of LSTM output not correct."
    assert np.array(output).shape == (n_time, n_example, n_hidden_dim), msg



if __name__ == "__main__":
    test_linear()
    test_lstm()
