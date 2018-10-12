"""Testing norm_flow.py classes."""


import numpy as np
import tensorflow as tf

from transform import LinearTransform


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
    print "Testing correct linear transform result"
    assert np.allclose(expected, tf_res), "Incorrect linear transform."

if __name__ == "__main__":
    test_linear()
