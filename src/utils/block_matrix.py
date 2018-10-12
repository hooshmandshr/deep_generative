"""Tools for sparse representation of block(bi-tri)diagonal matrices.

Proveds classes block diagonal, block-bi-diagonal, and block-tri-diagonal
marices and respectively their inverce, solving a linear system (Ax=B, in
other words x=A-1 B), and cholesky factor decomposition.

1) Inverse of a BlockDiagonalMatrix is another BlockDiagonalMatrix.
2) Solving Ax=B for x where A is a BlockBiDiagnolMatrix is a tensorflow.Tensor
of shape of resulting matrix (i.e. A-1B).
3) Cholesky factor of a BlockTriDiagonalMatrix is a BlockBiDiagonalMatrix.
"""

import numpy as np
import tensorflow as tf


class BlockDiagonalMatrix(object):

    def __init__(self, diag_block):
        """Set up the blocks of the matrix (lower-bi-diagonal) by default.

        params:
        -------
        diag_block: tensorflow.Tensor of shape (T, D, D) or (N, T, D, D)
            Respectively the T Digaonal blocks of the matrix (N matrices)
            where each have
            D x D dimensions.
        """
        shape = diag_block.shape
        if len(shape) < 3:
            raise ValueError(
                    "Shape of diag_block must be (..., T, D, D)")
        elif not shape[-1].value == shape[-2].value:
            raise ValueError(
                    "Shape of diag_block must be (..., T, D, D).")

        # Number of total blocks.
        self.outer_dim = shape[:-3]
        self.num_block = shape[-3].value
        # Dimension of each block.
        self.block_dim = shape[-1].value
        self.diag_block = diag_block
        # Necessary offsets for slicing the blocks of all matrices.
        self.matrix_offset_from = []
        self.matrix_offset_to = []
        for dim in diag_block.shape[:-3]:
            self.matrix_offset_from.append(0)
            self.matrix_offset_to.append(-1)
        self.matrix_offset_to += [1, -1, -1]
        self.diag_part = None

    def get_diag_part(self):
        """Returns the diagonal of all the blocks of matrices.

        returns:
        --------
        tf.Tensor of shape(..., T, D)
        """
        if self.diag_part is not None:
            return self.diag_part
        def get_diag_element(i):
            return tf.slice(self.diag_block,
                    self.matrix_offset_from + [0, i, i],
                    self.matrix_offset_from + [-1, 1, 1])
        self.diag_part = tf.squeeze(tf.concat(
                [get_diag_element(i) for i in range(self.block_dim)], axis=-1))
        return self.diag_part

    def get_block(self, block_num):
        """Gets the exact block_number of all matrices."""
        if block_num >= self.num_block:
            raise ValueError(
                    "block_num should be in range {}...{}.".format(
                        0, self.num_block - 1))
        return tf.slice(
                self.diag_block,
                self.matrix_offset_from + [block_num, 0, 0],
                self.matrix_offset_to)

    def matmul(self, b):
        """Matrix multiplication with another matrix."""
        pass

    def inverse(self):
        """Returns the inverse of the Block Diagonal Matrix."""
        # Diagonal blocks of the resulting matrix.
        return BlockDiagonalMatrix(tf.linalg.inv(self.diag_block))


class BlockBiDiagonalMatrix(BlockDiagonalMatrix):

    def __init__(self, diag_block, offdiag_block, lower=True):
        """Set up the blocks of the matrix (lower-bi-diagonal) by default.

        params:
        -------
        diag_block: tensorflow.Tensor of shape (T, D, D)
            Respectively the T Digaonal blocks of the matrix where each have
            D x D dimensions.
        offdiag_block: tensorflow.Tensor of shape (T-1, D, D)
            Respectively the T - 1 Digaonal blocks of the matrix where each
            have D x D dimensions.
        lower: bool
            True if the matrix is a lower-block-bi-diagonal matrix. Otherwise,
            upper-bi-diagonal-matrix.
        """
        super(BlockBiDiagonalMatrix, self).__init__(
            diag_block=diag_block)
        # Lower off diagonal blocks by default.
        err_msg = "Shape of diag_block and off_diag_block must be "
        err_msg += "(..., T, D, D) and (..., T - 1, D, D) respectively where "
        err_msg += "the first part of both should be the same."
        shape = offdiag_block.shape
        d_shape = self.diag_block.shape 
        if not len(shape) == len(d_shape):
            raise ValueError(err_msg)
        elif not shape[-3].value == d_shape[-3].value - 1:
            raise ValueError(err_msg)
        elif not(shape[-2:] == d_shape[-2:] and shape[:-3] == d_shape[:-3]):
            raise ValueError(err_msg)
        self.offdiag_block = offdiag_block
        self.lower = lower

    def get_block(self, block_num, diag=True):
        """Gets a particular block of all the diagonal/off-diagonal matrices."""
        if diag:
            return super(BlockBiDiagonalMatrix, self).get_block(block_num)
        elif block_num >= self.num_block - 1:
            raise ValueError(
                    "block_num should be in range {}...{}.".format(
                        0, self.num_block - 2))
        return tf.slice(
                self.offdiag_block,
                self.matrix_offset_from + [block_num, 0, 0],
                self.matrix_offset_to)

    def solve(self, b, transpose=False):
        """Returns x for which Ax=b where A is the matrix.

        params:
        -------
        b: tensorflow.Tensor of shape (..., ?, T, D) or (..., ?, TxD)
            The shape of the input tensor must be compatible with the matrix
            shape.
        transpose: bool
            If False, the reulst Ax=b is solved. If True (A^T)x=b is solved.
        Returns:
        --------
        tensorflow.Tensor of shape (..., ?, self.num_block * self.block_dim)
        which is the result of A^-1 * b.
        """
        # indicates whether the input is flattened in terms of time dimension
        # or T.
        flatten = False
        # Make sure the shape of input is compatible with the matrix(matrices).
        # Diagonal blocks of the resulting matrix.
        err_msg = "Shape of input b must be either {} or {}.".format(
                "(..., ?, T, D)", "(..., ?, T*D)")
        d_shape = self.diag_block.shape

        if len(b.shape) == len(d_shape):
            if not(b.shape[:-3] == d_shape[:-3] and b.shape[-2:] == d_shape[-3:-1]):
                raise ValueError(err_msg) 
        elif len(b.shape) == len(d_shape) - 1:
            # The input is time flattened. Unfold it.
            tot_dim = self.num_block * self.block_dim
            if not (b.shape[:-2] == d_shape[:-3] and b.shape[-1] == tot_dim):
                raise ValueError(err_msg)
            flatten = True
            b = tf.reshape(b,
                    b.shape.as_list()[:-1] + [self.num_block, self.block_dim])
        else:
            raise ValueError(err_msg)

        if transpose:
            self.transpose(in_place=True)

        def dot(M, x):
            return tf.expand_dims(tf.reduce_sum(M * x, axis=-1), axis=-2)

        def b_slice(time):
            return tf.slice(b,
                    self.matrix_offset_from + [0, time, 0],
                    self.matrix_offset_to[:-3] + [-1, 1, -1])

        # Get the inverse of all the diagonal blocks.
        diag_inv = super(BlockBiDiagonalMatrix, self).inverse()
        if self.lower: 
            x = []
            x.append(dot(diag_inv.get_block(0), b_slice(0)))
            for i in range(1, self.num_block):
                g = b_slice(i) - dot(self.get_block(i - 1, diag=False), x[-1])
                x.append(dot(diag_inv.get_block(i), g))
        else:
            x = []
            x.append(dot(
                diag_inv.get_block(self.num_block - 1),
                b_slice(self.num_block - 1)))
            for i in range(0, self.num_block - 1)[::-1]:
                g = b_slice(i) - dot(self.get_block(i, diag=False), x[-1])
                x.append(dot(diag_inv.get_block(i), g))
            x.reverse()

        if transpose:
            self.transpose(in_place=True)
        # Make sure the output has the same shape as the input.
        if flatten:
            return tf.concat([tf.squeeze(time_res) for time_res in x], axis=-1)
        return tf.concat(x, axis=-2)

    def transpose(self, in_place=False):
        """Transposes the matrix by transposing the blocks.

        params:
        -------
        in_place: bool
            If True, the transpose operation is done in place. Otherwise, a new
            BlockBiDiagonalMatrix is returned.

        returns:
        --------
        None or BlockBiDiagonalMatrix.
        """
        perm = [i for i in range(len(self.diag_block.shape))]
        # Exchange the last two dimensions of the block and off-block structure
        perm[-2:] = perm[-2:][::-1]

        if in_place:
            self.diag_block = tf.transpose(self.diag_block, perm=perm)
            self.offdiag_block = tf.transpose(
                    self.offdiag_block, perm=perm)
            self.lower = not self.lower
        else:
            return BlockBiDiagonalMatrix(
                    diag_block=tf.transpose(self.diag_block, perm=perm),
                    offdiag_block=tf.transpose(
                        self.offdiag_block, perm=perm),
                    lower=not(self.lower))

    def inverse(self):
        """Returns the inverse of the matrix."""
        #TODO: Implement if explicit inverse is required.
        # If result of multiplying the inverse by a matrix is wanted, however,
        # solve() method can be used.
        pass


class BlockTriDiagonalMatrix(BlockBiDiagonalMatrix):

    def __init__(self, diag_block, offdiag_block):
        """Set up the blocks of the matrix (lower-bi-diagonal) by default.

        params:
        -------
        diag_block: tensorflow.Tensor of shape (..., T, D, D)
            Respectively the T Digaonal blocks of the matrix where each have
            D x D dimensions.
        offdiag_block: tensorflow.Tensor of shape (..., T-1, D, D)
            Respectively the T - 1 Digaonal blocks of the matrix where each
            have D x D dimensions. 
        """

        super(BlockTriDiagonalMatrix, self).__init__(
            diag_block=diag_block, offdiag_block=offdiag_block)

    def cholesky(self):
        """Coputes the cholesky factor (lower triangular) of the matrix.

        returns:
        --------
        BlockBiDiagonalMatrix which represents the lower tirangular cholesky
        factor of block-tri-diagonal matrix that the objects represents.
        """
        result_diag = []
        result_offdiag = []
        result_diag.append(self.get_block(0))
        for i in range(self.num_block - 1):
            result_diag[i] = tf.cholesky(result_diag[i])
            result_offdiag.append(tf.matmul(
                self.get_block(i, diag=False),
                tf.linalg.inv(result_diag[i]), transpose_b=True))
            result_diag.append(self.get_block(i + 1) - tf.matmul(
                result_offdiag[i], result_offdiag[i], transpose_b=True))
        result_diag[-1] = tf.cholesky(result_diag[-1]) 

        # Concatenating the tensors into higher dimensional tensor.
        result_diag = tf.concat(result_diag, axis=-3)
        result_offdiag = tf.concat(result_offdiag, axis=-3)
        return BlockBiDiagonalMatrix(
            diag_block=result_diag, offdiag_block=result_offdiag)

    def inverse(self):
        """Returns the inverse matrix."""
        #TODO: Implement if required.
        pass

