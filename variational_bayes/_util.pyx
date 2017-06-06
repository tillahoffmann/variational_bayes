# cython: boundscheck=False

import numpy as np


def pack_block_diag(blocks, int offset=0, output=None):
    """
    Pack blocks of shape `(..., n, k, k)` into a batch of block-diagonal matrices
    with shape `(..., n * k + offset, n * k + offset)`.
    """
    cdef:
        double[:, :, :, :] x
        double[:, :, :] y
        int block_size, num_blocks, block, batch, i, j, num_batches

    # Determine basic shapes
    shape = np.shape(blocks)
    block_size = shape[-1]
    assert block_size == shape[-2], "last dimensions must be square"
    num_blocks = shape[-3]

    # Reshape to (num_batch, num_blocks, block_size, block_size)
    x = np.reshape(blocks, (-1, num_blocks, block_size, block_size))
    num_batches = x.shape[0]

    # Allocate the output matrix
    output_shape = (num_batches, num_blocks * block_size + offset, num_blocks * block_size + offset)
    if output is None:
        y = np.zeros(output_shape)
    else:
        y = np.reshape(output, output_shape)

    # Fill the array
    for batch in range(num_batches):
        for block in range(num_blocks):
            for i in range(block_size):
                for j in range(block_size):
                    y[batch, block * block_size + i + offset, block * block_size + j + offset] = \
                        x[batch, block, i, j]

    return np.reshape(y, (*shape[:-3], num_blocks * block_size + offset, num_blocks * block_size + offset))


def unpack_block_diag(matrices, int block_size, int offset=0, output=None):
    """
    Unpack blocks of shape `(..., n, k, k)` into a batch of block-diagonal matrices
    with shape `(..., n * k + offset, n * k + offset)`.
    """
    cdef:
        double [:, :, :, :] x
        double [:, :, :] y
        int num_blocks, block, batch, i, j, num_batches, size

    # Determine basic shapes
    shape = np.shape(matrices)
    size = shape[-1]
    assert size == shape[-2], "last dimensions must be square"
    num_blocks = (size - offset) // block_size

    y = np.reshape(matrices, (-1, size, size))
    num_batches = y.shape[0]

    output_shape = (num_batches, num_blocks, block_size, block_size)
    if output is None:
        x = np.zeros(output_shape)
    else:
        x = np.reshape(output, output_shape)

    # Fill the blocks
    for batch in range(num_batches):
        for block in range(num_blocks):
            for i in range(block_size):
                for j in range(block_size):
                    x[batch, block, i, j] = \
                        y[batch, block * block_size + i + offset, block * block_size + j + offset]

    return np.reshape(x, (*shape[:-2], num_blocks, block_size, block_size))
