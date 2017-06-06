import itertools as it
import numpy as np
import pytest
import variational_bayes as vb


@pytest.mark.parametrize('shape', [10, (3, 7)])
def test_softmax(shape):
    x = np.random.normal(0, 1, shape)
    proba = vb.softmax(x)
    np.testing.assert_array_less(0, proba)
    np.testing.assert_allclose(np.sum(proba, axis=-1), 1)


@pytest.mark.parametrize('num_blocks, block_size, offset',
                         it.product([1, 3, 7], [1, 5, 9], [0, 11]))
def test_pack_unpack_diag_roundtrip(num_blocks, block_size, offset):
    blocks = np.random.normal(0, 1, (num_blocks, block_size, block_size))
    packed = vb.pack_block_diag(blocks, offset)
    unpacked = vb.unpack_block_diag(packed, block_size, offset)
    np.testing.assert_allclose(blocks, unpacked)
