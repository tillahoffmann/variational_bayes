import itertools as it
import numpy as np
import scipy.special
from ._util import pack_block_diag, unpack_block_diag


def softmax(x):
    x = np.exp(x - np.max(x, axis=-1)[..., None])
    return x / np.sum(x, axis=-1)[..., None]


def multidigamma(x, p):
    return np.sum(scipy.special.digamma(x[..., None] - 0.5 * np.arange(p)), axis=-1)


def diag(x):
    i = np.arange(x.shape[-1])
    return x[..., i, i]


class Container:
    """
    Container whose attributes can be accessed using indexing.
    """
    def __init__(self, **attributes):
        self._attributes = attributes

    def __getitem__(self, name):
        return self._attributes[name]

    def __setitem__(self, name, value):
        assert name in self._attributes, "attribute '%s' was not defined in the constructor" % name
        self._attributes[name] = value


def sum_leading_dims(x, ndim):
    """
    Aggregate all but the `ndim` leading dimensions.
    """
    return np.sum(x, axis=tuple(range(np.ndim(x) - ndim)))


def sum_trailing_dims(x, ndim):
    """
    Aggregate all but the `ndim` trailing dimensions.
    """
    return np.sum(x, axis=tuple(range(ndim, np.ndim(x))))


def pad_dims(x, ndim):
    """
    Pad `x` with dimensions of size one on the right to ensure it has the desired `ndim`.
    """
    shape = np.shape(x)
    if ndim <= len(shape):
        return x
    return np.reshape(x, shape + (1, ) * (ndim - len(shape)))


def safe_log(x):
    return np.log(np.where(x == 0, 1, x))


def is_positive_definite(x):
    """
    Check whether a matrix is positive definite.
    """
    try:
        np.linalg.cholesky(x)
        return True
    except np.linalg.LinAlgError:
        return False


def are_broadcastable(*x):
    """
    Check whether a sequence of arrays is broadcastable.
    """
    try:
        np.broadcast(*x)
        return True
    except ValueError:
        return False


def assert_broadcastable(*x):
    """
    Assert that a sequence of arrays is broadcastable.
    """
    assert are_broadcastable(*x), "arrays must be broadcastable"


def array_repr(a):
    a = np.asarray(a)
    return "array(%s, %s)" % (a.shape, a.dtype)


def onehot(z, minlength=None):
    """
    Encode indices as one-hot.
    """
    minlength = max(np.max(z) + 1, minlength or 0)
    onehot = np.zeros((len(z), minlength))
    onehot[np.arange(len(z)), z] = 1
    return onehot


def cluster_order(inferred, actual):
    """
    Determine the best permutation for the cluster ordering.

    Parameters
    ----------
    inferred : np.ndarray
        responsibility matrix
    actual : np.ndarray
        cluster labels or onehot encoding

    Returns
    -------
    ordering : np.ndarray
        best permutation of cluster labels
    """
    if np.ndim(actual) == 1:
        actual = onehot(actual, inferred.shape[1])

    best_ordering = None
    best_loss = np.inf

    for ordering in it.permutations(range(inferred.shape[1])):
        # Compute the log loss for misclassification
        loss = - np.sum(actual[:, ordering] * np.log(np.where(actual[:, ordering] == 0, 1, inferred)))
        if loss < best_loss:
            best_loss = loss
            best_ordering = ordering

    return list(best_ordering)
