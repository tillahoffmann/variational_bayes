import numpy as np
import scipy.special


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
    Pad `x` with dimensions of size one to ensure it has the desired `ndim`.
    """
    shape = np.shape(x)
    return np.reshape(x, shape + (1, ) * (ndim - len(shape)))

def safe_log(x):
    return np.log(np.where(x == 0, 1, x))
