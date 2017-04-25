import collections
import numbers
import numpy as np
import scipy.special


def softmax(x):
    x = np.exp(x - np.max(x, axis=-1)[..., None])
    return x / np.sum(x, axis=-1)[..., None]


def multidigamma(x, p):
    return np.sum(scipy.special.digamma(x[..., None] - 0.5 * np.arange(p)), axis=-1)


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


class BaseDistribution(Container):
    pass


def assert_constant(x):
    assert not isinstance(x, BaseDistribution), "variable must be a constant"


_statistic_names = {
    1: 'mean',
    2: 'square',
}


def evaluate_statistic(x, statistic):
    """
    Evaluate a statistic of a distribution or value.

    Parameters
    ----------
    x : np.ndarray or Distribution
        value or distribution to evaluate
    statistic : str
        statistic to evaluate

    Returns
    -------
    value : np.ndarray
        evaluated statistic
    """
    if isinstance(x, BaseDistribution):
        statistic = _statistic_names.get(statistic, statistic)
        return getattr(x, statistic)
    elif isinstance(statistic, numbers.Real):
        return x ** statistic
    elif statistic == 'mean':
        return x
    elif statistic == 'square':
        return x * x
    elif statistic == 'log':
        return np.log(x)
    elif statistic == 'gammaln':
        return scipy.special.gammaln(x)
    elif statistic == 'outer':
        return x[..., :, None] * x[..., None, :]
    elif statistic == 'logdet':
        return np.linalg.slogdet(x)[1]
    else:
        raise KeyError(statistic)


def aggregate_leading_dims(x, ndim):
    """
    Aggregate all but the `ndim` leading dimensions.
    """
    return np.sum(x, axis=tuple(range(np.ndim(x) - ndim)))


def aggregate_trailing_dims(x, ndim):
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
