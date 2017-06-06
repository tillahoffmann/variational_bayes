import numbers
import numpy as np
import scipy.special

from ..util import sum_leading_dims
from .likelihood import Likelihood


class statistic:
    """
    Descriptor like `property` used to denote statistics of distributions.
    """
    def __init__(self, fget):
        self.fget = fget
        self.name = self.fget.__name__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # Get the statistic from the cache if enabled
        if obj._statistics is not None and self.name in obj._statistics:
            return obj._statistics[self.name]
        value = self.fget(obj)
        # Add the value to the cache
        if obj._statistics is not None:
            obj._statistics[self.name] = value
        return value


class Distribution:
    """
    Base class for distributions that act as factors in the approximate posterior.
    """
    sample_ndim = None

    def __init__(self, **parameters):
        # Define a cache for statistics
        self._statistics = {}
        self.parameters = {key: np.asarray(value) for key, value in parameters.items()}
        self.assert_valid_parameters()

    def update(self, canonical_parameters):
        # Clear the statistics cache
        self._statistics.clear()
        # Update the canonical parameters
        for key, value in canonical_parameters.items():
            assert key in self.parameters, "parameter %s is not part of %s" % (key, self)
            actual = np.shape(value)
            desired = np.shape(self.parameters[key])
            assert actual == desired, "cannot update %s of %s: expected shape %s but got %s" % \
                (key, self, desired, actual)
            self.parameters[key] = np.asarray(value)

        self.assert_valid_parameters()

    def update_from_natural_parameters(self, natural_parameters):
        if not isinstance(natural_parameters, dict):
            natural_parameters = self.aggregate_natural_parameters(natural_parameters)
        self.update(self.canonical_parameters(natural_parameters))

    def __getattr__(self, name):
        if name.strip('_') in self.parameters:
            return self.parameters[name.strip('_')]
        else:
            raise AttributeError(name)

    @property
    def statistics(self):
        """list[str] : names of supported statistics"""
        return [p for p in dir(self.__class__) if not p.startswith('_') and
                isinstance(getattr(self.__class__, p), statistic)]

    def aggregate_natural_parameters(self, args):
        """
        Aggregate coefficients of (sufficient) statistics.

        Parameters
        ----------
        args : list[dict]
            sequence of coefficients keyed by statistic name

        Returns
        -------
        aggregate : dict
            aggregated coefficients keyed by statistic name
        """
        aggregate = {}
        # Iterate over all natural parameters in the list
        for natural_parameters in args:
            # Iterate over all statistics
            for key, value in natural_parameters.items():
                if key not in aggregate:
                    aggregate[key] = 0
                # Aggregate coefficients over all leading dimensions except the batch dimension
                # of the distribution and the dimensionality of the statistic
                aggregate[key] += sum_leading_dims(
                    value, self.batch_ndim + self.statistic_ndim(key)
                )
        return aggregate

    @property
    def batch_ndim(self):
        """int : rank of the batch dimensions"""
        return np.ndim(self.mean) - self.sample_ndim

    def statistic_ndim(self, statistic):
        if statistic in ('mean', 'square', 'log', 'log1m'):
            return self.sample_ndim
        elif statistic in ('outer', ):
            assert self.sample_ndim == 1, "outer is only supported for vector distributions"
            return 2
        elif statistic in ('logdet', ):
            # -1 because the only dimensions that remain for 'logdet' are the batch dimensions
            return -1
        else:
            raise KeyError(statistic)

    @statistic
    def std(self):
        """np.ndarray : standard deviation"""
        return np.sqrt(self.var)

    @statistic
    def square(self):
        """np.ndarray : second moment"""
        return np.square(self.mean) + self.var

    @statistic
    def entropy(self):
        """np.ndarray : information entropy"""
        raise NotImplementedError

    @statistic
    def var(self):
        """np.ndarray : variance"""
        raise NotImplementedError

    @statistic
    def mean(self):
        """np.ndarray : first moment"""
        raise NotImplementedError

    @staticmethod
    def canonical_parameters(natural_parameters):
        """
        Obtain canonical parameters used to parametrize the distribution from natural parameters.
        """
        raise NotImplementedError

    @classmethod
    def from_natural_parameters(cls, natural_parameters):
        """
        Create a distribution from natural parameters.
        """
        return cls(**cls.canonical_parameters(natural_parameters))

    def assert_valid_parameters(self):
        raise NotImplementedError

    def natural_parameters(self, x, variable):
        """
        Evaluate the natural parameters associated with `variable` given observation `x`.
        """
        raise NotImplementedError

    def log_proba(self, x):
        """
        Evaluate the expected log-probability given observation `x`.
        """
        raise NotImplementedError

    def likelihood(self, x):
        return Likelihood(self, x)


class ReshapedDistribution(Distribution):
    def __init__(self, distribution, newshape):
        self._distribution = distribution
        self._newshape = newshape
        super(ReshapedDistribution, self).__init__()
        # Disable the cache
        self._statistics = None

    def _reshaped_statistic(self, statistic):
        # Get the statistic
        value = getattr(self._distribution, statistic)
        # Determine the new shape by popping leading dimensions until the size of popped dimensions
        # matches the desired size
        newsize = np.prod(self._newshape)
        shape = list(value.shape)
        size = 1
        while size < newsize:
            size *= shape.pop(0)

        # Sanity check
        assert size == newsize, "cannot reshape leading dimensions"
        newshape = self._newshape + tuple(shape)
        return np.reshape(value, newshape)

    @statistic
    def mean(self):
        return self._reshaped_statistic('mean')

    @statistic
    def var(self):
        return self._reshaped_statistic('var')

    @statistic
    def entropy(self):
        return self._reshaped_statistic('entropy')

    @statistic
    def outer(self):
        return self._reshaped_statistic('outer')

    def assert_valid_parameters(self):
        self._distribution.assert_valid_parameters()

    @staticmethod
    def canonical_parameters(natural_parameters):
        raise NotImplementedError


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
    if isinstance(x, Distribution):
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
    elif statistic == 'log1m':
        return np.log1p(-x)
    elif statistic == 'cov':
        return np.zeros(x.shape + (x.shape[-1], ))
    else:
        raise KeyError(statistic)


s = evaluate_statistic


def is_constant(*args):
    constant = [not isinstance(x, Distribution) for x in args]
    if len(constant) == 1:
        return constant[0]
    else:
        return constant


def assert_constant(*args):
    if len(args) == 1:
        assert is_constant(*args), "variable must be constant"
    else:
        assert all(is_constant(*args)), "variables must be constant"
