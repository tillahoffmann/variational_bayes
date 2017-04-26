import numbers
import collections
import numpy as np
import scipy.special


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
        # Add the statistic to the cache if not already present
        if not self.name in obj._statistics:
            obj._statistics[self.name] = self.fget(obj)
        return obj._statistics[self.name]


class Distribution:
    """
    Base class for distributions that act as factors in the approximate posterior.
    """
    sample_ndim = None
    likelihood = None

    def __init__(self, **parameters):
        self._statistics = {}
        self.parameters = {key: np.asarray(value) for key, value in parameters.items()}
        self.assert_valid_parameters()

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
        aggregate = collections.defaultdict(lambda: 0)
        # Iterate over all coefficients in the list
        for coefficients in args:
            # Iterate over all statistics
            for key, value in coefficients.items():
                # Aggregate coefficients over all leading dimensions except the batch dimension
                # of the distribution and the dimensionality of the statistic
                aggregate[key] += aggregate_leading_dims(
                    value, self.batch_ndim + self.statistic_ndim(key)
                )
        return aggregate

    @property
    def batch_ndim(self):
        """int : rank of the batch dimensions"""
        return np.ndim(self.mean) - self.sample_ndim

    def statistic_ndim(self, statistic):
        pass

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

    @classmethod
    def from_natural_parameters(cls, natural_parameters):
        """
        Create a distribution from natural parameters.
        """
        raise NotImplementedError

    def assert_valid_parameters(self):
        raise NotImplementedError

    def log_proba(self, x):
        assert self.likelihood is not None, "likelihood is not defined for %s" % self
        return self.likelihood.evaluate(x, **self.parameters)


class Likelihood:
    @staticmethod
    def evaluate(x, *parameters):
        raise NotImplementedError

    @staticmethod
    def natural_parameters(variable, x, *parameters):
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
    else:
        raise KeyError(statistic)


s = evaluate_statistic


def assert_constant(*args):
    for x in args:
        assert not isinstance(x, Distribution), "variable must be a constant"
