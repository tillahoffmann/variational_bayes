import numbers
import itertools as it
import numpy as np

from ..util import sum_leading_dims, array_repr


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
    VALIDATE_PARAMETERS = True

    def __init__(self, validate_parameters=None, **parameters):
        # Define a cache for statistics
        self._statistics = {}
        self._validate_parameters = validate_parameters
        self.parameters = {key: value if isinstance(value, Distribution) or value is None
                           else np.asarray(value) for key, value in parameters.items()}
        self.assert_valid_parameters()

    def update(self, canonical_parameters, **kwargs):
        """
        Update the distribution with the given canonical paramters.

        The statistics cache is automatically cleared.
        """
        # Clear the statistics cache
        self._statistics.clear()
        # Update statistics with keyword arguments
        duplicate_keys = set(canonical_parameters) & set(kwargs)
        assert not duplicate_keys, "duplicate keys: %s" % ", ".join(duplicate_keys)
        canonical_parameters.update(kwargs)
        # Update the canonical parameters
        for key, value in canonical_parameters.items():
            if key.startswith('_'):
                continue
            assert key in self.parameters, "parameter %s is not part of %s" % (key, self)
            actual = np.shape(value)
            desired = np.shape(self.parameters[key])
            assert actual == desired, "cannot update %s of %s: expected shape %s but got %s" % \
                (key, self, desired, actual)
            self.parameters[key] = np.asarray(value)

        validate_parameters = self.VALIDATE_PARAMETERS if self._validate_parameters is None else \
            self._validate_parameters

        if validate_parameters:
            self.assert_valid_parameters()

    def update_from_natural_parameters(self, natural_parameters):
        """
        Update the distribution from the given natural parameters.

        The statistics cache is automatically cleared.
        """
        if not isinstance(natural_parameters, dict):
            natural_parameters = self.aggregate_natural_parameters(natural_parameters)
        canonical_parameters = self.canonical_parameters(natural_parameters)
        assert not natural_parameters, "remaining statistics: %s" % \
            ", ".join(natural_parameters.keys())
        self.update(canonical_parameters)

    def __getattr__(self, name):
        if 'parameters' in self.__dict__ and name.strip('_') in self.parameters:
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
        """
        Get the number of dimensions of the given statistic.

        For example, the dimensionality of the mean is equal to the dimensionality of the sample,
        the dimensionality of the outer product is two (assuming that samples are vector-valued).
        """
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
        canonical_parameters = {
            key: value for key, value in cls.canonical_parameters(natural_parameters).items()
            if not key.startswith('_')
        }
        return cls(**canonical_parameters)

    def assert_valid_parameters(self):
        """
        Validate that the parameters of the distribution are valid.
        """
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
        """
        Obtain a likelihood for observation `x` under this distribution.
        """
        return Likelihood(self, x)

    @property
    def _repr_parameters(self):
        return ["%s=%s" % (key, array_repr(value) if isinstance(value, np.ndarray) else value)
                for key, value in self.parameters.items()]

    def __repr__(self):
        return "%s@0x%x(%s)" % (self.__class__.__name__, id(self), ", ".join(self._repr_parameters))


class DerivedDistribution(Distribution):
    """
    A derived distribution which is a deterministic function of one or more distributions.
    """
    def __init__(self, *parents, **parameters):
        self._parents = parents
        super(DerivedDistribution, self).__init__(**parameters)
        # Disable any caching
        self._statistics = None

    def assert_valid_parameters(self):
        pass

    @property
    def _repr_parameters(self):
        parameters = super(DerivedDistribution, self)._repr_parameters
        parameters.insert(0, "parents=%s" % [self._parents])
        return parameters

    def is_child(self, distribution):
        for parent in self._parents:
            if distribution is parent or (isinstance(parent, DerivedDistribution)
                                          and parent.is_child(distribution)):
                return True
        return False

    def transform_natural_parameters(self, distribution, natural_parameters):
        """
        Transform the natural parameters of the child distribution to match the natural parameters
        of the parent distribution.
        """
        raise NotImplementedError

    def _transformed_statistic(self, statistic):
        raise NotImplementedError

    @statistic
    def mean(self):
        return self._transformed_statistic('mean')

    @statistic
    def var(self):
        return self._transformed_statistic('var')

    @statistic
    def entropy(self):
        return self._transformed_statistic('entropy')

    @statistic
    def outer(self):
        return self._transformed_statistic('outer')

    def __getattr__(self, name):
        try:
            return super(DerivedDistribution, self).__getattr__(name)
        except AttributeError:
            if name.startswith('_') and not name.startswith('__'):
                return self._transformed_statistic(name)
            else:
                raise


class Likelihood:
    def __init__(self, distribution, x):
        self.distribution = distribution
        self.x = x

    def natural_parameters(self, variable):
        """
        Evaluate the natural parameters associated with `variable` given observation `x`.
        """
        # Get the original natural parameters
        natural_parameters = self.distribution.natural_parameters(self.x, variable)
        if natural_parameters is None:
            return None
        # Now, for each child distribution, we want to transform the natural parameters to match
        # the parent distribution
        for current in it.chain([self.x], self.distribution.parameters.values()):
            if is_dependent(current, variable):
                break

        while isinstance(current, DerivedDistribution):
            natural_parameters = current.transform_natural_parameters(variable, natural_parameters)
            for parent in current._parents:
                if is_dependent(parent, variable):
                    current = parent
        return natural_parameters

    def evaluate(self):
        """
        Evaluate the expected log-probability given observation `x`.
        """
        return self.distribution.log_proba(self.x)

    def __repr__(self):
        return "Likelihood@0x%x(distribution=%s, x=%s)" % \
            (id(self), self.distribution,
            array_repr(self.x) if isinstance(self.x, np.ndarray) else self.x)


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
    # Allow shorthands for statistics
    if statistic == 1:
        statistic = 'mean'
    elif statistic == 2:
        statistic = 'square'

    if isinstance(x, Distribution):
        return getattr(x, statistic)
    elif isinstance(statistic, numbers.Real):
        return x ** statistic
    elif statistic == 'mean':
        return x
    elif statistic == 'square':
        return x * x
    elif statistic == 'log':
        return np.log(x)
    # elif statistic == 'gammaln':
    #     return scipy.special.gammaln(x)
    elif statistic == 'outer':
        return x[..., :, None] * x[..., None, :]
    elif statistic == 'logdet':
        return np.linalg.slogdet(x)[1]
    elif statistic == 'log1m':
        return np.log1p(-x)
    elif statistic == 'cov':
        return np.zeros(x.shape + (x.shape[-1], ))
    elif statistic == 'interaction':
        assert x.ndim == 2, "interaction statistic is only defined for matrices"
        zz = np.einsum('ik,jl->ijkl', x, x)
        return zz
    else:
        raise KeyError(statistic)


s = evaluate_statistic


def is_constant(*args):
    """
    Determine whether the argument(s) are constant.
    """
    constant = [not isinstance(x, Distribution) for x in args]
    if len(constant) == 1:
        return constant[0]
    else:
        return constant


def assert_constant(*args):
    """
    Assert that the argument(s) are constant.
    """
    if len(args) == 1:
        assert is_constant(*args), "variable must be constant"
    else:
        assert all(is_constant(*args)), "variables must be constant"


def is_dependent(parent, child):
    """
    Check whether `child` depends on `parent` deterministically.
    """
    # The child and parent are identical
    if child is parent:
        return True
    # The parent is a derived distribution, so check the parents recursively
    if isinstance(parent, DerivedDistribution):
        for _parent in parent._parents:
            if is_dependent(_parent, child):
                return True
    return False
