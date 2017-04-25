import collections
import numpy as np
import scipy.special

from .util import *
from .likelihoods import *


class statistic:
    """
    Descriptor like `property` used to denote statistics of distributions.
    """
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)


class Distribution(BaseDistribution):
    """
    Base class for distributions that act as factors in the approximate posterior.
    """
    likelihood_cls = None
    sample_ndim = None

    def __init__(self, **parameters):
        self.likelihood_cls.assert_valid_parameters(parameters)
        super(Distribution, self).__init__(**parameters)

    @property
    def statistics(self):
        """list[str] : names of supported statistics"""
        return [p for p in dir(self.__class__) if isinstance(getattr(self.__class__, p), statistic)]

    def aggregate_coefficients(self, args):
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
                    value, self.batch_ndim + self.statistic_ndim[key]
                )
        return aggregate

    def update(self, coefficients):
        """
        Update the parameters of the distribution given the coefficients of sufficient statistics.

        Parameters
        ----------
        coefficients : list[dict]
            coefficients of sufficient statistics keyed by name, e.g. 'mean', 'square', etc.
        """
        parameters = self.coefficients_to_parameters(coefficients)
        self.likelihood_cls.assert_valid_parameters(parameters)
        for key, value in parameters.items():
            assert np.shape(value) == np.shape(self[key]), "cannot update parameter '%s': expected " \
                "shape %s but got %s" % (key, np.shape(self[key]), np.shape(value))
            self._attributes[key] = value

    @property
    def batch_ndim(self):
        """int : rank of the batch dimensions"""
        return np.ndim(self.mean) - self.sample_ndim

    @property
    def statistic_ndim(self):
        return {
            'mean': self.sample_ndim,
            'square': self.sample_ndim,
            'log': self.sample_ndim,
        }

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

    def coefficients_to_parameters(self, coefficients):
        """
        Convert coefficients of sufficient statistics to parameters of the distribution.
        """
        raise NotImplementedError

    def log_proba(self, x):
        """
        Evaluate the logarithm of the probability of `x` under the distribution.
        """
        return self.likelihood_cls(x, **self._attributes).evaluate()


class NormalDistribution(Distribution):
    """
    Univariate normal distribution.

    Parameters
    ----------
    mean : np.ndarray
        mean of the distribution
    precision : np.ndarray
        precision or inverse variance of the distribution
    """
    sample_ndim = 0
    likelihood_cls = NormalLikelihood

    def __init__(self, mean, precision):
        super(NormalDistribution, self).__init__(mean=mean, precision=precision)

    @statistic
    def mean(self):
        return self['mean']

    @statistic
    def var(self):
        return 1.0 / self['precision']

    @statistic
    def entropy(self):
        return 0.5 * (np.log(2 * np.pi) + 1 - np.log(self['precision']))

    def coefficients_to_parameters(self, coefficients):
        precision = - 2 * coefficients['square']
        mean = coefficients['mean'] / precision
        return {
            'mean': mean,
            'precision': precision,
        }


class GammaDistribution(Distribution):
    """
    Univariate gamma distribution.

    Parameters
    ----------
    shape : np.ndarray
        shape parameter
    scale : np.ndarray
        scale parameter
    """
    likelihood_cls = GammaLikelihood
    sample_ndim = 0

    def __init__(self, shape, scale):
        super(GammaDistribution, self).__init__(shape=shape, scale=scale)

    @statistic
    def mean(self):
        return self['shape'] / self['scale']

    @statistic
    def var(self):
        return self['shape'] / np.square(self['scale'])

    @statistic
    def entropy(self):
        return self['shape'] - np.log(self['scale']) + scipy.special.gammaln(self['shape']) + \
                (1 - self['shape']) * scipy.special.digamma(self['shape'])

    @statistic
    def log(self):
        return scipy.special.digamma(self['shape']) - np.log(self['scale'])

    def coefficients_to_parameters(self, coefficients):
        return {
            'shape': coefficients['log'] + 1,
            'scale': -coefficients['mean'],
        }


class CategoricalDistribution(Distribution):
    """
    Vector categorical distribution.

    Parameters
    ----------
    proba : np.ndarray
        vector of probabilities
    """
    likelihood_cls = CategoricalLikelihood
    sample_ndim = 1

    def __init__(self, proba):
        super(CategoricalDistribution, self).__init__(proba=proba)

    @statistic
    def mean(self):
        return self['proba']

    @statistic
    def square(self):
        return self['proba']

    @statistic
    def var(self):
        return self['proba'] * (1 - self['proba'])

    @statistic
    def entropy(self):
        summands = np.log(np.where(self['proba'] > 0, self['proba'], 1.0))
        return - np.sum(self['proba'] * summands, axis=-1)

    def coefficients_to_parameters(self, coefficients):
        return {
            'proba': softmax(coefficients['mean'])
        }


class MultiNormalDistribution(Distribution):
    likelihood_cls = MultinormalLikelihood
    sample_ndim = 1
