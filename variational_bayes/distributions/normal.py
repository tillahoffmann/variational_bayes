import operator
import numpy as np

from .util import Distribution, statistic, s, Likelihood


class NormalLikelihood(Likelihood):
    @staticmethod
    def evaluate(x, mean, precision):  # pylint: disable=W0221
        return 0.5 * (s(precision, 'log') - np.log(2 * np.pi) - s(precision, 1) * (
            s(x, 2) - 2 * s(x, 1) * s(mean, 1) + s(mean, 2)
        ))

    @staticmethod
    def natural_parameters(variable, x, mean, precision):  # pylint: disable=W0221
        # Get an object of ones with the correct broadcasted shape
        ones = np.ones(np.broadcast(s(x, 1), s(mean, 1)).shape)

        if variable == 'x':
            return {
                'mean': s(precision, 1) * s(mean, 1) * ones,
                'square': - 0.5 * s(precision, 1) * ones
            }
        elif variable == 'mean':
            return {
                'mean': s(precision, 1) * s(x, 1) * ones,
                'square': - 0.5 * s(precision, 1) * ones
            }
        elif variable == 'precision':
            return {
                'log': 0.5 * ones,
                'mean': - 0.5 * (s(x, 2) - 2 * s(x, 1) * s(mean, 1) + s(mean, 2))
            }
        else:
            raise KeyError(variable)


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
    _likelihood = NormalLikelihood

    def __init__(self, mean, precision):
        super(NormalDistribution, self).__init__(mean=mean, precision=precision)

    @statistic
    def mean(self):
        return self._mean

    @statistic
    def var(self):
        return 1.0 / self._precision

    @statistic
    def entropy(self):
        return 0.5 * (np.log(2 * np.pi) + 1 - np.log(self._precision))

    @classmethod
    def from_natural_parameters(cls, natural_parameters):
        precision = - 2 * natural_parameters['square']
        mean = natural_parameters['mean'] / precision
        return cls(mean, precision)

    def assert_valid_parameters(self):
        assert np.all(np.isfinite(self._mean)), "mean must be finite"
        np.testing.utils.assert_array_compare(operator.__le__, 0, self._precision,
                                              "precision must be non-negative")
        assert np.shape(self._mean) == np.shape(self._precision), "shape of mean and precision " \
            "must match"
