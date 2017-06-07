import operator
import numpy as np

from .distribution import Distribution, statistic, s
from ..util import assert_broadcastable


class NormalDistribution(Distribution):
    r"""
    Univariate normal distribution.

    The univariate normal distribution with mean $\mu$ and precision $\tau$ has log-pdf
    $$
    \frac{1}{2}\left(\log\frac{\tau}{2\pi} - \tau(x - 2 \mu x + \mu^2)\right).
    $$

    Parameters
    ----------
    mean : np.ndarray
        mean of the distribution
    precision : np.ndarray
        precision or inverse variance of the distribution
    """
    sample_ndim = 0

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

    @staticmethod
    def canonical_parameters(natural_parameters):
        precision = - 2 * natural_parameters['square']
        mean = natural_parameters['mean'] / precision
        return {
            'mean': mean,
            'precision': precision,
        }

    def assert_valid_parameters(self):
        # Check the expected values as a sanity check
        mean = s(self._mean, 1)
        precision = s(self._precision, 1)
        assert np.all(np.isfinite(mean)), "mean must be finite"
        np.testing.utils.assert_array_compare(operator.__le__, 0, precision,
                                              "precision must be non-negative")
        assert_broadcastable(mean, precision)

    def log_proba(self, x):
        return 0.5 * (s(self._precision, 'log') - np.log(2 * np.pi) - s(self._precision, 1) * (
            s(x, 2) - 2 * s(x, 1) * s(self._mean, 1) + s(self._mean, 2)
        ))

    def natural_parameters(self, x, variable):
        ones = np.ones(np.broadcast(s(x, 1), s(self._mean, 1)).shape)

        if variable == 'x':
            return {
                'mean': s(self._precision, 1) * s(self._mean, 1) * ones,
                'square': - 0.5 * s(self._precision, 1) * ones
            }
        elif variable == 'mean':
            return {
                'mean': s(self._precision, 1) * s(x, 1) * ones,
                'square': - 0.5 * s(self._precision, 1) * ones
            }
        elif variable == 'precision':
            return {
                'log': 0.5 * ones,
                'mean': - 0.5 * (
                    s(x, 2) - 2 * s(x, 1) * s(self._mean, 1) + s(self._mean, 2)
                )
            }
        else:
            raise KeyError(variable)
