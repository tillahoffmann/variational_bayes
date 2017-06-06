import numpy as np

from .distribution import Distribution, s, statistic
from .likelihood import Likelihood
from ..util import diag, is_positive_definite


class MultiNormalLikelihood(Likelihood):
    def __init__(self, x, mean, precision):
        super(MultiNormalLikelihood, self).__init__(x=x, mean=mean, precision=precision)

    @staticmethod
    def evaluate(x, mean, precision):  # pylint: disable=W0221
        _cross = s(mean, 1)[..., None, :] * s(x, 1)[..., :, None]
        chi2 = np.einsum('...ij,...ij', s(precision, 1), s(x, 'outer') + s(mean, 'outer') -
                         _cross - np.swapaxes(_cross, -1, -2))
        return 0.5 * (s(precision, 'logdet') - np.log(2 * np.pi) * _cross.shape[-1] - chi2)

    @staticmethod
    def natural_parameters(variable, x, mean, precision):  # pylint: disable=W0221
        # Get an object of ones with the correct broadcasted shape
        ones = np.ones(np.broadcast(s(x, 1), s(mean, 1)).shape)

        if variable == 'x':
            return {
                'mean': np.einsum('...ij,...i', s(precision, 1), s(mean, 1)) * ones,
                'outer': - 0.5 * s(precision, 1) * ones[..., None]
            }
        elif variable == 'mean':
            return {
                'mean': np.einsum('...ij,...i', s(precision, 1), s(x, 1)) * ones,
                'outer': - 0.5 * s(precision, 1) * ones[..., None]
            }
        elif variable == 'precision':
            _cross = s(x, 1)[..., None, :] * s(mean, 1)[..., :, None]
            return {
                'logdet': 0.5 * ones[..., 0],
                'mean': - 0.5 * (s(x, 'outer') + s(mean, 'outer') -
                                 _cross - np.swapaxes(_cross, -1, -2))
            }
        else:
            raise KeyError(variable)


class MultiNormalDistribution(Distribution):
    """
    Vector normal distribution.
    """
    sample_ndim = 1
    likelihood = MultiNormalLikelihood

    def __init__(self, mean, precision):
        super(MultiNormalDistribution, self).__init__(mean=mean, precision=precision)

    @statistic
    def mean(self):
        return self._mean

    @statistic
    def cov(self):
        return np.linalg.inv(self._precision)

    @statistic
    def var(self):
        return diag(self.cov)

    @statistic
    def entropy(self):
        p = np.shape(self.mean)[-1]
        return 0.5 * p * (np.log(2 * np.pi) + 1) - 0.5 * np.linalg.slogdet(self._precision)[1]

    @statistic
    def outer(self):
        return self.mean[..., None] * self.mean[..., None, :] + self.cov

    @staticmethod
    def canonical_parameters(natural_parameters):
        precision = - 2 * natural_parameters['outer']
        cov = np.linalg.inv(precision)
        mean = np.einsum('...ij,...j', cov, natural_parameters['mean'])
        return {
            'mean': mean,
            'precision': precision,
        }

    def assert_valid_parameters(self):
        assert self._mean.ndim > 0, "the mean must be at least one-dimensional"
        assert self._precision.ndim == self._mean.ndim + 1, "dimensionality of the precision must " \
            "be one larger than the dimensionality of the mean"
        assert self._precision.shape[-1] == self._precision.shape[-2], "last two dimensions of the " \
            "precision must be equal"
        assert np.all(np.isfinite(self._mean)), "mean must be finite"
        assert is_positive_definite(self._precision), "precision must be positive definite"
