import numpy as np

from .distribution import Distribution, statistic, s, is_dependent
from ..util import diag, is_positive_definite


class MultiNormalDistribution(Distribution):
    """
    Vector normal distribution.
    """
    sample_ndim = 1

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
        precision = - 2 * natural_parameters.pop('outer')
        cov = np.linalg.inv(precision)
        mean = np.einsum('...ij,...j', cov, natural_parameters.pop('mean'))
        return {
            'mean': mean,
            'precision': precision,
            '_cov': cov,  # Add the covariance matrix because we've already computed it
        }

    def update(self, canonical_parameters, **kwargs):
        super(MultiNormalDistribution, self).update(canonical_parameters, **kwargs)
        # See whether the covariance is precomputed
        cov = canonical_parameters.pop('_cov', None)
        if cov is not None:
            self._statistics['cov'] = cov

    def assert_valid_parameters(self):
        mean = s(self._mean, 1)
        precision = s(self._precision, 1)
        assert mean.ndim > 0, "the mean must be at least one-dimensional"
        assert precision.ndim == mean.ndim + 1, "dimensionality of the precision must be one " \
            "larger than the dimensionality of the mean"
        assert precision.shape[-1] == precision.shape[-2], "last two dimensions of the precision " \
            "must be equal"
        assert np.all(np.isfinite(mean)), "mean must be finite"
        assert is_positive_definite(precision), "precision must be positive definite"

    def log_proba(self, x):
        _outer = s(self._mean, 1)[..., None, :] * s(x, 1)[..., :, None]
        chi2 = np.einsum('...ij,...ij', s(self._precision, 1), s(x, 'outer') +
                         s(self._mean, 'outer') - _outer - np.swapaxes(_outer, -1, -2))
        return 0.5 * (s(self._precision, 'logdet') - np.log(2 * np.pi) * _outer.shape[-1] - chi2)

    def natural_parameters(self, x, variable):
        # Get an object of ones with the correct broadcasted shape
        ones = np.ones(np.broadcast(s(x, 1), s(self._mean, 1)).shape)

        if is_dependent(x, variable):
            return {
                'mean': np.einsum('...ij,...i', s(self._precision, 1), s(self._mean, 1)) * ones,
                'outer': - 0.5 * s(self._precision, 1) * ones[..., None]
            }
        elif is_dependent(self._mean, variable):
            return {
                'mean': np.einsum('...ij,...i', s(self._precision, 1), s(x, 1)) * ones,
                'outer': - 0.5 * s(self._precision, 1) * ones[..., None]
            }
        elif is_dependent(self._precision, variable):
            _outer = s(x, 1)[..., None, :] * s(self._mean, 1)[..., :, None]
            return {
                'logdet': 0.5 * ones[..., 0],
                'mean': - 0.5 * (s(x, 'outer') + s(self._mean, 'outer') -
                                 _outer - np.swapaxes(_outer, -1, -2))
            }

    def aggregate_natural_parameters(self, args):
        # Convert "square" factors to "outer" factors
        for natural_parameters in args:
            square = natural_parameters.pop('square', None)
            if square is not None:
                shape = square.shape + (square.shape[-1],)
                outer = np.zeros(shape)
                i = np.arange(square.shape[-1])
                outer[..., i, i] = square
                natural_parameters['outer'] = outer

        return super(MultiNormalDistribution, self).aggregate_natural_parameters(args)
