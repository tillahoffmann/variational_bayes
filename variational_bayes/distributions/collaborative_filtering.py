import numpy as np

from .distribution import Distribution, statistic, s, is_dependent


class CollaborativeFilteringDistribution(Distribution):
    """
    Distribution for collaborative filtering.

    Parameters
    ----------
    features :
        matrix of latent-space positions with shape `(n, p)`, where `n` is the number of points and
        `p` is the number of latent-space dimensions
    coefficients :
        matrix of latent-space positions with shape `(m, p)`, where `m` is the number of points and
        `p` is the number of latent-space dimensions
    noise :
        array of regression noise broadcastable with shape `(n, m)`
    """
    def __init__(self, features, coefficients, precision, strict=True):
        super(CollaborativeFilteringDistribution, self).__init__(
            features=features, coefficients=coefficients, precision=precision
        )
        # Disable the cache
        self._statistics = None
        self.strict = strict

    @statistic
    def mean(self):
        return np.dot(s(self._features, 1), s(self._coefficients, 1).T)

    def assert_valid_parameters(self):
        features = s(self._features, 1)
        precision = s(self._precision, 1)
        coefficients = s(self._coefficients, 1)
        assert np.ndim(features) == 2, "`features` must be two-dimensional"
        assert np.ndim(coefficients) == 2, "`coefficients` must be two-dimensional"
        assert features.shape[1] == coefficients.shape[1], "second dimension of `features` and " \
            "`coefficients` must match"
        np.testing.assert_array_less(0, precision, "`precision` must be positive")

    @staticmethod
    def canonical_parameters(natural_parameters):
        raise NotImplementedError

    def natural_parameters(self, x, variable):
        # Make sure we have the right data
        mask = ~np.isnan(s(x, 1))
        assert not self.strict or np.all(mask), "missing values"

        mean = np.dot(s(self._features, 1), s(self._coefficients, 1).T)
        ones = np.ones_like(mean)
        precision = s(self._precision, 1) * ones

        if is_dependent(x, variable):
            return {
                'mean': precision * mean,
                'square': -0.5 * precision
            }
        precision *= mask

        if is_dependent(self._features, variable):
            return {
                'mean': np.dot(precision * np.nan_to_num(s(x, 1)), s(self._coefficients, 1)),
                'outer': -0.5 * np.nansum(
                    precision[..., None, None] * s(self._coefficients, 'outer'), axis=1
                )
            }
        elif is_dependent(self._coefficients, variable):
            return {
                'mean': np.einsum('ij,ip->jp', precision * np.nan_to_num(s(x, 1)),
                                  s(self._features, 1)),
                'outer': -0.5 * np.sum(
                    precision[..., None, None] * s(self._features, 'outer')[:, None], axis=0
                )
            }
        elif is_dependent(self._precision, variable):
            _outer = np.sum(s(self._features, 'outer')[:, None] * s(self._coefficients, 'outer'),
                            axis=(2, 3))
            return {
                'log': 0.5 * mask,
                'mean': -0.5 * (np.nan_to_num(s(x, 'square')) -
                                2 * np.nan_to_num(s(x, 'mean')) * mean + _outer)
            }

    def log_proba(self, x):
        # Get the coefficients for the noise
        natural_parameters = self.natural_parameters(x, self._precision)
        return natural_parameters['log'] * s(self._precision, 'log') - 0.5 * np.log(2 * np.pi) + \
            s(self._precision, 'mean') * natural_parameters['mean']
