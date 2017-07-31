import numpy as np

from .distribution import Distribution, statistic, s, is_dependent


class LinearRegressionDistribution(Distribution):
    """
    Distribution for linear regression.

    Parameters
    ----------
    features :
        matrix of features for the linear regression. The leading dimension corresponds to the batch
        dimension and the last dimension corresponds to the feature dimension.
    coefficients :
        vector of regression coefficients
    noise :
        array of regression noise. All dimensions correspond to the batch dimension.
    """
    sample_ndim = 1

    def __init__(self, features, coefficients, precision):
        super(LinearRegressionDistribution, self).__init__(features=features, coefficients=coefficients,
                                                           precision=precision)
        # Disable the cache
        self._statistics = None

    @statistic
    def mean(self):
        return np.dot(self._features, self._coefficients)

    @statistic
    def var(self):
        return np.ones_like(self.mean) * 1 / self._precision

    def assert_valid_parameters(self):
        features = s(self._features, 1)
        precision = s(self._precision, 1)
        coefficients = s(self._coefficients, 1)
        assert np.ndim(features) == 2, "`features` must be two-dimensional"
        assert np.ndim(coefficients) == 1, "`coefficients` must be one-dimensional"
        assert features.shape[1] == coefficients.shape[0], "second dimension of `features` and " \
            "first dimension of `coefficients` must match"
        np.testing.assert_array_less(0, precision, "`precision` must be positive")

    @staticmethod
    def canonical_parameters(natural_parameters):
        raise NotImplementedError

    def natural_parameters(self, x, variable):
        mean = np.dot(s(self._features, 1), s(self._coefficients, 1))
        ones = np.ones_like(mean)
        precision = s(self._precision, 1) * ones
        if is_dependent(x, variable):
            return {
                'mean': precision * mean,
                'square': -0.5 * precision
            }
        elif is_dependent(self._features, variable):
            return {
                'mean': (precision * s(x, 1))[:, None] * s(self._coefficients, 1),
                'outer': -0.5 * precision[:, None, None] * s(self._coefficients, 'outer')
            }
        elif is_dependent(self._coefficients, variable):
            return {
                'mean': (precision * s(x, 1))[:, None] * s(self._features, 1),
                'outer': -0.5 * precision[:, None, None] * s(self._features, 'outer')
            }
        elif is_dependent(self._precision, variable):
            _outer = np.sum(s(self._features, 'outer') * s(self._coefficients, 'outer'),
                            axis=(1, 2))
            return {
                'log': 0.5 * ones,
                'mean': -0.5 * (s(x, 'square') - 2 * s(x, 'mean') * mean + _outer)
            }

    def log_proba(self, x):
        # Get the coefficients for the noise
        natural_parameters = self.natural_parameters(x, self._precision)
        return natural_parameters['log'] * s(self._precision, 'log') - 0.5 * np.log(2 * np.pi) + \
            s(self._precision, 'mean') * natural_parameters['mean']
