import numpy as np

from .distribution import Distribution, Likelihood, assert_constant, s, DerivedDistribution, \
    statistic, is_dependent
from ..util import unpack_block_diag, pack_block_diag, diag, pad_dims, is_positive_definite


def shift(x, p):
    """
    Shift `x` by `p` steps and pad with zeros.

    Parameters
    ----------
    x : array_like
        matrix of `t` observations for `n` nodes with shape `(t, n)`
    p : int
        number of steps
    """
    # Apply the shift
    x = np.roll(x, p, axis=0)
    # Pad with zeros
    x[:p] = 0
    return x


def evaluate_features(y, p):
    """
    Construct a feature tensor with `p` back-shifted instances of `x`.

    Parameters
    ----------
    x : array_like
        matrix of `t` observations for `n` nodes with shape `(t, n)`
    p : int
        number of steps

    Returns
    -------
    features: np.ndarray
        tensor of features with shape `(t, n, p)`
    """
    # Apply the shift operator `p` times to get a shape (p, t, n)
    features = np.asarray([shift(y, q + 1) for q in range(p)])
    # Move the lag axis to the back to get shape (t, n, p)
    features = np.rollaxis(features, 0, 3)
    return features


def pack_coefficients(adjacency, bias, shape=None):
    """
    Pack an adjacency tensor of shape (n, n, p) and a bias vector of shape (n,) into a condensed
    coefficient matrix of shape (n, 1 + n * p).
    """
    if shape is None:
        _, n, p = np.shape(adjacency)
        shape = (n, n * p + 1)

    coefficients = np.zeros(shape)

    if bias is not None:
        bias_num, = np.shape(bias)
        assert shape[0] == bias_num
        coefficients[:, 0] = bias

    if adjacency is not None:
        adjacency_num, n, p = np.shape(adjacency)
        assert shape[0] == adjacency_num
        coefficients[:, 1:] = np.reshape(adjacency, (adjacency_num, n * p))
    return coefficients


def unpack_coefficients(coefficients):
    """
    Unpack a condensed coefficient matrix of shape (n, 1 + n * p) to an adjacency tensor of shape
    (n, n, p) and a bias vector of shape (n,).
    """
    return unpack_adjacency(coefficients), unpack_bias(coefficients)


def unpack_bias(coefficients):
    """
    Unpack a condensed coefficient matrix of shape (n, 1 + n * p) to a bias vector of shape (n,).
    """
    return coefficients[..., 0]


def unpack_adjacency(coefficients):
    """
    Unpack a condensed coefficient matrix of shape (n, 1 + n * p) to an adjacency tensor of shape
    (n, n, p).
    """
    adjacency = coefficients[..., 1:]
    order = adjacency.shape[1] // adjacency.shape[0]
    return np.reshape(adjacency, (adjacency.shape[0], adjacency.shape[1] // order, order))


def pack_coefficient_var(adjacency_var, bias_var, shape=None):
    """
    Pack an adjacency tensor variance of shape (n, n, p, p) and a bias vector variance of shape (n,)
    into a condensed coefficient matrix variance of shape (n, n * p + 1, n * p + 1) which is a
    sequence of n block diagonal matrices.
    """
    if shape is None:
        n, _n, p, _p = np.shape(adjacency_var)
        shape = (n, n * p + 1, n * p + 1)
    else:
        shape = shape + (shape[1], )

    var = np.zeros(shape)

    if adjacency_var is not None:
        n, _n, p, _p = np.shape(adjacency_var)
        assert n == _n, "adjacency_var must be square in the first two dimension"
        assert p == _p, "adjacency_var must be a tensor of square matrices"
        pack_block_diag(adjacency_var, 1, var)

    if bias_var is not None:
        bias_num, = np.shape(bias_var)
        assert shape[0] == bias_num, "leading shape does not match"
        var[..., 0, 0] = bias_var
    return var


def unpack_bias_var(var):
    """
    Unpack a condensed coefficient matrix variance of shape (n, n * p + 1, n * p + 1) to a bias
    vector variance of shape (n,).
    """
    return var[..., 0, 0]


def unpack_adjacency_var(var, order=None):
    """
    Unpack a condensed coefficient matrix variance of shape (n, n * p + 1, n * p + 1) to an
    adjacency tensor variance of shape (n, n, p, p).
    """
    n, a, _a = var.shape
    assert a == _a, "var must be a tensor of square matrices"
    order = (a - 1) // n
    return unpack_block_diag(var, order, 1)

def unpack_coefficient_var(var):
    return unpack_adjacency_var(var), unpack_bias_var(var)


class VARDistribution(Distribution):
    def __init__(self, coefficients, noise_precision, z=None):
        super(VARDistribution, self).__init__(z=z, coefficients=coefficients, noise_precision=noise_precision)

    def assert_valid_parameters(self):
        noise_precision = s(self._noise_precision, 1)
        assert np.ndim(noise_precision) == 1
        coefficients = s(self._coefficients, 1)
        assert np.ndim(coefficients) == 2
        n, a = coefficients.shape
        assert (a - 1) % n == 0
        if self._z is None:
            assert noise_precision.shape[0] == n
        else:
            z = s(self._z, 1)
            assert z.shape[0] == n

    @staticmethod
    def evaluate_features(x, order, filter_nans=False):
        # Assert that we have constant observations
        assert_constant(x)
        assert x.ndim == 2, "observations must be two-dimensional"
        num_steps, num_nodes = x.shape
        assert num_steps > num_nodes * order + 1, "number of steps must be at least as large as " \
            "the number of dimensions of the coefficient vector (num_nodes * order + 1)"
        # Evaluate the features, flatten along the last dimension to get shape (t, n * p)
        shape = (num_steps, num_nodes * order)
        features = evaluate_features(x, order).reshape(shape)
        # Prepend a bias feature
        features = np.hstack([np.ones((num_steps, 1)), features])
        # Drop the first `order` features and observations because the zero-padding can lead to oddities
        features = features[order:]
        x = x[order:]
        if np.any(np.isnan(x)) or np.any(np.isnan(features)):
            assert filter_nans, "found missing values in features or time series"
            missing = np.any(np.isnan(x), axis=1) | np.any(np.isnan(features), axis=1)
            x = x[~missing]
            features = features[~missing]
        return x, features

    @staticmethod
    def summary_statistics(x, order, filter_nans=False):
        x, features = VARDistribution.evaluate_features(x, order, filter_nans)
        # Precompute the square of the observations ...
        x2 = np.sum(np.square(x), axis=0)
        # ... the product of the observations and features ...
        xfeatures = np.einsum('ti,ta->ia', x, features)
        # ... and the square of the features
        features2 = np.einsum('ta,tb->ab', features, features)
        assert is_positive_definite(features2), "outer product of features is not positive " \
            "semi-definite"

        return x2, xfeatures, features2, x.shape[0]

    @staticmethod
    def coefficient_mle(x, order):
        _x2, xfeatures, features2, _num_steps = VARDistribution.summary_statistics(x, order)
        return np.einsum('ab,ia->ib', np.linalg.inv(features2), xfeatures)

    def natural_parameters(self, x, variable):
        # Extract the summary statistics
        x2, xfeatures, features2, num_steps = x

        if is_dependent(self._coefficients, variable):
            # Compute the noise-precision per node
            if self._z is None:
                noise_precision = s(self._noise_precision, 1)
            else:
                noise_precision = np.dot(s(self._z, 1), s(self._noise_precision, 1))
            return {
                # This has shape (n, a)
                'mean': xfeatures * noise_precision[:, None],
                # This has shape (n, a, a)
                'outer': - 0.5 * pad_dims(noise_precision, 3) * features2,
            }

        residuals2 = self._evaluate_residuals2(x)

        if self._z is not None and is_dependent(self._z, variable):
            return {
                'mean': 0.5 * (s(self._noise_precision, 'log') - np.log(2 * np.pi)) * num_steps - \
                    0.5 * residuals2[:, None] * s(self._noise_precision, 1)
            }

        elif is_dependent(self._noise_precision, variable):
            if self._z is None:
                return {
                    'log': 0.5 * np.ones(residuals2.shape[0]) * num_steps,
                    'mean': -0.5 * residuals2
                }
            else:
                return {
                    'log': 0.5 * np.sum(s(self._z, 1), axis=0) * num_steps,
                    'mean': -0.5 * np.sum(s(self._z, 1) * residuals2[:, None], axis=0)
                }

    def _evaluate_residuals2(self, x):
        x2, xfeatures, features2, num_steps = x
        return x2 - 2 * np.sum(xfeatures * s(self._coefficients, 1), axis=1) + \
            np.sum(features2 * s(self._coefficients, 'outer'), axis=(1, 2))

    def log_proba(self, x):
        if self._z is None:
            residuals2 = self._evaluate_residuals2(x)
            num_steps = x[3]
            return 0.5 * (s(self._noise_precision, 'log') - np.log(2 * np.pi)) * num_steps - \
                    0.5 * residuals2 * s(self._noise_precision, 1)
        else:
            return s(self._z, 1) * self.natural_parameters(x, self._z)['mean']


class VARBiasDistribution(DerivedDistribution):
    """
    Child distribution representing the shape `(n, )` bias of each series.
    """
    def __init__(self, coefficients):
        super(VARBiasDistribution, self).__init__(coefficients)
        self._parent = coefficients

    @statistic
    def mean(self):
        return unpack_bias(s(self._parent, 1))

    @statistic
    def var(self):
        return unpack_bias_var(s(self._parent, 'cov'))

    def transform_natural_parameters(self, distribution, natural_parameters):
        shape = s(self._parent, 1).shape
        return {
            'mean': pack_coefficients(None, natural_parameters['mean'], shape),
            'outer': pack_coefficient_var(None, natural_parameters['square'], shape)
        }


class VARAdjacencyDistribution(DerivedDistribution):
    """
    Child distribution representing the shape `(n, n, p)` adjacency.
    """
    def __init__(self, coefficients):
        super(VARAdjacencyDistribution, self).__init__(coefficients)
        self._parent = coefficients

    @statistic
    def mean(self):
        return unpack_adjacency(s(self._parent, 1))

    @statistic
    def cov(self):
        return unpack_adjacency_var(s(self._parent, 'cov'))

    @statistic
    def outer(self):
        return self.mean[..., None, :] * self.mean[..., :, None] + self.cov

    @statistic
    def var(self):
        return diag(self.cov)

    def transform_natural_parameters(self, distribution, natural_parameters):
        return {
            'mean': pack_coefficients(natural_parameters['mean'], None),
            'outer': pack_coefficient_var(natural_parameters['outer'], None)
        }


class VARDiagAdjacencyDistribution(DerivedDistribution):
    """
    Child distribution representing the shape `(n, p)` diagonal of the adjacency.
    """
    def __init__(self, adjacency):
        super(VARDiagAdjacencyDistribution, self).__init__(adjacency)
        self._parent = adjacency
        # Create an index
        self._diag_indices = np.diag_indices(s(self._parent, 1).shape[0])

    @statistic
    def mean(self):
        return s(self._parent, 1)[self._diag_indices]

    @statistic
    def cov(self):
        return s(self._parent, 'cov')[self._diag_indices]

    @statistic
    def outer(self):
        return self.mean[..., None, :] * self.mean[..., :, None] + self.cov

    @statistic
    def var(self):
        return diag(self.cov)

    def transform_natural_parameters(self, distribution, natural_parameters):
        """
        For the natural parameter associated with the mean, we have an `(n, p)` matrix which we want
        to transform to an `(n, n, p)` tensor for the full adjacency tensor.

        For the outer product, we have an `(n, p, p)` tensor which we want to transform to an
        `(n, n, p, p)` tensor for the full adjacency tensor.
        """
        n, p = natural_parameters['mean'].shape
        mean = np.zeros((n, n, p))
        mean[self._diag_indices] = natural_parameters['mean']

        outer = np.zeros((n, n, p, p))
        outer[self._diag_indices] = natural_parameters['outer']

        return {
            'mean': mean,
            'outer': outer,
        }



def simulate_series(bias, ar_coefficients, noise_precision, num_steps, x=None):
    # Get a list for the series observations
    if x is None:
        x = []
    else:
        x = list(x)

    num_nodes, _, order = np.shape(ar_coefficients)

    # Generate the actual series
    for _ in range(num_steps):
        predictor = np.copy(bias)
        for q in range(min(order, len(x))):
            predictor += np.dot(ar_coefficients[..., q], x[- (q + 1)])

        x.append(predictor + np.random.normal(0, 1, num_nodes) / np.sqrt(noise_precision))

    return np.asarray(x)
