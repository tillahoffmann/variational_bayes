import numpy as np

from .distributions import Distribution, assert_constant, s, statistic
from .util import pack_block_diag, unpack_block_diag


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


def pack_coefficients(adjacency, bias):
    """
    Pack an adjacency tensor of shape (n, n, p) and a bias vector of shape (n,) into a condensed
    coefficient matrix of shape (n, 1 + n * p).
    """
    adjacency_num, n, p = np.shape(adjacency)
    bias_num = np.shape(bias)[0]
    assert adjacency_num == bias_num, "leading shape does not match"
    coefficients = np.reshape(adjacency, (adjacency_num, n * p))
    coefficients = np.concatenate([bias[..., None], coefficients], axis=1)
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


def pack_coefficient_var(adjacency_var, bias_var):
    """
    Pack an adjacency tensor variance of shape (n, n, p, p) and a bias vector variance of shape (n,)
    into a condensed coefficient matrix variance of shape (n, n * p + 1, n * p + 1) which is a
    sequence of n block diagonal matrices.
    """
    adjacency_num, _n, p, _p = np.shape(adjacency_var)
    assert p == _p, "adjacency_var must be a tensor of square matrices"
    bias_num = np.shape(bias_var)[0]
    assert adjacency_num == bias_num, "leading shape does not match"
    var = pack_block_diag(adjacency_var, 1)
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


class VectorAutoregressiveHelper:
    r"""
    Helper for vector autoregressive (VAR) models with Gaussian noise.

    Let `x` with shape `(t, n)` be a vector-value time-series with `n` observations for each of `t`
    time steps. The observation `x[i + 1]` at time step `x[i]` of an order-p VAR process is given by
    (in Einsteim summation convention notation)
    $$
    x_{ti} = \phi_i + x_{(t - q)j} \theta_{ijq} + \epsilon_{ti},
    $$
    where $\theta$ are the autoregressive coefficients, $\phi$ is a bias term, the implied sums are
    q=1...p, j=1...n, and $\epsilon_{ti}$ is independent noise for each node and time step with
    precision $\lambda_i$.

    It is convenient to collapse the tensor of autoregressive coefficients and the bias term into
    a single matrix $\xi$ such that
    $$
    \xi_{i1} = \phi_i,
    \xi_{i(1 + (j - 1) * p + q)} = \theta_{ijq},
    $$
    e.g. for an order-2 process $\xi_2 = (\phi_2, \theta_{211}, \theta{212}, \theta_{221}, ...)$.
    Similarly, we create a feature tensor `y` of shape `(t, n, 1 + n * p)` such that the regression
    problem can be expressed as
    $$
    x_{ti} = \xi_{ia} y_{tia} + \epsilon_{ti}.
    $$

    Then the likelihood becomes
    $$
    -\frac 12 \lambda_i (x_{ti} - \xi_{ia} y_{tia}) (x_{ti} - \xi_{ib} y_{tib}).
    $$
    We expand the terms and note that the coefficients of interest do not depend on the time index
    t such that we can precompute the sum
    $$
    \frac 12 \lambda_i (x_{ti} x_{ti} - 2 \xi{ia} y_{tia} x_{ti} + y_{tia}y_{tib} \xi_{ia}\xi{ib})
    $$
    """
    def __init__(self, x, coefficients):
        # Assert that we have constant observations
        assert_constant(x)
        self.x = s(x, 1)
        assert self.x.ndim == 2, "observations must be two-dimensional"
        self.num_steps, self.num_nodes = self.x.shape
        # Check the coefficients
        self.coefficients = coefficients
        num_nodes, a = s(coefficients, 1)
        assert num_nodes == self.num_nodes, "number of nodes does not match"
        self.order = (a - 1) // num_nodes
        # Evaluate the features, flatten along the last dimension to get shape (t, n * p)
        shape = (self.num_steps, self.num_nodes * self.order)
        self.features = evaluate_features(x, self.order).reshape(shape)
        # Prepend a bias feature
        self.features = np.hstack([np.ones((self.num_steps, 1)), self.features])
        # Precompute the square of the observations ...
        self.x2 = np.sum(np.square(x), axis=0)
        # ... the product of the observations and features ...
        self.xfeatures = np.einsum('ti,ta->ia', x, self.features)
        # ... and the square of the features
        self.features2 = np.einsum('ta,tb->ab', self.features, self.features)

        # Add proxy distributions
        self.bias = VectorAutoregressiveBiasDistribution(self.coefficients)
        self.adjacency = VectorAutoregressiveAdjacencyDistribution(self.coefficients)


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


class VectorAutoregressiveBiasDistribution(Distribution):
    def __init__(self, coefficients):
        super(VectorAutoregressiveBiasDistribution, self).__init__()
        self._coefficients = coefficients

    @statistic
    def mean(self):
        return unpack_bias(self._coefficients.mean)

    @statistic
    def var(self):
        return unpack_bias_var(self._coefficients.cov)


class VectorAutoregressiveAdjacencyDistribution(Distribution):
    def __init__(self, coefficients):
        super(VectorAutoregressiveAdjacencyDistribution, self).__init__()
        self._coefficients

    @statistic
    def mean(self):
        return unpack_adjacency(self._coefficients.mean)

    @statistic
    def cov(self):
        return unpack_adjacency_var(self._coefficients.cov)
