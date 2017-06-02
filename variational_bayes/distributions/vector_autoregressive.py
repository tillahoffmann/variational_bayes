from .likelihood import *
from .distribution import *


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


class VectorAutoregressiveLikelihood(Likelihood):
    r"""
    Likelihood for vector autoregressive (VAR) models with Gaussian noise.

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
    def __init__(self, x, coefficients, precision, order):
        # Assert that we have constant observations
        assert_constant(x)
        x = s(x, 1)
        assert x.ndim == 2, "observations must be two-dimensional"
        num_steps, num_nodes = x.shape
        # Evaluate the features, flatten along the last dimension to get shape (t, n * p)
        y = evaluate_features(x, order).reshape((num_steps, num_nodes * order))
        y = np.hstack([np.ones((num_steps, 1)), y])
        # Flatten the last two dimensions
        xx = np.sum(np.square(x), axis=0)
        xy = np.einsum('ti,ta->ia', x, y)
        yy = np.einsum('ta,tb->ab', y, y)

        super(VectorAutoregressiveLikelihood, self).__init__(
            x=x, coefficients=coefficients, precision=precision, xy=xy, yy=yy, xx=xx, order=order
        )

    @staticmethod
    def evaluate(x, coefficients, precision, order, *, xx=None, xy=None, yy=None):  # pylint: disable=W0221
        assert xy is not None, "xy must be precomputed"
        assert yy is not None, "yy must be precomputed"

        x = s(x, 1)
        num_steps, num_nodes = x.shape

        chi2 = np.dot(s(precision, 1), xx - 2 * np.sum(xy * s(coefficients, 1), axis=1) +
                      np.einsum('ab,iab', yy, s(coefficients, 'outer')))
        return - (chi2 + num_steps * np.sum(s(precision, 'log') * np.ones(num_nodes))) / 2

    @staticmethod
    def natural_parameters(variable, x, coefficients, precision, order, *, xx=None, yy=None,  # pylint: disable=W0221
                           xy=None):
        assert xy is not None, "xy must be precomputed"
        assert yy is not None, "yy must be precomputed"
        assert_constant(x)
        x = s(x, 1)
        num_steps, num_nodes = x.shape

        if variable == 'x':
            raise NotImplementedError
        elif variable == 'precision':
            return {
                'mean': - (xx - 2 * np.einsum('ia,ia->i', xy, s(coefficients, 1)) +
                           np.einsum('ab,iab', yy, s(coefficients, 'outer'))) / 2,
                'log': num_steps * np.ones(num_nodes) / 2,
            }
        elif variable == 'coefficients':
            precision = s(precision, 1) * np.ones(num_nodes)
            return {
                'mean': precision[:, None] * xy,
                'outer': - precision[:, None, None] * yy / 2,
            }
        else:
            raise KeyError(variable)


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
