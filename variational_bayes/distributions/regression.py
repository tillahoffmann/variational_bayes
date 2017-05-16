import numpy as np
from scipy.special import expit

from .likelihood import Likelihood
from .distribution import s, assert_constant


def expitln(x):
    """
    Logarithm of the logistic sigmoid.
    """
    return - np.log1p(np.exp(-x))


class RegressionLikelihood(Likelihood):
    """
    Likelihood for generalised linear regression.

    Parameters
    ----------
    x : np.ndarray | Distribution
        observations
    features : np.ndarray | Distribution
        features used to predict the observations
    theta : np.ndarray | Distribution
        coefficients used to predict the observations
    **kwargs : dict[str, np.ndarray | Distribution]
        additional parameters
    """
    def __init__(self, x, features, theta, **kwargs):
        super(RegressionLikelihood, self).__init__(x=x, features=features, theta=theta, **kwargs)


class LogisticLikelihood(RegressionLikelihood):
    """
    Likelihood for binary logistic regression.

    Parameters
    ----------
    x : np.ndarray | Distribution
        observations
    features : np.ndarray | Distribution
        features used to predict the observations
    theta : np.ndarray | Distribution
        coefficients used to predict the observations
    xi : np.ndarray | Distribution

    References
    ----------
    https://arxiv.org/pdf/1310.5438.pdf
    """
    def __init__(self, x, features, theta, xi):
        super(LogisticLikelihood, self).__init__(x, features, theta, xi=xi)

    @staticmethod
    def _lambda(xi):
        return (expit(xi) - 0.5) / (2 * xi)

    @staticmethod
    def evaluate(x, features, theta, xi):  # pylint: disable=W0221
        # See equation 46 in https://arxiv.org/pdf/1310.5438.pdf
        assert_constant(xi)
        xi = s(xi, 1)
        # The paper used -1, +1 rather than 0, 1
        x_1 = 2 * s(x, 1) - 1

        _lambda = LogisticLikelihood._lambda(xi)
        return 0.5 * x_1 * np.einsum('...i,i', s(features, 1), s(theta, 1)) - \
            _lambda * np.einsum('...ij,...ij', s(features, 'outer'), s(theta, 'outer')) + \
            expitln(xi) - xi / 2 + _lambda * np.square(xi)

    @staticmethod
    def natural_parameters(variable, x, features, theta, xi):  # pylint: disable=W0221
        assert_constant(xi)
        xi = s(xi, 1)

        # The paper used -1, +1 rather than 0, 1
        x_1 = 2 * s(x, 1) - 1

        if variable == 'x':
            return {
                'mean': 0.5 * np.dot(s(features, 1), s(theta, 1))
            }
        elif variable == 'features':
            return {
                'mean': 0.5 * x_1 * s(theta, 1),
                'outer': - LogisticLikelihood._lambda(xi)[..., None, None] * s(theta, 'outer')
            }
        elif variable == 'theta':
            return {
                'mean': 0.5 * x_1[..., None] * s(features, 1),
                'outer': - LogisticLikelihood._lambda(xi)[..., None, None] * s(features, 'outer')
            }
        elif variable == 'xi':
            return {
                'value': np.sqrt(np.einsum('...ij,...ij', s(features, 'outer'), s(theta, 'outer')))
            }
        else:
            raise KeyError(variable)
