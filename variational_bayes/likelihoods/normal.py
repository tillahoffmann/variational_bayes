import numpy as np

from .likelihood import Likelihood
from ..distributions import s


class NormalLikelihood(Likelihood):
    r"""
    Univariate normal distribution.

    The univariate normal distribution with mean $\mu$ and precision $\tau$ has log-pdf
    $$
    \frac{1}{2}\left(\log\frac{\tau}{2\pi} - \tau(x - 2 \mu x + \mu^2)\right).
    $$
    """
    def __init__(self, x, mean, precision):
        super(NormalLikelihood, self).__init__(x=x, mean=mean, precision=precision)

    def evaluate(self):  # pylint: disable=W0221
        return 0.5 * (s(self._precision, 'log') - np.log(2 * np.pi) - s(self._precision, 1) * (
            s(self._x, 2) - 2 * s(self._x, 1) * s(self._mean, 1) + s(self._mean, 2)
        ))

    def natural_parameters(self, variable):  # pylint: disable=W0221
        # Get an object of ones with the correct broadcasted shape
        ones = np.ones(np.broadcast(s(self._x, 1), s(self._mean, 1)).shape)

        if variable == 'x':
            return {
                'mean': s(self._precision, 1) * s(self._mean, 1) * ones,
                'square': - 0.5 * s(self._precision, 1) * ones
            }
        elif variable == 'mean':
            return {
                'mean': s(self._precision, 1) * s(self._x, 1) * ones,
                'square': - 0.5 * s(self._precision, 1) * ones
            }
        elif variable == 'precision':
            return {
                'log': 0.5 * ones,
                'mean': - 0.5 * (
                    s(self._x, 2) - 2 * s(self._x, 1) * s(self._mean, 1) + s(self._mean, 2)
                )
            }
        else:
            raise KeyError(variable)
