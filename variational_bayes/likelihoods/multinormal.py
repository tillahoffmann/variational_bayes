import numpy as np

from .likelihood import Likelihood
from ..distributions import s


class MultiNormalLikelihood(Likelihood):
    def __init__(self, x, mean, precision):
        super(MultiNormalLikelihood, self).__init__(x=x, mean=mean, precision=precision)

    def evaluate(self):
        _outer = s(self._mean, 1)[..., None, :] * s(self._x, 1)[..., :, None]
        chi2 = np.einsum('...ij,...ij', s(self._precision, 1), s(self._x, 'outer') +
                         s(self._mean, 'outer') - _outer - np.swapaxes(_outer, -1, -2))
        return 0.5 * (s(self._precision, 'logdet') - np.log(2 * np.pi) * _outer.shape[-1] - chi2)

    def natural_parameters(self, variable):
        # Get an object of ones with the correct broadcasted shape
        ones = np.ones(np.broadcast(s(self._x, 1), s(self._mean, 1)).shape)

        if variable == 'x':
            return {
                'mean': np.einsum('...ij,...i', s(self._precision, 1), s(self._mean, 1)) * ones,
                'outer': - 0.5 * s(self._precision, 1) * ones[..., None]
            }
        elif variable == 'mean':
            return {
                'mean': np.einsum('...ij,...i', s(self._precision, 1), s(self._x, 1)) * ones,
                'outer': - 0.5 * s(self._precision, 1) * ones[..., None]
            }
        elif variable == 'precision':
            _outer = s(self._x, 1)[..., None, :] * s(self._mean, 1)[..., :, None]
            return {
                'logdet': 0.5 * ones[..., 0],
                'mean': - 0.5 * (s(self._x, 'outer') + s(self._mean, 'outer') -
                                 _outer - np.swapaxes(_outer, -1, -2))
            }
        else:
            raise KeyError(variable)
