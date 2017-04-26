import numpy as np

from .util import s, Likelihood
from ..util import pad_dims, sum_trailing_dims


class MixtureLikelihood(Likelihood):
    """
    Mixture likelihood.
    """
    @staticmethod
    def evaluate(z, likelihood, *args):   # pylint: disable=W0221
        likelihood = likelihood.evaluate(*args)
        z_1 = s(z, 1)
        assert z_1.ndim == 2, "indicator must be two-dimensional"
        assert z_1.shape == likelihood.shape[:2], "leading dimensions of the likelihood %s and " \
            "indicators %s must match" % (likelihood.shape, z_1.shape)
        # Aggregate with respect to the indicator dimension
        return np.sum(pad_dims(z_1, likelihood.ndim) * likelihood, axis=1)

    @staticmethod
    def natural_parameters(variable, z, likelihood, *args):  # pylint: disable=W0221
        if variable == 'z':
            # Just evaluate the responsibilities and aggregate any trailing dimensions
            likelihood = likelihood.evaluate(*args)
            return {
                'mean': sum_trailing_dims(likelihood, 2),
            }
        else:
            # Evaluate the statistics of the likelihood and sum up over the observations
            z_1 = s(z, 1)
            assert z_1.ndim == 2, "indicator must be two-dimensional"
            natural_parameters = {}
            for key, value in likelihood.natural_parameters(variable, *args).items():
                # Ensure the leading dimensions of the coefficients and the indicators match
                assert z_1.shape == value.shape[:2], "leading dimensions of the coefficients " \
                    "'%s' %s and the indicators %s must match" % (key, value.shape, z_1.shape)
                natural_parameters[key] = np.sum(pad_dims(z_1, value.ndim) * value, axis=0)
            return natural_parameters
