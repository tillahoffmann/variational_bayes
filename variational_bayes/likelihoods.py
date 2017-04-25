import operator
import numpy as np
from .util import *


class Likelihood(Container):
    def evaluate(self):
        """
        Evaluate the likelihood.
        """
        raise NotImplementedError

    def natural_parameters(self, variable):
        """
        Evaluate the natural parameters, i.e. coefficients of the sufficient statistics, for the
        given variable.

        Parameters
        ----------
        variable : str or Distribution
            name of a variable or a Distribution instance

        Returns
        -------
        natural_parameters : dict
            natural parameters, i.e. coefficients of the sufficient statistics, keyed by statistic
            name, e.g. 'mean', 'square', etc.
        """
        if isinstance(variable, BaseDistribution):
            # Look up the name of the variable
            for key, value in self._attributes.items():
                if variable is value:
                    return self._natural_parameters(key)
            # The variable is probably not associated with this likelihood
            return {}
        else:
            assert variable in self._attributes, "`variable` must be one of %s" % \
                self._attributes.keys()
            return self._natural_parameters(variable)

    def _es(self, name, statistic):
        """Short hand for evaluating statistics of attributes of the likelihood."""
        return evaluate_statistic(self._attributes[name], statistic)

    def _natural_parameters(self, variable):
        raise NotImplementedError

    @staticmethod
    def assert_valid_parameters(parameters):
        raise NotImplementedError


class NormalLikelihood(Likelihood):
    """
    Univariate normal likelihood.

    Parameters
    ----------
    x : np.ndarray or Distribution
        observations
    mean : np.ndarray or Distribution
        mean of the normal distribution
    precision : np.ndarray or Distribution
        precision or inverse variance of the normal distribution
    """
    def __init__(self, x, mean, precision):
        super(NormalLikelihood, self).__init__(x=x, mean=mean, precision=precision)

    def evaluate(self):
        return 0.5 * (self._es('precision', 'log') - np.log(2 * np.pi)) - \
            0.5 * self._es('precision', 1) * (self._es('x', 2) - 2 * self._es('x', 1) *
                                              self._es('mean', 1) + self._es('mean', 2))

    def _natural_parameters(self, variable):
        x_1 = self._es('x', 1)
        ones = np.ones_like(x_1 + self._es('mean', 1))
        if variable == 'precision':
            return {
                'log': 0.5 * ones,
                'mean': -0.5 * (self._es('x', 2) - 2 * x_1 * self._es('mean', 1) +
                                self._es('mean', 2))
            }
        elif variable == 'x':
            return {
                'square': -0.5 * self._es('precision', 1) * ones,
                'mean': self._es('precision', 1) * self._es('mean', 1)
            }
        elif variable == 'mean':
            return {
                'square': -0.5 * self._es('precision', 1) * ones,
                'mean': self._es('precision', 1) * self._es('x', 1)
            }
        else:
            raise KeyError(variable)

    @staticmethod
    def assert_valid_parameters(parameters):
        np.testing.assert_array_less(0, parameters['precision'], "precision must be positive")
        assert np.all(np.isfinite(parameters['mean'])), "mean must be finite"


class MixtureLikelihood(Likelihood):
    """
    Mixture likelihood.
    """
    def __init__(self, z, likelihood):
        # Add the attributes and the attributes of the other likelihood (for easier lookup in
        # natural_parameters).
        super(MixtureLikelihood, self).__init__(z=z, likelihood=likelihood,
                                                **likelihood._attributes)

    def evaluate(self):
        likelihood = self['likelihood'].evaluate()
        z_1 = self._es('z', 1)
        assert z_1.ndim == 2, "indicator must be two-dimensional"
        assert z_1.shape == likelihood.shape[:2], "leading dimensions of the likelihood %s and " \
            "indicators %s must match" % (likelihood.shape, z_1.shape)
        return np.sum(pad_dims(z_1, likelihood.ndim) * likelihood, axis=1)

    def _natural_parameters(self, variable):
        if variable == 'z':
            # Just evaluate the responsibilities and aggregate any trailing dimensions
            return {'mean': aggregate_trailing_dims(self['likelihood'].evaluate(), 2)}
        else:
            # Evaluate the statistics of the likelihood and sum up over the observations
            z_1 = self._es('z', 1)
            assert z_1.ndim == 2, "indicator must be two-dimensional"
            _coefficients = self['likelihood'].natural_parameters(variable)
            coefficients = {}
            for key, value in _coefficients.items():
                # Ensure the leading dimensions of the coefficients and the indicators match
                assert z_1.shape == value.shape[:2], "leading dimensions of the coefficients " \
                    "'%s' %s and the indicators %s must match" % (key, value.shape, z_1.shape)
                coefficients[key] = np.sum(pad_dims(z_1, value.ndim) * value, axis=0)
            return coefficients

    @staticmethod
    def assert_valid_parameters(parameters):
        z_1 = evaluate_statistic(parameters['z'], 1)
        np.testing.utils.assert_array_compare(operator.__le__, 0, z_1, "z must be non-negative")
        np.testing.assert_allclose(np.sum(z_1, axis=-1), 1, err_msg="entries of z must sum to one")


class GammaLikelihood(Likelihood):
    def __init__(self, x, shape, scale):
        super(GammaLikelihood, self).__init__(x=x, shape=shape, scale=scale)

    def evaluate(self):
        shape_1 = self._es('shape', 1)
        return shape_1 * self._es('scale', 'log') + (shape_1 - 1.0) * self._es('x', 'log') - \
            self._es('scale', 1) * self._es('x', 1) - self._es('shape', 'gammaln')

    def _natural_parameters(self, variable):
        if variable == 'x':
            return {
                'log': self._es('shape', 1) - 1,
                'mean': - self._es('scale', 1),
            }
        else:
            raise KeyError(variable)

    @staticmethod
    def assert_valid_parameters(parameters):
        np.testing.assert_array_less(0, parameters['shape'], "shape must be positive")
        np.testing.assert_array_less(0, parameters['scale'], "scale must be positive")


class CategoricalLikelihood(Likelihood):
    def __init__(self, x, proba):
        super(CategoricalLikelihood, self).__init__(x=x, proba=proba)

    def evaluate(self):
        return np.sum(self._es('x', 1) * self._es('proba', 'log'), axis=-1)

    def _natural_parameters(self, variable):
        if variable == 'x':
            return {
                'mean': self._es('proba', 'log')
            }
        else:
            raise KeyError(variable)

    @staticmethod
    def assert_valid_parameters(parameters):
        np.testing.utils.assert_array_compare(operator.__le__, 0, parameters['proba'],
                                              "proba must be non-negative")
        np.testing.assert_allclose(np.sum(parameters['proba'], axis=-1), 1,
                                   err_msg="proba must sum to one")


class MultiNormalLikelihood(Likelihood):
    def __init__(self, x, mean, precision):
        super(MultiNormalLikelihood, self).__init__(x=x, mean=mean, precision=precision)

    def evaluate(self):
        x_outer = self._es('x', 'outer')
        x_1 = self._es('x', 1)
        mean_1 = self._es('mean', 1)
        mean_outer = self._es('mean', 'outer')
        precision_1 = self._es('precision', 1)
        precision_logdet = self._es('precision', 'logdet')

        chi2 = np.einsum('...ij,...ij', mean_outer, precision_1) - \
            2 * np.einsum('...ij,...i,...j', precision_1, mean_1, x_1) + \
            np.einsum('...ij,...ij', precision_1, x_outer)

        return -0.5 * chi2 + 0.5 * (precision_logdet - x_1.shape[-1] * np.log(2 * np.pi))

    def _natural_parameters(self, variable):
        ones = np.ones_like(self._es('x', 1)[..., None] + self._es('mean', 1)[..., None, :])
        if variable == 'x':
            return {
                'outer': -0.5 * self._es('precision', 1) * ones,
                'mean': np.einsum('...ij,...j', self._es('precision', 1), self._es('mean', 1)),
            }
        elif variable == 'mean':
            return {
                'outer': -0.5 * self._es('precision', 1) * ones,
                'mean': np.einsum('...ij,...j', self._es('precision', 1), self._es('x', 1)),
            }
        elif variable == 'precision':
            mean_1 = self._es('mean', 1)
            x_1 = self._es('x', 1)
            return {
                'mean': -0.5 * (self._es('mean', 'outer') + self._es('x', 'outer') -
                                mean_1[..., :, None] * x_1[..., None, :] -
                                mean_1[..., None, :] * x_1[..., :, None]),
                'logdet': 0.5 * ones[..., 0, 0]
            }
        else:
            raise KeyError(variable)

    @staticmethod
    def assert_valid_parameters(parameters):
        assert np.all(np.isfinite(parameters['mean'])), "mean must be finite"
        # TODO: positive definite precision


class WishartLikelihood(Likelihood):
    def __init__(self, x, shape, scale):
        super(WishartLikelihood, self).__init__(x=x, shape=shape, scale=scale)

    def evaluate(self):
        assert_constant(self['shape'])
        assert_constant(self['scale'])
        x_logdet = self._es('x', 'logdet')
        x_1 = self._es('x', 1)
        shape_1 = self._es('shape', 1)
        scale_1 = self._es('scale', 1)
        p = scale_1.shape[-1]
        scale_logdet = self._es('scale', 'logdet')

        return 0.5 * x_logdet * (shape_1 - p - 1.0) - 0.5 * np.sum(scale_1 * x_1, axis=(-1, -2)) -\
             0.5 * shape_1 * p * np.log(2) - scipy.special.multigammaln(0.5 * shape_1, p) + 0.5 * \
             shape_1 * scale_logdet

    def _natural_parameters(self, variable):
        scale_1 = self._es('scale', 1)
        p = scale_1.shape[-1]
        if variable == 'x':
            return {
                'logdet': 0.5 * (self._es('shape', 1) - p - 1) * np.ones(scale_1.shape[:-2]),
                'mean': -0.5 * scale_1,
            }
        else:
            raise KeyError(variable)

    @staticmethod
    def assert_valid_parameters(parameters):
        pass
