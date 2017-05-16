import numpy as np

from .likelihood import Likelihood
from .distribution import s
from .categorical import CategoricalDistribution
from ..util import pad_dims, sum_trailing_dims, softmax


class MixtureLikelihood(Likelihood):
    """
    Mixture likelihood.
    """
    def __init__(self, z, likelihood, **kwargs):
        super(MixtureLikelihood, self).__init__(z=z, likelihood=likelihood, **kwargs)

    @staticmethod
    def evaluate(z, likelihood, **kwargs):   # pylint: disable=W0221
        likelihood = likelihood.evaluate(**kwargs)
        z_1 = s(z, 1)
        assert z_1.ndim == 2, "indicator must be two-dimensional"
        assert z_1.shape == likelihood.shape[:2], "leading dimensions of the likelihood %s and " \
            "indicators %s must match" % (likelihood.shape, z_1.shape)
        # Aggregate with respect to the indicator dimension
        return np.sum(pad_dims(z_1, likelihood.ndim) * likelihood, axis=1)

    @staticmethod
    def natural_parameters(variable, z, likelihood, **kwargs):  # pylint: disable=W0221
        if variable == 'z':
            # Just evaluate the responsibilities and aggregate any trailing dimensions
            likelihood = likelihood.evaluate(**kwargs)
            return {
                'mean': sum_trailing_dims(likelihood, 2),
            }
        else:
            z_1 = s(z, 1)
            assert z_1.ndim == 2, "indicator must be two-dimensional"
            # Iterate over the natural parameters of the likelihood
            natural_parameters = {}
            for key, value in likelihood.natural_parameters(variable, **kwargs).items():
                # We assume that the natural parameters have shape `(n, k, ...)`, where `n` is the
                # number of observations, `k` is the number of mixture components, and `...` is the
                # shape associated with the natural parameter
                assert z_1.shape == value.shape[:2], "leading dimensions %s of the natural " \
                    "parameters %s of %s and the indicators %s must match" % \
                    (value.shape, key, variable, z_1.shape)
                # Sum over the dimension corresponding to samples
                natural_parameters[key] = np.sum(pad_dims(z_1, value.ndim) * value, axis=0)
            return natural_parameters


class InteractingMixtureLikelihood(Likelihood):
    def __init__(self, z, likelihood, **kwargs):
        super(InteractingMixtureLikelihood, self).__init__(z=z, likelihood=likelihood, **kwargs)

    @staticmethod
    def zz(z):
        # Compute the expected outer product
        z_1 = s(z, 1)
        assert z_1.ndim == 2, "indicator must be two-dimensional"
        n, _ = z_1.shape
        zz = z_1[:, None, :, None] * z_1[None, :, None, :]
        i = np.arange(n)
        zz[i, i] += s(z, 'cov')
        return zz

    @staticmethod
    def evaluate(z, likelihood, **kwargs):
        likelihood = likelihood.evaluate(**kwargs)
        zz = InteractingMixtureLikelihood.zz(z)
        # Aggregate with respect to the indicator dimensions
        return np.sum(pad_dims(zz, likelihood.ndim) * likelihood, axis=(2, 3))

    @staticmethod
    def natural_parameters(variable, z, likelihood, **kwargs):
        if variable == 'z':
            raise NotImplementedError("Natural parameters for indicators cannot be obtained jointly")
        else:
            zz = InteractingMixtureLikelihood.zz(z)
            # Iterate over the natural parameters of the likelihood
            natural_parameters = {}
            for key, value in likelihood.natural_parameters(variable, **kwargs).items():
                # We assume that the natural parameters have shape `(n, n, k, k, ...)` where `n` is
                # the number of observations, `k` is the number of mixture components, and `...` is
                # the shape associated with the natural parameter
                assert zz.shape == value.shape[:4], "leading dimensions %s of the natural " \
                    "parameters %s of %s and the indicators %s must match" % \
                    (value.shape, key, variable, zz.shape)
                # Sum over the dimensions corresponding to samples
                natural_parameters[key] = np.sum(pad_dims(zz, value.ndim) * value, axis=(0, 1))
            return natural_parameters

    @staticmethod
    def indicator_natural_parameters(natural_parameters, z, likelihood, **kwargs):
        """
        Update the indicator variables for component membership.

        The natural parameters of the indicators cannot be obtained jointly and we need to run an
        iterative update scheme in which each indicator variable is updated in turn.

        Parameters
        ----------
        z : np.ndarray or CategoricalDistribution
            indicator variables for component membership
        natural_parameters : dict
            natural parameters or messages sent to the indicator variables from the rest of the
            Bayesian network
        likelihood : Likelihood
            likelihood to evaluate the likelihood of `args`
        args : list
            parameters passed to the evaluation of `likelihood`
        """
        # Start with the initial probabilities
        proba = s(z, 1)
        assert proba.ndim == 2, "indicator must be two-dimensional"
        n, k = proba.shape

        # Unpack the natural parameter associated with the mean because it is the only relevant one
        natural_parameters = natural_parameters['mean']
        # Ensure the shape is broadcast
        natural_parameters = np.ones(np.broadcast(natural_parameters, proba).shape) * natural_parameters
        #ndim = np.ndim(natural_parameters)
        #assert np.shape(natural_parameters) == np.shape(proba)[:-ndim], "trailing shape %s of the natural " \
        #    "parameter associated with the mean must match the shape %s of the expected indicators" % \
        #    (np.shape(natural_parameters), np.shape(proba))

        # Evaluate the likelihood
        likelihood = likelihood.evaluate(**kwargs)
        assert likelihood.ndim >= 4, "evaluated likelihood must have at least four dimensions"
        assert likelihood.shape[:4] == (n, n, k, k), "evaluated likelihood must have leading shape " \
            "%s" % [(n, n, k, k)]
        # Collapse the trailing dimensions
        likelihood = sum_trailing_dims(likelihood, 4)

        # Iterate over the observations
        for i in range(n):
            # Initialize the natural parameters for this observation
            _np = natural_parameters[i]
            # Add the diagonal term
            _np += np.diag(likelihood[i, i])
            # Add the interaction terms
            _np += np.einsum('jb,jab', proba, likelihood[i])
            # Subtract the self interaction (we shouldn't have added this term in the first place
            # but the summation is easier this way)
            _np -= np.sum(likelihood[i, i], axis=1)
            # Set the probability of the observations
            proba[i] = softmax(_np)

        return {
            'mean': np.log(proba)
        }
