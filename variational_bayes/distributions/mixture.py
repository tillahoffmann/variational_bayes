import numpy as np
from .distribution import Distribution, s, is_dependent
from ..util import pad_dims, sum_trailing_dims, softmax


class MixtureDistribution(Distribution):  # pylint: disable=W0223
    """
    A mixture distribution.

    Parameters
    ----------
    z : np.ndarray | Distribution
        mixture indicators
    parent : type
        parent distribution of the mixture
    """
    def __init__(self, z, parent):
        self._parent = parent
        # We add the parameters of the parent to this distribution (which is a hack cf. #7)
        super(MixtureDistribution, self).__init__(z=z, **self._parent.parameters)

    def log_proba(self, x):
        # Evaluate the log probability of the observations under the individual distributions with
        # expected shape `(n, k)`, where `n` is the number of observations and `k` is the number
        # of groups
        log_proba = self._parent.log_proba(x)
        # Contract with the expected indicators of shape `(n, k)`
        return np.sum(s(self._z, 1) * log_proba, axis=-1)

    def natural_parameters(self, x, variable):
        if is_dependent(self._z, variable):
            # Evaluate the log probability of the observations under the individual distributions
            return {
                'mean': self._parent.log_proba(x)
            }

        # Get the natural parameters of the parent distribution
        natural_parameters = self._parent.natural_parameters(x, variable)
        if natural_parameters is None:
            return None

        z = s(self._z, 1)

        if is_dependent(x, variable):
            # Sum over the axis corresponding to different components
            axis = 1
        else:
            # We sum over the leading dimensions of the indicator but the last (which corresponds to
            # different components of the mixture)
            axis = tuple(range(z.ndim - 1))
        for key, value in natural_parameters.items():
            # Aggregate the parameters. The indicators need to be padded
            natural_parameters[key] = np.sum(pad_dims(z, value.ndim) * value, axis)
        return natural_parameters

    def assert_valid_parameters(self):
        z = s(self._z, 1)
        assert z.ndim > 0, "z must be at least one-dimensional"


class InteractingMixtureDistribution(Distribution):  # pylint: disable=W0223
    def __init__(self, z, parent):
        self._parent = parent
        # We add the parameters of the parent to this distribution (which is a hack cf. #7)
        super(InteractingMixtureDistribution, self).__init__(z=z, **self._parent.parameters)

    def assert_valid_parameters(self):
        z = s(self._z, 1)
        assert z.ndim == 2, "z must be two-dimensional"

    def log_proba(self, x):
        # Evaluate the log-probability of the parent distribution with expected shape `(n, n, k, k)`,
        # where `n` is the number of observations and `k` is the number of components
        log_proba = self._parent.log_proba(x)
        # Sum over the trailing dimensions weighted by the interaction
        zz = s(self.z, 'interaction')
        return np.sum(log_proba * pad_dims(zz, log_proba.ndim), axis=(0, 1))

    def natural_parameters(self, x, variable):
        if is_dependent(self._z, variable):
            raise NotImplementedError

        # Get the natural parameters from the parent distribution which should have shape
        # `(n, n, k, k, ...)` where `...` denotes any sample dimensions
        natural_parameters = self._parent.natural_parameters(x, variable)
        if natural_parameters is None:
            return None

        zz = s(self.z, 'interaction')
        if is_dependent(x, variable):
            # Aggregate over the indicators
            axis = (2, 3)
        else:
            # Aggregate over the observatios
            axis = (0, 1)

        for key, value in natural_parameters.items():
            # Aggregate the parameters. The indicators need to be padded.
            natural_parameters[key] = np.sum(pad_dims(zz, value.ndim) * value, axis)
        return natural_parameters

    def natural_parameters_z(self, x, natural_parameters):
        r"""
        Obtain the natural parameters for the component indicators.

        The log-likelihood for interacting mixtures takes the form
        $$
        \sum_{ijkl} z_{ik}z_{jl} \log P(x_{ij}|\theta_{kl}),
        $$
        where $x$ are observations, $z$ are component indicators, and $\theta$ are parameters. We
        are interested in the optimal posterior factor of the indicator $z_{ik}$ and take the
        expectation with respect to all other factors to find
        \begin{align}
        q_i &= \sum_{k} z_{ik} \E{\log P(x_{ii}|theta_{kk})} \\
            &\quad + \sum_{j\neq i, k, l} z_{ik}\E{z_{jl}} \E{\log P(x_{ij}|theta_{kl})}
        \end{align}

        Parameters
        ----------
        x : np.ndarray | Distribution
            observations
        natural_parameters : dict
            natural parameters due to the rest of the model

        Returns
        -------
        natural_parameters : dict
            natural parameters for the component indicators
        """
        # Start with the initial probabilities
        proba = np.copy(s(self.z, 1))
        assert proba.ndim == 2, "indicator must be two-dimensional"
        n, k = proba.shape

        # Unpack the natural parameter associated with the mean because it is the only relevant one
        natural_parameters = natural_parameters['mean']
        # Ensure the shape is broadcast
        natural_parameters = np.ones(np.broadcast(natural_parameters, proba).shape) * natural_parameters

        # Evaluate the likelihood
        likelihood = self._parent.log_proba(x)
        assert likelihood.ndim >= 4, "evaluated likelihood must have at least four dimensions"
        assert likelihood.shape[:4] == (n, n, k, k), "evaluated likelihood must have leading shape " \
            "%s" % [(n, n, k, k)]
        # Collapse the trailing dimensions
        likelihood = sum_trailing_dims(likelihood, 4)

        # Iterate over the observations
        for i in np.random.permutation(n):
            # Initialize the natural parameters for this observation
            _np = natural_parameters[i]
            # Add the diagonal term
            _np += np.diag(likelihood[i, i])
            # Add the interaction terms (but set the proba associated with i to zero everywhere
            # first to avoid double counting)
            proba[i] = 0
            _np += np.einsum('jl,jkl', proba, likelihood[i])
            # Set the probability of the observations
            proba[i] = softmax(_np)

        # Ignore divide by zero errors when one of the probabilities is zero
        old_settings = np.seterr(divide='ignore')
        natural_parameters = {
            'mean': np.log(proba)
        }
        np.seterr(**old_settings)
        return natural_parameters
