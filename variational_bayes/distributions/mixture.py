import numpy as np
from .distribution import Distribution, s
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
        if variable == 'z':
            # Evaluate the log probability of the observations under the individual distributions
            return {
                'mean': self._parent.log_proba(x)
            }
        else:
            # Get the natural parameters of the parent distribution and compute the
            # indicator-weighted mean
            natural_parameters = self._parent.natural_parameters(x, variable)
            z = s(self._z, 1)
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
        if variable == 'z':
            raise NotImplementedError
        else:
            # Get the natural parameters from the parent distribution which should have shape
            # `(n, n, k, k, ...)` where `...` denotes any sample dimensions
            natural_parameters = self._parent.natural_parameters(x, variable)
            zz = s(self.z, 'interaction')
            # Aggregate over the two leading dimensions
            for key, value in natural_parameters.items():
                # Aggregate the parameters. The indicators need to be padded.
                natural_parameters[key] = np.sum(pad_dims(zz, value.ndim) * value, (0, 1))
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
        proba = s(self.z, 1)
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
        for i in range(n):
            # Initialize the natural parameters for this observation
            _np = natural_parameters[i]
            # Add the diagonal term
            _np += np.diag(likelihood[i, i])
            # Add the interaction terms
            _np += np.einsum('jl,jkl', proba, likelihood[i])
            # Subtract the term that we have erroneously added above, i.e.
            # \E{z_{il}} \E{\log P(x_{ii}|theta_{kl})}
            _np -= np.einsum('l,kl', proba[i], likelihood[i, i])
            # Set the probability of the observations
            proba[i] = softmax(_np)

        return {
            'mean': np.log(proba)
        }
