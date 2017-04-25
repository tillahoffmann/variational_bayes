import numbers

from .util import *


class Model(Container):
    """
    Model combining factors of the posterior and likelihoods.
    """
    def __init__(self, factors, likelihoods):
        super(Model, self).__init__(**factors)
        self._likelihoods = likelihoods

    def update(self, steps, tqdm=None):
        """
        Update the factors of posterior.

        Parameters
        ----------
        steps : int or iterable
            update steps
        tqdm : None or callable
            progress indicator

        Returns
        -------
        elbo : np.ndarray
            trace of the evidence lower bound
        """
        if isinstance(steps, numbers.Integral):
            steps = range(steps)
        if tqdm:
            steps = tqdm(steps)

        elbo = [self.elbo]

        for _ in steps:
            # Update each factor
            for name, factor in self._attributes.items():
                # These are the messages in variational message passing
                natural_parameters = factor.aggregate_natural_parameters(
                    [likelihood.natural_parameters(factor) for likelihood in self._likelihoods]
                )
                assert natural_parameters, "failed to update '%s' because natural parameters are " \
                    "missing" % name
                factor.update(natural_parameters)

            elbo.append(self.elbo)

        return np.asarray(elbo)

    @property
    def joint(self):
        """float : expected joint distribution"""
        return np.sum([np.sum(likelihood.evaluate()) for likelihood in self._likelihoods])

    @property
    def entropies(self):
        """list[float] : sequence of entropies"""
        return {name: factor.entropy for name, factor in self._attributes.items()}

    @property
    def entropy(self):
        """float : total entropy"""
        return np.sum([np.sum(entropy) for entropy in self.entropies.values()])

    @property
    def elbo(self):
        """float : evidence lower bound"""
        return self.joint + self.entropy
