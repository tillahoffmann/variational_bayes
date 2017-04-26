import numbers

from .util import *


class Model:
    """
    Model combining factors of the posterior and likelihoods.
    """
    def __init__(self, factors, likelihoods):
        self._factors = factors
        self._likelihoods = likelihoods

    def __getitem__(self, name):
        return self._factors[name]

    def update(self, steps, tqdm=None):
        """
        Update the factors of the approximate posterior.

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
            for factor in self._factors:
                self.update_factor(factor)

            elbo.append(self.elbo)

        return np.asarray(elbo)

    def natural_parameters(self, factor):
        """
        Obtain the natural parameters for a given factor.
        """
        if isinstance(factor, str):
            factor = self._factors[factor]
        # Iterate over all likelihoods
        natural_parameters = []
        for likelihood in self._likelihoods:
            # Check if this factor is part of the likelihood
            parameter = likelihood.parameter_name(factor)
            if parameter:
                natural_parameters.append(
                    likelihood.natural_parameters(parameter, **likelihood.parameters)
                )
        return natural_parameters

    def aggregate_natural_parameters(self, factor):
        if isinstance(factor, str):
            factor = self._factors[factor]
        return factor.aggregate_natural_parameters(self.natural_parameters(factor))

    def update_factor(self, factor):
        """
        Update the given factor.
        """
        if isinstance(factor, str):
            factor = self._factors[factor]
        # Construct the sequence of natural parameters used to update this factor
        natural_parameters = self.aggregate_natural_parameters(factor)
        assert natural_parameters, "failed to update %s because natural parameters are " \
            "missing" % factor
        factor.update_from_natural_parameters(natural_parameters)

    @property
    def joint(self):
        """float : expected joint distribution"""
        return np.sum([np.sum(likelihood.evaluate(**likelihood.parameters))
                       for likelihood in self._likelihoods])

    @property
    def entropies(self):
        """list[float] : sequence of entropies"""
        return {name: factor.entropy for name, factor in self._factors.items()}

    @property
    def entropy(self):
        """float : total entropy"""
        return np.sum([np.sum(entropy) for entropy in self.entropies.values()])

    @property
    def elbo(self):
        """float : evidence lower bound"""
        return self.joint + self.entropy
