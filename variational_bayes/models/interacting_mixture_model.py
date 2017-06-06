from .model import *


class InteractingMixtureModel(Model):
    def __init__(self, factors, likelihoods):
        super(InteractingMixtureModel, self).__init__(factors, likelihoods)
        # Determine the interacting mixture likelihood and the corresponding factor
        self._interacting_likelihood = None
        self._indicators = None
        for likelihood in self._likelihoods:
            if isinstance(likelihood, InteractingMixtureLikelihood):
                assert self._interacting_likelihood is None, "there may only be one interacting " \
                    "mixture likelihood"
                self._interacting_likelihood = likelihood
                self._indicators = likelihood.parameters['z']
                assert self._indicators in self._factors.values(), "indicators are not part of model"

    def aggregate_natural_parameters(self, factor, exclude=None):
        if isinstance(factor, str):
            factor = self._factors[factor]

        if exclude is None:
            exclude = []

        if factor is self._indicators and self._interacting_likelihood not in exclude:
            # Get the natural parameters due to all other contributions
            natural_parameters = self.aggregate_natural_parameters(
                factor, [self._interacting_likelihood] + exclude
            )
            # Get the natural parameters after one sequence of interacting mixture model updates
            return self._interacting_likelihood.indicator_natural_parameters(
                natural_parameters, **self._interacting_likelihood.parameters
            )
        else:
            return super(InteractingMixtureModel, self).aggregate_natural_parameters(
                factor, exclude
            )
