from .model import Model
from ..distributions import InteractingMixtureDistribution, Distribution


class InteractingMixtureModel(Model):
    def __init__(self, factors, likelihoods, update_order=None):
        super(InteractingMixtureModel, self).__init__(factors, likelihoods, update_order)
        # Determine the interacting mixture likelihood and the corresponding factor
        self._interacting_likelihood = None
        self._indicators = None
        for likelihood in self._likelihoods:
            if isinstance(likelihood.distribution, InteractingMixtureDistribution):
                assert self._interacting_likelihood is None, "there may only be one interacting " \
                    "mixture likelihood"
                self._interacting_likelihood = likelihood
                self._indicators = likelihood.distribution.z
                if isinstance(self._indicators, Distribution):
                    assert self._indicators in self._factors.values(), \
                        "indicators are not part of model"
                else:
                    # Ignore the indicators if they are fixed
                    self._indicators = None

    def aggregate_natural_parameters(self, factor, exclude=None, nodes=None):  # pylint:disable=W0221
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
            return self._interacting_likelihood.distribution.natural_parameters_z(
                self._interacting_likelihood.x, natural_parameters, nodes
            )
        else:
            return super(InteractingMixtureModel, self).aggregate_natural_parameters(
                factor, exclude
            )
