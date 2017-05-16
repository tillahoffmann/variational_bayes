import numpy as np

from .distribution import Distribution, ReshapedDistribution


class Likelihood:
    def __init__(self, **parameters):
        self.parameters = {key: value if isinstance(value, Distribution) or
                                (isinstance(value, type) and issubclass(value, Likelihood))
                                else np.asarray(value) for key, value in parameters.items()}

    def parameter_name(self, x):
        """
        Get the parameter name of `x`.
        """
        for key, value in self.parameters.items():
            # Return the name of the parameter if the distribution matches or we have a reshaped
            # distribution whose parent matches
            if value is x or (isinstance(value, ReshapedDistribution) and value._distribution is x):
                return key
        return None

    @staticmethod
    def evaluate(x, *parameters):
        raise NotImplementedError

    @staticmethod
    def natural_parameters(variable, x, *parameters):
        raise NotImplementedError
