
class Likelihood:
    def __init__(self, **parameters):
        self.parameters = parameters

    def __getattr__(self, name):
        if name.strip('_') in self.parameters:
            return self.parameters[name.strip('_')]
        else:
            raise AttributeError(name)

    def parameter_name(self, x):
        """
        Get the parameter name of `x`.
        """
        for key, value in self.parameters.items():
            if value is x:
                return key
        return None

    def evaluate(self):
        """
        Evaluate the expected log-likelihood.
        """
        raise NotImplementedError

    def natural_parameters(self, variable):
        """
        Evaluate the natural parameters of `variable`, i.e. the coefficients of the sufficient
        statistics.
        """
        raise NotImplementedError
