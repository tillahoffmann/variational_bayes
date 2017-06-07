class Likelihood:
    def __init__(self, distribution, x):
        self.distribution = distribution
        self.x = x

    def natural_parameters(self, variable):
        """
        Evaluate the natural parameters associated with `variable` given observation `x`.
        """
        return self.distribution.natural_parameters(self.x, variable)

    def evaluate(self):
        """
        Evaluate the expected log-probability given observation `x`.
        """
        return self.distribution.log_proba(self.x)

    def parameter_name(self, parameter):
        """
        Get the name of the given parameter.
        """
        if parameter is self.x:
            return 'x'
        else:
            return self.distribution.parameter_name(parameter)

    def __repr__(self):
        return "Likelhood(%s, %s)" % (self.distribution, self.x)
