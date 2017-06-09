import numpy as np

from .distribution import DerivedDistribution, s, is_dependent


class ReshapedDistribution(DerivedDistribution):
    """
    Distribution with the same number of elements but different shape.

    Parameters
    ----------
    parent : Distribution
        distribution to reshape
    newshape : tuple
        new batch shape of the distribution (the sample dimension cannot be reshaped)
    """
    def __init__(self, parent, newshape):
        self._parent = parent
        self._newshape = newshape
        super(ReshapedDistribution, self).__init__(parent)

    def _transformed_statistic(self, statistic):
        # Get the statistic
        value = s(self._parent, statistic)
        # Determine the new shape by popping leading dimensions until the size of popped dimensions
        # matches the desired size
        newsize = np.prod(self._newshape)
        shape = list(value.shape)
        size = 1
        while size < newsize:
            size *= shape.pop(0)

        # Sanity check
        assert size == newsize, "cannot reshape leading dimensions"
        newshape = self._newshape + tuple(shape)
        return np.reshape(value, newshape)

    def assert_valid_parameters(self):
        super(ReshapedDistribution, self).assert_valid_parameters()

    def log_proba(self, x):
        # Evaluate the probability of the parent distribution and then reshape
        log_proba = self._parent.log_proba(x)
        return np.reshape(log_proba, self._newshape)

    @property
    def _repr_parameters(self):
        return ["parent=%s" % self._parent, "newshape=%s" % [self._newshape]]

    def transform_natural_parameters(self, distribution, natural_parameters):
        if is_dependent(self._parent, distribution):
            for key, value in natural_parameters.items():
                shape = s(self._parent, key).shape
                natural_parameters[key] = value.reshape(shape)
            return natural_parameters
        else:
            raise KeyError
