import numpy as np

from .distribution import ChildDistribution, statistic


class ReshapedDistribution(ChildDistribution):
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
        self._newshape = newshape
        super(ReshapedDistribution, self).__init__(parent)

    def _reshaped_statistic(self, statistic):
        # Get the statistic
        value = getattr(self._parent, statistic)
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

    @statistic
    def mean(self):
        return self._reshaped_statistic('mean')

    @statistic
    def var(self):
        return self._reshaped_statistic('var')

    @statistic
    def entropy(self):
        return self._reshaped_statistic('entropy')

    @statistic
    def outer(self):
        return self._reshaped_statistic('outer')

    def assert_valid_parameters(self):
        super(ReshapedDistribution, self).assert_valid_parameters()

    def log_proba(self, x):
        # Evaluate the probability of the parent distribution and then reshape
        log_proba = self._parent.log_proba(x)
        return np.reshape(log_proba, self._newshape)

    @property
    def _repr_parameters(self):
        return ["parent=%s" % self._parent, "newshape=%s" % [self._newshape]]

    def transform_natural_parameters(self, natural_parameters):
        return {key: np.reshape(value, (-1, ) + getattr(self._parent, key).shape)
                for key, value in natural_parameters.items()}
