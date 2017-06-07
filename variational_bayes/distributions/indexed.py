import numpy as np

from .distribution import ChildDistribution, s
from ..util import array_repr


class IndexedDistribution(ChildDistribution):
    """
    Distribution obtained by indexing a distribution.

    Parameters
    ----------
    parent : Distribution
        distribution to reshape
    z : np.ndarray | CategoricalDistribution
        indicator selecting elements from the batch shape of the parent distribution
    """
    def __init__(self, parent, z):
        self._z = z
        super(IndexedDistribution, self).__init__(parent)

    def _transformed_statistic(self, statistic):
        # Get the statistic
        value = s(self._parent, statistic)
        # Compute the weighted mean
        return np.dot(s(self._z, 1), value)

    def assert_valid_parameters(self):
        super(IndexedDistribution, self).assert_valid_parameters()

    def log_proba(self, x):
        # Evaluate the probability of the parent distribution
        log_proba = self._parent.log_proba(x)
        # Compute the weighted mean
        return np.dot(s(self._z, 1), log_proba)

    @property
    def _repr_parameters(self):
        return ["parent=%s" % self._parent, "z=%s" % array_repr(self._z)]

    def transform_natural_parameters(self, natural_parameters):
        return {key: np.einsum('ik,i...', s(self._z, 1), value)
                for key, value in natural_parameters.items()}
