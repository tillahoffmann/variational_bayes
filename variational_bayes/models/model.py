import numbers
import logging
import numpy as np

from ..distributions import Distribution, DerivedDistribution
from ..util import timeit


logger = logging.getLogger(__name__)


def evaluate_natural_parameters(factor, likelihoods, exclude=None):
    """
    Obtain the natural parameters for a given factor.
    """
    # Iterate over all likelihoods
    args = []
    for likelihood in likelihoods:
        if exclude and likelihood in exclude:
            continue
        # Get the natural parameters
        natural_parameters = likelihood.natural_parameters(factor)
        if natural_parameters:
            args.append(natural_parameters)
    return args


def aggregate_natural_parameters(factor, likelihoods, exclude=None):
    return factor.aggregate_natural_parameters(
        evaluate_natural_parameters(factor, likelihoods, exclude)
    )


class ConvergencePredicate:
    """
    Returns `True` if the provided sequence of ELBOs does not increase by more than `threshold`
    over `num_steps`.
    """
    def __init__(self, threshold, num_steps):
        self.threshold = threshold
        self.num_steps = num_steps

    def __call__(self, elbos):
        if len(elbos) < self.num_steps + 1:
            return False
        return elbos[-1] - elbos[-(self.num_steps + 1)] < self.threshold


class Model:
    """
    Model combining factors of the posterior and likelihoods.
    """
    def __init__(self, factors, likelihoods, update_order=None):
        self._factors = factors
        self._likelihoods = likelihoods
        self._update_order = update_order
        self._update_times = {}
        self._aggregate_natural_parameters_times = {}

        # Run over all the likelihoods, extract the 'x' parameter and ensure the factors all have
        # a prior
        lookup = {v: k for k, v in factors.items()}
        priors = {v: [] for v in factors.values()}

        for likelihood in self._likelihoods:
            x = likelihood.x
            if isinstance(x, Distribution):
                priors[x].append(likelihood)

        # Run over the factors and ensure they all have exactly one prior
        for factor, p in priors.items():
            assert p, "%s does not have a prior" % lookup[factor]
            assert len(p) < 2, "%s has more than one prior: %s" % (lookup[factor], p)

        # Ensure only real distributions are added as factors and no child distributions
        for key, value in self._factors.items():
            assert not isinstance(value, DerivedDistribution), "%s is a ChildDistribution" % key

    def __getitem__(self, name):
        return self._factors[name]

    def update(self, steps, tqdm=None, update_order=None, convergence_predicate=None, **kwargs):
        """
        Update the factors of the approximate posterior.

        Parameters
        ----------
        steps : int, iterable or None
            number or iterable of update steps. If `None`, an infinite iterator is used and
            `convergence_predicate` must be provided
        tqdm : None or callable
            progress indicator
        update_order : list[str]
            order in which to update the parameters
        convergence_predicate : float or callable
            predicate to assess convergence. If a `float`, denotes the change in evidence lower
            bound of successive iterations that is considered to have converged. If a `callable`,
            should accept the a sequence of elbos and return `True` if the optimisation has
            converged.

        Returns
        -------
        elbo : np.ndarray
            trace of the evidence lower bound
        """
        if isinstance(steps, numbers.Integral):
            steps = range(steps)
        elif steps is None:
            assert convergence_predicate is not None, "a convergence predicate must be provided " \
                "if the number of steps is not fixed"
            # Define an infinite iterator (http://stackoverflow.com/a/5739258/1150961)
            steps = iter(int, 1)
        if tqdm:
            steps = tqdm(steps)
        if isinstance(convergence_predicate, numbers.Real):
            convergence_predicate = ConvergencePredicate(convergence_predicate, 1)

        update_order = update_order or (self._update_order or self._factors)

        elbo = [self.elbo]

        for _ in steps:
            # Update each factor
            for factor in update_order:
                self.update_factor(factor, **kwargs)

            elbo.append(self.elbo)

            improvement = elbo[-1] - elbo[-2]
            if improvement < - 100 * np.finfo(np.float32).eps:
                logger.warning("ELBO decreased by %g from %g to %g", - improvement, elbo[-2],
                               elbo[-1])

            if convergence_predicate and convergence_predicate(elbo):
                break

        return np.asarray(elbo), \
            None if convergence_predicate is None else convergence_predicate(elbo)

    def natural_parameters(self, factor, exclude=None):
        """
        Obtain the natural parameters for a given factor.
        """
        if isinstance(factor, str):
            factor = self._factors[factor]
        return evaluate_natural_parameters(factor, self._likelihoods, exclude)

    def aggregate_natural_parameters(self, factor, exclude=None, **kwargs):
        if isinstance(factor, str):
            factor = self._factors[factor]
        return aggregate_natural_parameters(factor, self._likelihoods, exclude, **kwargs)

    def update_factor(self, factor, **kwargs):
        """
        Update the given factor.
        """
        if isinstance(factor, str):
            factor = self._factors[factor]
        # Construct the sequence of natural parameters used to update this factor
        with timeit(self._aggregate_natural_parameters_times, factor):
            natural_parameters = self.aggregate_natural_parameters(factor, **kwargs)
        assert natural_parameters, "failed to update %s because natural parameters are " \
            "missing" % factor
        with timeit(self._update_times, factor):
            factor.update_from_natural_parameters(natural_parameters)
        return factor

    @property
    def joint(self):
        """float : expected joint distribution"""
        return np.sum(self.joint_terms)

    @property
    def joint_terms(self):
        return [np.sum(likelihood.evaluate()) for likelihood in self._likelihoods]

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
