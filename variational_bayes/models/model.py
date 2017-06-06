import numbers

from ..util import *
from ..distributions import Distribution, ReshapedDistribution


def evaluate_natural_parameters(factor, likelihoods, exclude=None):
    """
    Obtain the natural parameters for a given factor.
    """
    # Iterate over all likelihoods
    args = []
    for likelihood in likelihoods:
        if exclude and likelihood in exclude:
            continue
        # Check if this factor is part of the likelihood
        parameter = likelihood.parameter_name(factor)
        if parameter:
            # Get the natural parameters
            natural_parameters = likelihood.natural_parameters(parameter)
            """
            # Check if the distribution was reshaped and apply the transforms if necessary
            if isinstance(likelihood.parameters[parameter], ReshapedDistribution):
                natural_parameters = {key: np.reshape(value, (-1, *getattr(factor, key).shape))
                                      for key, value in natural_parameters.items()}
            """
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
        if len(elbos) < self.num_steps:
            return False
        return elbos[-1] - elbos[-self.num_steps] < self.threshold


class Model:
    """
    Model combining factors of the posterior and likelihoods.
    """
    def __init__(self, factors, likelihoods):
        self._factors = factors
        self._likelihoods = likelihoods

        # Run over all the likelihoods, extract the 'x' parameter and ensure the factors all have
        # a prior
        lookup = {v: k for k, v in factors.items()}
        priors = {k: [] for k in factors.values()}

        for likelihood in self._likelihoods:
            x = likelihood.x
            if isinstance(x, Distribution):
                priors[x].append(likelihood)

        # Run over the priors and ensure they all have exactly one prior
        for factor, p in priors.items():
            assert p, "%s does not have a prior" % lookup[factor]
            assert len(p) < 2, "%s has more than one prior: %s" % (lookup[factor], p)

    def __getitem__(self, name):
        return self._factors[name]

    def update(self, steps, tqdm=None, order=None, convergence_predicate=None):
        """
        Update the factors of the approximate posterior.

        Parameters
        ----------
        steps : int, iterable or None
            number or iterable of update steps. If `None`, an infinite iterator is used and
            `convergence_predicate` must be provided
        tqdm : None or callable
            progress indicator
        order : list[str]
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
        if order is None:
            order = self._factors
        if isinstance(convergence_predicate, numbers.Real):
            convergence_predicate = ConvergencePredicate(convergence_predicate, 1)

        elbo = [self.elbo]

        for _ in steps:
            # Update each factor
            for factor in order:
                self.update_factor(factor)

            elbo.append(self.elbo)
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

    def aggregate_natural_parameters(self, factor, exclude=None):
        if isinstance(factor, str):
            factor = self._factors[factor]
        return aggregate_natural_parameters(factor, self._likelihoods, exclude)

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
        return np.sum([np.sum(likelihood.evaluate()) for likelihood in self._likelihoods])

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


class ModelEnsemble:
    """
    Ensemble of models to ensure we don't converge to a local optimum.

    Parameters
    ----------
    model_init : callable
        callable to create a model
    model_args : iterable
        sequence of arguments passed to `model_init`
    keep_models : bool
        whether to store all models (may have a significant memory footprint)
    """
    def __init__(self, model_init, model_args=None, keep_models=False):
        self.keep_models = keep_models
        self.model_init = model_init
        self.model_args = model_args or []
        self.elbos = []
        self.converged = []
        self._models = []
        self.best_model = None
        self.best_elbo = -np.inf

    @property
    def models(self):
        """list[Model] : sequence of models"""
        assert self.keep_models, "set `keep_models` to `True` to store models"
        return self._models

    def update(self, num_models, steps, tqdm=None, order=None, convergence_predicate=None):
        """
        Update models in the ensemble.

        Parameters
        ----------
        num_models : int
            number of models to update.
        steps : int
            see `Model.update`.
        tqdm : callable
            progress indicator.
        order : list[str]
            see `Model.update`.
        convergence_predicate : float or callable
            see `Model.update`.
        """
        if isinstance(num_models, numbers.Integral):
            num_models = range(num_models)
        if tqdm:
            num_models = tqdm(num_models)
        # Iterate over the models
        for _ in num_models:
            model = self.model_init(*self.model_args)
            # Optimise the model
            elbos, converged = model.update(
                steps, order=order, convergence_predicate=convergence_predicate
            )
            # Update the state
            elbo = elbos[-1]
            self.elbos.append(elbo)
            self.converged.append(converged)
            if self.keep_models:
                self._models.append(model)
            # Check for the best model
            if elbo > self.best_elbo:
                self.best_elbo = elbo
                self.best_model = model

        return self.best_model
