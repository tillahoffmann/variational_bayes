import numbers
import functools as ft
import multiprocessing
import numpy as np


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

    def update(self, num_models, steps, tqdm=None, update_order=None, convergence_predicate=None,
               num_processes=1):
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
        update_order : list[str]
            see `Model.update`.
        convergence_predicate : float or callable
            see `Model.update`.
        """
        if isinstance(num_models, numbers.Integral):
            num_models = range(num_models)

        if num_processes == 1:
            _map = map
            pool = None
        else:
            pool = multiprocessing.Pool(num_processes)
            _map = pool.imap_unordered

        # Iterate over the models
        try:
            partial = ft.partial(self._optimize_single_model, steps, update_order, convergence_predicate)
            generator = _map(partial, num_models)
            if tqdm:
                generator = tqdm(generator, total=len(num_models))
            for model, elbo, converged in generator:
                self.elbos.append(elbo)
                self.converged.append(converged)
                if self.keep_models:
                    self._models.append(model)
                # Check for the best model
                if elbo > self.best_elbo:
                    self.best_elbo = elbo
                    self.best_model = model
        finally:
            # Clean up multiprocessing resources
            if pool:
                pool.close()

        return self.best_model

    def _optimize_single_model(self, steps, update_order, convergence_predicate, *_):
        model = self.model_init(*self.model_args)
        # Optimise the model
        elbos, converged = model.update(
            steps, update_order=update_order, convergence_predicate=convergence_predicate
        )
        return model, elbos[-1], converged
