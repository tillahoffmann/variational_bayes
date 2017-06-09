import numpy as np

from ..distributions import NormalDistribution, GammaDistribution, MultiNormalDistribution, \
    MixtureDistribution, InteractingMixtureDistribution, CategoricalDistribution, \
    DirichletDistribution, VARAdjacencyDistribution, VARBiasDistribution, VARDistribution, \
    ReshapedDistribution, WishartDistribution, Distribution
from .interacting_mixture_model import InteractingMixtureModel


def var_model(x, order, num_groups, given=None):
    """
    Build a hierarchical vector-autoregressive model.

    Given is the set of factors that is assumed known for debugging purposes.
    """
    given = given or {}
    num_steps, num_nodes = x.shape

    epsilon = 1e-6

    factors = {
        'coefficients': MultiNormalDistribution(
            np.random.normal(0, epsilon, (num_nodes, order * num_nodes + 1)),
            np.ones((num_nodes, 1, 1)) * np.eye(order * num_nodes + 1) * epsilon
        ),
        'density': DirichletDistribution(
            np.random.uniform(1 - epsilon, 1 + epsilon, num_groups)
        ),
        'z': CategoricalDistribution(
            np.random.dirichlet(np.ones(num_groups) / epsilon, num_nodes),
        ),
        'bias_mean': NormalDistribution(
            np.random.normal(0, epsilon, num_groups),
            np.ones(num_groups) * epsilon,
        ),
        'bias_precision': GammaDistribution(
            1e-3 + np.random.normal(0, epsilon, num_groups),
            1e-3 * np.ones(num_groups),
        ),
        'adjacency_mean': MultiNormalDistribution(
            np.random.normal(0, epsilon, (num_groups, num_groups, order)),
            np.ones((num_groups, num_groups, 1, 1)) * np.eye(order) * epsilon
        ),
        'adjacency_precision': WishartDistribution(
            order - 1 + 1e-3 + np.random.normal(0, epsilon, (num_groups, num_groups)),
            (order - 1 + 1e-3) * np.ones((num_groups, num_groups, 1, 1)) * np.eye(order)
        ),
        'noise_precision': GammaDistribution(
            1e-3 + np.random.normal(0, epsilon, num_groups),
            1e-3 * np.ones(num_groups),
        )
    }

    factors.update(given or {})

    # Extract the factors
    q_coefficients = factors['coefficients']
    q_z = factors['z']
    q_bias_mean = factors['bias_mean']
    q_bias_precision = factors['bias_precision']
    q_adjacency_mean = factors['adjacency_mean']
    q_adjacency_precision = factors['adjacency_precision']
    q_noise_precision = factors['noise_precision']
    q_density = factors['density']
    q_adjacency = VARAdjacencyDistribution(q_coefficients)
    q_bias = VARBiasDistribution(q_coefficients)


    likelihoods = [
        MixtureDistribution(q_z, NormalDistribution(q_bias_mean, q_bias_precision)).likelihood(
            ReshapedDistribution(q_bias, (num_nodes, 1))
        ),
        InteractingMixtureDistribution(
            q_z, MultiNormalDistribution(q_adjacency_mean, q_adjacency_precision)
        ).likelihood(
            ReshapedDistribution(q_adjacency, (num_nodes, num_nodes, 1, 1, order))
        ),
        CategoricalDistribution(q_density).likelihood(q_z),
        VARDistribution(q_z, q_coefficients, q_noise_precision).likelihood(
            VARDistribution.summary_statistics(x, order)
        ),
        # Prior for the density
        DirichletDistribution(np.ones(num_groups)).likelihood(q_density),
        # Priors for the bias
        NormalDistribution(0, 1e-4).likelihood(q_bias_mean),
        GammaDistribution(1e-6, 1e-6).likelihood(q_bias_precision),
        # Priors for the adjacency
        MultiNormalDistribution(np.zeros(order), 1e-4 * np.eye(order)).likelihood(q_adjacency_mean),
        WishartDistribution(order - 1 + 1e-6, np.eye(order) * 1e-6).likelihood(q_adjacency_precision)
    ]

    factors = {key: value for key, value in factors.items() if isinstance(value, Distribution)}
    model = InteractingMixtureModel(factors, likelihoods)
    return model
