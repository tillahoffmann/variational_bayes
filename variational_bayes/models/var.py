import numpy as np

from ..distributions import NormalDistribution, GammaDistribution, MultiNormalDistribution, \
    MixtureDistribution, InteractingMixtureDistribution, CategoricalDistribution, \
    DirichletDistribution, VARAdjacencyDistribution, VARBiasDistribution, VARDistribution, \
    ReshapedDistribution, WishartDistribution, Distribution, VARDiagAdjacencyDistribution
from .interacting_mixture_model import InteractingMixtureModel


def var_model(x, order, num_groups, update_order=None, given=None, shared_noise=True,
              independent_diag=False, filter_nans=False):
    """
    Build a hierarchical vector-autoregressive model.

    Parameters
    ----------
    x : np.ndarray
        tensor of observations with shape `(t, n)`
    order : int
        order of the autoregressive process
    num_groups : int
        number of groups
    update_order : None
        order in which to update the parameters
    given : dict
        dictionary of values assumed known for debugging purposes keyed by the same name
        as the factors of the model
    uniform_ic : bool
        whether to use uniform initial conditions (except for the community assignments to break
        symmetry)
    shared_noise : bool
        whether to share the noise parameter within groups (quite a strong assumption and it makes
        generalising the model more difficult)
    independent_diag : bool
        whether to use independent priors for the diagonal of the adjacency matrix rather than
        imposing the same prior as for within-group interactions
    """
    given = given or {}
    _, num_nodes = x.shape

    epsilon = 1e-6

    if shared_noise:
        noise_precision = GammaDistribution(
            1e-3 + np.random.normal(0, epsilon, num_groups),
            1e-3 * np.ones(num_groups),
        )
    else:
        noise_precision = GammaDistribution(
            1e-3 + np.random.normal(0, epsilon, num_nodes),
            1e-3 * np.ones(num_nodes),
        )

    factors = {
        'coefficients': MultiNormalDistribution(
            np.zeros((num_nodes, order * num_nodes + 1)),
            np.ones((num_nodes, 1, 1)) * np.eye(order * num_nodes + 1) * epsilon
        ),
        'density': DirichletDistribution(
            np.ones(num_groups)
        ),
        'bias_mean': NormalDistribution(
            np.ones(num_groups),
            np.ones(num_groups) * epsilon,
        ),
        'bias_precision': GammaDistribution(
            epsilon * np.ones(num_groups),
            epsilon * np.ones(num_groups),
        ),
        'adjacency_mean': MultiNormalDistribution(
            np.zeros((num_groups, num_groups, order)),
            np.ones((num_groups, num_groups, 1, 1)) * np.eye(order) * epsilon
        ),
        'adjacency_precision': WishartDistribution(
            order * np.ones((num_groups, num_groups)),
            order * np.ones((num_groups, num_groups, 1, 1)) * np.eye(order)
        ),
        'noise_precision': noise_precision,
        # Only random initial conditions to break symmetry
        'z': CategoricalDistribution(
            np.random.dirichlet(np.ones(num_groups) / epsilon, num_nodes),
        ),
    }

    # Add factors for the diagonal of the adjacency matrix
    if independent_diag:
        factors['diag_mean'] = MultiNormalDistribution(
            np.zeros((num_groups, order)),
            np.ones((num_groups, 1, 1)) * np.eye(order) * epsilon
        )
        factors['diag_precision'] = WishartDistribution(
            order * np.ones(num_groups),
            order * np.ones((num_groups, 1, 1)) * np.eye(order)
        )

    given = given or {}
    factors.update(given)

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
    q_diag = VARDiagAdjacencyDistribution(q_adjacency)


    likelihoods = [
        MixtureDistribution(q_z, NormalDistribution(q_bias_mean, q_bias_precision)).likelihood(
            ReshapedDistribution(q_bias, (num_nodes, 1))
        ),
        InteractingMixtureDistribution(
            q_z, MultiNormalDistribution(q_adjacency_mean, q_adjacency_precision),
            self_interaction=not independent_diag
        ).likelihood(
            ReshapedDistribution(q_adjacency, (num_nodes, num_nodes, 1, 1))
        ),
        CategoricalDistribution(q_density).likelihood(q_z),
        VARDistribution(q_coefficients, q_noise_precision, q_z if shared_noise else None).likelihood(
            VARDistribution.summary_statistics(x, order, filter_nans)
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

    # Add likelihoods for the independent diagonal
    if independent_diag:
        likelihoods.extend([
            MixtureDistribution(q_z, MultiNormalDistribution(
                factors['diag_mean'], factors['diag_precision']
            )).likelihood(ReshapedDistribution(q_diag, (num_nodes, 1))),
            # Priors for the diagonal part
            MultiNormalDistribution(np.zeros(order), 1e-4 * np.eye(order))
                .likelihood(factors['diag_mean']),
            WishartDistribution(order - 1 + 1e-6, np.eye(order) * 1e-6)
                .likelihood(factors['diag_precision'])
        ])

    if independent_diag:
        default_update_order = [
            "coefficients", "noise_precision", "bias_mean", "bias_precision", "adjacency_mean",
            "adjacency_precision", 'diag_mean', 'diag_precision', "z", "density"
        ]
    else:
        default_update_order = [
            "coefficients", "noise_precision", "bias_mean", "bias_precision", "adjacency_mean",
            "adjacency_precision", "z", "density"
        ]
    update_order = update_order or default_update_order

    # Do not update fixed factors
    for key in given:
        update_order.remove(key)

    factors = {key: value for key, value in factors.items() if isinstance(value, Distribution)}
    model = InteractingMixtureModel(factors, likelihoods, update_order)
    model.adjacency = q_adjacency
    model.bias = q_bias
    model.adjacency_diag = q_diag
    return model
