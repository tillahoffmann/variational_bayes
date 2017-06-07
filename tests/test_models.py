import numpy as np
import variational_bayes as vb


def test_model():
    x = np.random.normal(0, 1, 100)

    # Define factors
    q_mean = vb.NormalDistribution(0, 1e-3)
    q_precision = vb.GammaDistribution(1e-3, 1e-3)

    # Define likelihoods
    likelihood = vb.NormalDistribution(q_mean, q_precision).likelihood(x)
    mean_prior = vb.NormalDistribution(0, 1e-4).likelihood(q_mean)
    precision_prior = vb.GammaDistribution(1e-3, 1e-3).likelihood(q_precision)

    # Define the model and update
    model = vb.Model({'mean': q_mean, 'precision': q_precision},
                     [likelihood, mean_prior, precision_prior])
    model.update(100)

    # Ensure the values are sensible
    assert np.abs(q_mean.mean / q_mean.std) < 5, "unexpected mean"
    assert np.abs((q_precision.mean - 1) / q_precision.std) < 5, "unexpected precision"


def test_reshape():
    # Generate data
    x = np.random.normal(0, 1, (100, 5, 5))
    # Define factors and likelihood (using reshape)
    q = vb.NormalDistribution(np.zeros(25), np.ones(25) * 1e-3)
    reshaped = vb.ReshapedDistribution(q, (5, 5))
    likelihood = vb.NormalDistribution(reshaped, 1).likelihood(x)
    model = vb.Model({'mean': q}, [likelihood, vb.NormalDistribution(0, 1e-3).likelihood(q)])

    # Check the natural parameters
    for factor in ['mean', q]:
        natural_parameters = model.aggregate_natural_parameters(factor)
        np.testing.assert_allclose(natural_parameters['mean'], np.sum(x, axis=0).ravel(),
                                   err_msg="'mean' natural parameter does not match")
        np.testing.assert_allclose(natural_parameters['square'], - 0.5 * (100 + 1e-3),
                                   err_msg="'square' natural parameter does not match")
