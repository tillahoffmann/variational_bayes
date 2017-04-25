import numpy as np
import variational_bayes as vb


def test_model():
    x = np.random.normal(0, 1, 100)

    # Define factors
    q_mean = vb.NormalDistribution(0, 1e-3)
    q_precision = vb.GammaDistribution(1e-3, 1e-3)

    # Define likelihoods
    likelihood = vb.NormalLikelihood(x, q_mean, q_precision)
    mean_prior = vb.NormalLikelihood(q_mean, 0, 1e-4)
    precision_prior = vb.GammaLikelihood(q_precision, 1e-3, 1e-3)

    # Define the model and update
    model = vb.Model({'mean': q_mean, 'precision': q_precision},
                     [likelihood, mean_prior, precision_prior])
    model.update(100)

    # Ensure the values are sensible
    assert np.abs(q_mean.mean / q_mean.std) < 5, "unexpected mean"
    assert np.abs((q_precision.mean - 1) / q_precision.std) < 5, "unexpected precision"
