from matplotlib import pyplot as plt
import numpy as np


def plot_proba(distribution, start=None, stop=None, num=50, scale=3, reference=None, ax=None,
               **kwargs):
    assert distribution.sample_ndim == 0, "plotting probabilities is only supported for univariate " \
        "distributions"

    ax = ax or plt.gca()

    # Evaluate the pdf
    if start is None:
        start = distribution.mean - scale * distribution.std
    if stop is None:
        stop = distribution.mean + scale * distribution.std
    shape = (num, ) + distribution.batch_ndim * (1, )
    linx = start + np.reshape(np.linspace(0, 1, num), shape) * (stop - start)
    proba = np.exp(distribution.log_proba(linx))

    # Plot the pdf
    shape = (num, -1)
    lines = ax.plot(np.reshape(linx, shape), np.reshape(proba, shape), **kwargs)

    # Add reference values
    if reference is not None:
        for line, value in zip(lines, np.ravel(reference)):
            ax.axvline(value, color=line.get_color())

    return lines
