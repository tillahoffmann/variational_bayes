from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
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


def plot_comparison(distribution, reference, scale=3, diagonal=True, aspect=None, ax=None,
                    **kwargs):
    default_kwargs = {
        'ls': 'none',
    }
    default_kwargs.update(kwargs)
    ax = ax or plt.gca()
    if aspect:
        ax.set_aspect(aspect)
    if diagonal:
        xy = np.min(reference), np.max(reference)
        ax.plot(xy, xy, ls=':')
    return ax.errorbar(reference.ravel(), distribution.mean.ravel(), distribution.std.ravel() *
                       scale, **default_kwargs)


def plot_residuals(distribution, reference, scale=3, zeroline=True, ax=None, **kwargs):
    default_kwargs = {
        'ls': 'none',
    }
    default_kwargs.update(kwargs)
    ax = ax or plt.gca()
    if zeroline:
        ax.axhline(0, ls=':')
    return ax.errorbar(reference.ravel(), distribution.mean.ravel() - reference.ravel(),
                       distribution.std.ravel() * scale, **default_kwargs)


def ellipse_from_cov(xy, cov, scale=3, **kwargs):
    """
    Create an ellipse from a covariance matrix.

    Parameters
    ----------
    xy : np.ndarray
        position of the ellipse
    cov : np.ndarray
        covariance matrix
    scale : float
        scale of the ellipse (default is three standard deviations)
    kwargs : dict
        keyword arguments passed on to `matplotlib.patches.Ellipse`

    Returns
    -------
    ellipse
    """
    evals, evecs = np.linalg.eigh(cov)
    # Get the angle associated with the dominant eigenvector
    angle = np.arctan2(*evecs[::-1, 1])
    height, width = scale * np.sqrt(evals)
    return mpatches.Ellipse(xy, width, height, np.rad2deg(angle), **kwargs)
