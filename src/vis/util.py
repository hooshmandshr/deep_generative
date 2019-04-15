"""General helper classes for visualization"""


import matplotlib.pyplot as plt
import numpy as np


def plot_elbo(elbo, skip=10, epoch=None, offset=0, **kwargs):
    """Plotter for elbo/loss rolling averages.

    params:
    -------
    elbo: list or np.array
        List of elbo per iteration.
    skip: int
        Window of averaging across time.
    epoch: None or int
        How many iterations make up an epoch. If None, x will be iteration
        number.
    offset: int
        Truncates the first offset iterations of the loss.
    **kwargs:
        parameters to be passed to matplotlib.plot
    """

    # Truncate pre offset
    elbo_ = np.array(elbo)[offset:]
    # Truncate a bit from the beginning so that average
    offset_2 = len(elbo_) % skip
    elbo_ = elbo_[offset_2:]
    elbo_avg = np.mean(np.reshape(elbo_, [-1, skip]), axis=-1)
    x = np.arange(offset + offset_2, len(elbo), skip)
    if epoch is not None:
        x = x // epoch
        plt.xlabel("Epochs")
    else:
        plt.xlabel("Iterations")

    print len(x), len(elbo_avg)
    plt.plot(x, elbo_avg, **kwargs)
    plt.ylabel("ELBO")
