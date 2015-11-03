"""
created on August 5, 2015

@author: Nikola Jajcay, jajcay(at)cs.cas.cz
"""

import numpy as np


def autocorr_delay_crit():
    pass


def mutual_inf_delay_crit():
    pass


def false_neighbors_dimension_crit():
    pass



def time_delay_embed(ts, dim, tau, past = True):
    """
    Reconstructs the phase space of the attractor by delaying time series according to
    Takens' [1] embedding theorem.
    Input time series ts shuld be 1D. 
    dim and tau as dimension of embedding and delay are integers.
    Returns delay vector - list of dim numpy arrays with appropriate embedded time series as
        [x(t), x(t - tau), x(t - 2tau), x(t - (d-1)tau)]
    if past is False, the future embed is used
        [x(t), x(t + tau), x(t + 2tau), x(t + (d-1)tau)]
    If output should be numpy array with shape (dim x length) use np.array(time_delay_embed),
        where length is len(ts) - (dim - 1) * tau
    
        [1] Takens, F. (1981) Springer, vol. 898 of Lecture Notes in Mathematics.
    """

    if np.squeeze(ts).ndim > 1:
        raise Exception("Time delaying should be used with 1D input time series")

    n = dim - 1

    # "now" time series
    delay_vector = [ts[n * tau:]] if past else [ts[:-n * tau]]

    for i in range(1, dim):
        if past:
            delay_vector.append(np.roll(ts, i * tau)[n * tau:])
            if delay_vector[-1].shape != delay_vector[-2].shape:
                raise Exception("Something went wrong, check the code and/or input time series!")
        else:
            delay_vector.append(np.roll(ts, -i * tau)[:-n * tau])
            if delay_vector[-1].shape != delay_vector[-2].shape:
                raise Exception("Something went wrong, check the code and/or input time series!")

    return delay_vector




