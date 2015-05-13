"""
created on May 6, 2015

@author: Nikola Jajcay, jajcay(at)cs.cas.cz
"""

import numpy as np


def mutual_information(x, y, algorithm = 'EQQ', bins = 8, log2 = True):
    """
    Computes mutual information between two time series x and y as
        I(x; y) = sum( p(x,y) * log( p(x,y) / p(x)p(y) ),
        where p(x), p(y) and p(x, y) are probability distributions.
    The probability distributions could be estimated using these algorithms:
        equiquantal binning - algorithm keyword 'EQQ'
        equidistant binning - algorithm keyword 'EQD'
        (preparing more...)
    If log2 is True (default), the units of mutual information are bits, if False
      the mutual information is be estimated using natural logarithm and therefore
      units are nats.
    """

    log_f = np.log2 if log2 else np.log

    if algorithm == 'EQD':
        x_bins = bins
        y_bins = bins
        xy_bins = bins

    elif algorithm == 'EQQ':
        # create EQQ bins
        x_sorted = np.sort(x)
        x_bins = [x.min()]
        [x_bins.append(x_sorted[i*x.shape[0]/bins]) for i in range(1, bins)]
        x_bins.append(x.max())

        y_sorted = np.sort(y)
        y_bins = [y.min()]
        [y_bins.append(y_sorted[i*y.shape[0]/bins]) for i in range(1, bins)]
        y_bins.append(y.max())
        
        xy_bins = [x_bins, y_bins]

    # histo
    count_x = np.histogramdd([x], bins = [x_bins])[0]
    count_y = np.histogramdd([y], bins = [y_bins])[0]
    count_xy = np.histogramdd([x, y], bins = xy_bins)[0]

    # normalise
    count_xy /= np.float(np.sum(count_xy))
    count_x /= np.float(np.sum(count_x))
    count_y /= np.float(np.sum(count_y))

    # sum
    mi = 0
    for i in range(bins):
        for j in range(bins):
            if count_x[i] != 0 and count_y[j] != 0 and count_xy[i, j] != 0:
                mi += count_xy[i, j] * log_f(count_xy[i, j] / (count_x[i] * count_y[j]))

    return mi


def cond_mutual_information(x, y, z, algorithm = 'EQQ', bins = 8, log2 = True):
    """
    Computes conditional mutual information between two time series x and y 
    conditioned on a third z (which can be multi-dimensional) as
        I(x; y | z) = sum( p(x,y,z) * log( p(z)*p(x,y,z) / p(x,z)*p(y,z) ),
        where p(z), p(x,z), p(y,z) and p(x,y,z) are probability distributions.
    The probability distributions could be estimated using these algorithms:
        equiquantal binning - algorithm keyword 'EQQ' or 'EQQ2'
            EQQ - equiquantality is forced (even if many samples have the same value 
                at and near the bin edge), can happen that samples with same value fall
                into different bin
            EQQ2 - if more than one sample has the same value at the bin edge, the edge is shifted,
                so that all samples with the same value fall into the same bin, can happen that bins
                do not necessarily contain the same amount of samples
        equidistant binning - algorithm keyword 'EQD'
        (preparing more...)
    If log2 is True (default), the units of cond. mutual information are bits, if False
      the mutual information is estimated using natural logarithm and therefore
      units are nats.
    """

    log_f = np.log2 if log2 else np.log

    # for multi-dimensional condition -- create array from list as (dim x length of ts)
    z = np.atleast_2d(z)

    if algorithm == 'EQQ':
        # create EQQ bins -- condition [possibly multidimensional]
        z_bins = [] # arrays of bins for all conditions
        for cond in range(z.shape[0]):
            z_sorted = np.sort(z[cond, :])
            z_bin = [z_sorted.min()]
            [z_bin.append(z_sorted[i*z_sorted.shape[0]/bins]) for i in range(1, bins)]
            z_bin.append(z_sorted.max())
            z_bins.append(np.array(z_bin))

        # create EQQ bins -- variables
        x_sorted = np.sort(x)
        x_bin = [x.min()]
        [x_bin.append(x_sorted[i*x.shape[0]/bins]) for i in range(1, bins)]
        x_bin.append(x.max())
        x_bin = np.array(x_bin)

        y_sorted = np.sort(y)
        y_bin = [y.min()]
        [y_bin.append(y_sorted[i*y.shape[0]/bins]) for i in range(1, bins)]
        y_bin.append(y.max())
        y_bin = np.array(y_bin)

        # create multidim bins for histogram
        xyz_bins = [x_bin, y_bin]
        [xyz_bins.append(cond_bins) for cond_bins in z_bins]

        xz_bins = [x_bin]
        [xz_bins.append(cond_bins) for cond_bins in z_bins]

        yz_bins = [y_bin]
        [yz_bins.append(cond_bins) for cond_bins in z_bins]


    elif algorithm == 'EQQ2':
        # create EQQ bins -- condition [possibly multidimensional]
        z_bins = [] # arrays of bins for all conditions
        for cond in range(z.shape[0]):
            z_sorted = np.sort(z[cond, :])
            z_bin = [z_sorted.min()]
            one_bin_count = z_sorted.shape[0] / bins
            for i in range(1, bins):
                idx = i * one_bin_count
                if np.all(np.diff(z_sorted[idx-1:idx+2]) != 0):
                    z_bin.append(z_sorted[idx])
                elif np.any(np.diff(z_sorted[idx-1:idx+2]) == 0):
                    where = np.where(np.diff(z_sorted[idx-1:idx+2]) != 0)[0]
                    expand_idx = 1
                    while where.size == 0:
                        where = np.where(np.diff(z_sorted[idx-expand_idx:idx+1+expand_idx]) != 0)[0]
                        expand_idx += 1
                    if where[0] == 0:
                        z_bin.append(z_sorted[idx-expand_idx])
                    else:
                        z_bin.append(z_sorted[idx+expand_idx])
            z_bin.append(z_sorted.max())
            z_bin = np.array(z_bin)
            z_bins.append(np.array(z_bin))

        # create EQQ bins -- variables
        x_sorted = np.sort(x)
        x_bin = [x.min()]
        one_bin_count = x.shape[0] / bins
        for i in range(1, bins):
            idx = i * one_bin_count
            if np.all(np.diff(x_sorted[idx-1:idx+2]) != 0):
                x_bin.append(x_sorted[idx])
            elif np.any(np.diff(x_sorted[idx-1:idx+2]) == 0):
                where = np.where(np.diff(x_sorted[idx-1:idx+2]) != 0)[0]
                expand_idx = 1
                while where.size == 0:
                    where = np.where(np.diff(x_sorted[idx-expand_idx:idx+1+expand_idx]) != 0)[0]
                    expand_idx += 1
                if where[0] == 0:
                    x_bin.append(x_sorted[idx-expand_idx])
                else:
                    x_bin.append(x_sorted[idx+expand_idx])
        x_bin.append(x.max())
        x_bin = np.array(x_bin)

        y_sorted = np.sort(y)
        y_bin = [y.min()]
        one_bin_count = y.shape[0] / bins
        for i in range(1, bins):
            idx = i * one_bin_count
            if np.all(np.diff(y_sorted[idx-1:idx+2]) != 0):
                y_bin.append(y_sorted[idx])
            elif np.any(np.diff(y_sorted[idx-1:idx+2]) == 0):
                where = np.where(np.diff(y_sorted[idx-1:idx+2]) != 0)[0]
                expand_idx = 1
                while where.size == 0:
                    where = np.where(np.diff(y_sorted[idx-expand_idx:idx+1+expand_idx]) != 0)[0]
                    expand_idx += 1
                if where[0] == 0:
                    y_bin.append(y_sorted[idx-expand_idx])
                else:
                    y_bin.append(y_sorted[idx+expand_idx])
        y_bin.append(y.max())
        y_bin = np.array(y_bin)

        # create multidim bins for histogram
        xyz_bins = [x_bin, y_bin]
        [xyz_bins.append(cond_bins) for cond_bins in z_bins]

        xz_bins = [x_bin]
        [xz_bins.append(cond_bins) for cond_bins in z_bins]

        yz_bins = [y_bin]
        [yz_bins.append(cond_bins) for cond_bins in z_bins]

    
    elif algorithm == 'EQD':
        # bins are just integer
        xyz_bins = bins
        xz_bins = bins
        yz_bins = bins
        z_bins = bins

    # histo
    count_z = np.histogramdd(z.T, bins = z_bins)[0]

    xyz = [x, y]
    [xyz.append(cond) for cond in z]
    count_xyz = np.histogramdd(xyz, bins = xyz_bins)[0]

    xz = [x]
    [xz.append(cond) for cond in z]
    count_xz = np.histogramdd(xz, bins = xz_bins)[0]

    yz = [y]
    [yz.append(cond) for cond in z]
    count_yz = np.histogramdd(yz, bins = yz_bins)[0]

    # normalise
    count_z /= np.float(np.sum(count_z))
    count_xyz /= np.float(np.sum(count_xyz))
    count_xz /= np.float(np.sum(count_xz))
    count_yz /= np.float(np.sum(count_yz))

    # sum
    cmi = 0
    iterator = np.nditer(count_xyz, flags = ['multi_index'])
    while not iterator.finished:
        idx = iterator.multi_index
        xz_idx = tuple([ item for sublist in [[idx[0]], idx[2:]] for item in sublist ]) # creates index for xz histo
        if count_xyz[idx] == 0 or count_z[idx[2:]] == 0 or count_xz[xz_idx] == 0 or count_yz[idx[1:]] == 0:
            iterator.iternext()
            continue
        else:
            cmi += count_xyz[idx] * log_f(count_z[idx[2:]] * count_xyz[idx] / (count_xz[xz_idx] * count_yz[idx[1:]]))

        iterator.iternext()

    return cmi
