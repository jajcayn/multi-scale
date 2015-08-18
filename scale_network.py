import numpy as np
from src.data_class import load_NCEP_data_monthly, load_NCEP_data_daily
import src.wavelet_analysis as wvlt
from datetime import datetime


def _get_oscillatory_modes(a):
    """
    Gets oscillatory modes in terms of phase and amplitude from wavelet analysis for given data.
    """
    i, j, s0, data = a
    wave, _, _, _ = wvlt.continous_wavelet(data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = 6.)
    phase = np.arctan2(np.imag(wave), np.real(wave))
    amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))
    
    return i, j, phase, amplitude


def _get_phase_coherence(a):
    """
    Gets mean phase coherence for given data.
    """

    i, j, ph1, ph2 = a
    # get continuous phase
    for ii in range(ph1.shape[0] - 1):
        if np.abs(ph1[ii+1] - ph1[ii]) > 1:
            ph1[ii+1: ] += 2 * np.pi
        if np.abs(ph2[ii+1] - ph2[ii]) > 1:
            ph2[ii+1: ] += 2 * np.pi

    # get phase diff
    diff = ph1 - ph2

    # compute mean phase coherence
    coh = np.power(np.mean(np.cos(diff)), 2) + np.power(np.mean(np.sin(diff)), 2)
    print("computing for %d -- %d grid point" % (i, j))

    return i, j, coh


def _get_mutual_inf_gauss(a):
    """
    Gets mutual information using Gauss algorithm for given data.
    """

    i, j, ph1, ph2 = a
    corr = np.corrcoef([ph1, ph2])[0, 1]
    mi = -0.5 * np.log(1 - np.power(corr, 2))
    
    return i, j, mi


def _get_mutual_inf_EQQ(a):
    """
    Gets mutual information using EQQ algorithm for given data.
    """

    import sys
    sys.path.append('/home/nikola/Work/phd/mutual_information')
    from mutual_information import mutual_information
    
    i, j, ph1, ph2 = a
    return i, j, mutual_information(ph1, ph2, algorithm = 'EQQ2', bins = 8, log2 = False)




class ScaleSpecificNetwork():
    """
    Class holds geo data and can construct networks.
    """

    def __init__(self, fname, varname, start_date, end_date, sampling = 'monthly', anom = False):
        """
        Initialisation of the class.
        """

        if sampling == 'monthly':
            self.g = load_NCEP_data_monthly(fname, varname, start_date, end_date, None, None, None, anom)
        elif sampling == 'daily':
            self.g = load_NCEP_data_daily(fname, varname, start_date, end_date, None, None, None, anom)

        self.phase = None
        self.amplitude = None
        self.adjacency_matrix = None

        self.num_lats = self.g.lats.shape[0]
        self.num_lons = self.g.lons.shape[0]
        self.sampling = sampling



    def wavelet(self, period, pool = None):
        """
        Performs wavelet analysis on the data.
        """

        k0 = 6. # wavenumber of Morlet wavelet used in analysis, suppose Morlet mother wavelet
        if self.sampling == 'monthly':
            y = 12
        elif self.sampling == 'daily':
            y = 365.25
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
        per = period * y # frequency of interest
        s0 = per / fourier_factor # get scale

        self.phase = np.zeros_like(self.g.data)
        self.amplitude = np.zeros_like(self.g.data)

        if pool is None:
            map_func = map
        elif pool is not None:
            map_func = pool.map

        job_args = [ (i, j, s0, self.g.data[:, i, j]) for i in range(self.num_lats) for j in range(self.num_lons) ]
        job_result = map_func(_get_oscillatory_modes, job_args)
        del job_args

        for i, j, ph, am in job_result:
            self.phase[:, i, j] = ph
            self.amplitude[:, i, j] = am

        del job_result



    def get_adjacency_matrix(self, method = "MPC", pool = None):
        """
        Gets the matrix of mean phase coherence between each two grid-points.
        Methods for adjacency matrix:
            MPC - mean phase coherence
            MIEQQ - mutual information - equiquantal algorithm
            MIGAU - mutual information - Gauss algorithm
        """
        
        self.phase = self.g.flatten_field(self.phase)

        self.adjacency_matrix = np.zeros((self.phase.shape[1], self.phase.shape[1]))

        if pool is None:
            map_func = map
        elif pool is not None:
            map_func = pool.map
        job_args = [ (i, j, self.phase[:, i], self.phase[:, j]) for i in range(self.phase.shape[1]) for j in range(i, self.phase.shape[1]) ]
        start = datetime.now()
        if method == 'MPC':
            job_results = map_func(_get_phase_coherence, job_args)
        elif method == 'MIGAU':
            job_results = map_func(_get_mutual_inf_gauss, job_args)
        elif method == 'MIEQQ':
            job_results = map_func(_get_mutual_inf_EQQ, job_args)
        end = datetime.now()
        print end-start
        del job_args

        for i, j, coh in job_results:
            self.adjacency_matrix[i, j] = coh
            self.adjacency_matrix[j, i] = coh

        del job_results
        
        self.phase = self.g.reshape_flat_field(self.phase)

