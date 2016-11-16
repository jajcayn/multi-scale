import numpy as np
from src.data_class import DataField
import src.wavelet_analysis as wvlt
from datetime import datetime
from src.mutual_information import mutual_information, cond_mutual_information

import multiprocessing as mp
from time import sleep


def _get_oscillatory_modes(a):
    """
    Gets oscillatory modes in terms of phase and amplitude from wavelet analysis for given data.
    """
    i, j, s0, flag, flag2, data = a
    wave, _, _, _ = wvlt.continous_wavelet(data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = 6.)
    phase = np.arctan2(np.imag(wave), np.real(wave))
    if flag:
        amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))

    ret = [phase]
    if flag:
        ret.append(amplitude)
    if flag2:
        ret.append(wave)
    
    return i, j, ret


def _filtered_data(a):

    i, j, ph, amp = a

    return i, j, amp * np.cos(ph)


def _get_phase_coherence(a):
    """
    Gets mean phase coherence for given data.
    """

    i, j, ph1, ph2 = a

    # get phase diff
    diff = ph1 - ph2

    # compute mean phase coherence
    coh = np.power(np.mean(np.cos(diff)), 2) + np.power(np.mean(np.sin(diff)), 2)

    return i, j, coh


def _get_continuous_phase(a):
    """
    Tranforms phases to continuous, strictly increasing.
    """

    i, j, ph1 = a
    # get continuous phase
    for ii in range(ph1.shape[0] - 1):
        if np.abs(ph1[ii+1] - ph1[ii]) > 1:
            ph1[ii+1: ] += 2 * np.pi

    return i, j, ph1



def _get_phase_fluctuations(a):
    """
    Gets phase fluctuations.
    """

    i, j, omega, ph = a

    ph0 = ph[0]
    for t in range(ph.shape[0]):
        ph[t] -= ph0 + omega*t 

    return i, j, ph



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
    
    i, j, ph1, ph2 = a
    return i, j, mutual_information(ph1, ph2, algorithm = 'EQQ2', bins = 4, log2 = False)


def _get_automutual_info(a):
    """
    Gets automutual information function.
    """

    i, j, to, ph = a
    result = []
    for tau in range(1,to):
        result.append(mutual_information(ph[tau:], ph[:-tau], algorithm = 'EQQ2', bins = 4, log2 = False))

    return i, j, np.array(result)




class ScaleSpecificNetwork(DataField):
    """
    Class holds geo data (inherits methods from DataField) and can construct networks.
    """

    def __init__(self, fname, varname, start_date, end_date, lats, lons, level = None, sampling = 'monthly', anom = False, pickled = False):
        """
        Initialisation of the class.
        """

        # if sampling == 'monthly':
        #     self.g = load_NCEP_data_monthly(fname, varname, start_date, end_date, None, None, None, anom)
        # elif sampling == 'daily':
        #     self.g = load_NCEP_data_daily(fname, varname, start_date, end_date, None, None, None, anom)

        DataField.__init__(self)
        if not pickled:
            self.load(fname, varname, dataset = "NCEP", print_prog = False)
        else:
            self.load_field(fname)
        self.select_date(start_date, end_date)
        self.select_lat_lon(lats, lons)
        if level is not None:
            self.select_level(level)
        if anom:
            self.anomalise()
        day, month, year = self.extract_day_month_year()
        print("[%s] NCEP data loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
                % (str(datetime.now()), str(self.data.shape), day[0], month[0], 
                   year[0], day[-1], month[-1], year[-1]))


        self.phase = None
        self.amplitude = None
        self.wave = None
        self.adjacency_matrix = None

        self.num_lats = self.lats.shape[0]
        self.num_lons = self.lons.shape[0]
        self.sampling = sampling



    def wavelet(self, period, get_amplitude = False, save_wavelet = False, pool = None):
        """
        Performs wavelet analysis on the data.
        """

        self.period = period

        k0 = 6. # wavenumber of Morlet wavelet used in analysis, suppose Morlet mother wavelet
        if self.sampling == 'monthly':
            y = 12
        elif self.sampling == 'daily':
            y = 365.25
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
        per = period * y # frequency of interest
        s0 = per / fourier_factor # get scale

        self.phase = np.zeros_like(self.data)
        if get_amplitude:
            self.amplitude = np.zeros_like(self.data)
        if save_wavelet:
            self.wave = np.zeros_like(self.data, dtype = np.complex64)

        if pool is None:
            map_func = map
        elif pool is not None:
            map_func = pool.map

        job_args = [ (i, j, s0, get_amplitude, save_wavelet, self.data[:, i, j]) for i in range(self.num_lats) for j in range(self.num_lons) ]
        job_result = map_func(_get_oscillatory_modes, job_args)
        del job_args

        for i, j, res in job_result:
            self.phase[:, i, j] = res[0]
            if get_amplitude:
                self.amplitude[:, i, j] = res[1]
            if save_wavelet and get_amplitude:
                self.wave[:, i, j] = res[2]
            if save_wavelet and not get_amplitude:
                self.wave[:, i, j] = res[1]

        del job_result



    def get_filtered_data(self, pool = None):
        """
        Returns filtered data as x = A * cos phi, where A and phi are derived from wavelet.
        """

        if pool is None:
            map_func = map
        elif pool is not None:
            map_func = pool.map

        self.filtered_data = np.zeros_like(self.data)
        job_args = [ (i, j, self.phase[:, i, j], self.amplitude[:, i, j]) for i in range(self.num_lats) for j in range(self.num_lons) ]
        job_result = map_func(_filtered_data, job_args)
        del job_args

        for i, j, res in job_result:
            self.filtered_data[:, i, j] = res

        del job_result



    def get_continuous_phase(self, pool = None):
        """
        Transforms phase from wavelet to continuous, increasing.
        """

        if pool is None:
            map_func = map
        elif pool is not None:
            map_func = pool.map

        job_args = [ (i, j, self.phase[:, i, j]) for i in range(self.num_lats) for j in range(self.num_lons) ]
        job_result = map_func(_get_continuous_phase, job_args)
        del job_args

        for i, j, res in job_result:
            self.phase[:, i, j] = res

        del job_result


    def get_automutualinf(self, endpoint, pool = None):
        """
        Gets auto-mutual information function (nonlinear autocorrelation).
        endpoint in months
        """

        if self.phase is None:
            raise Exception("Automutual information is computed from phases, do a wavelet first!")

        if pool is None:
            map_func = map
        elif pool is not None:
            map_func = pool.map

        job_args = [ (i, j, endpoint, self.phase[:, i, j]) for i in range(self.num_lats) for j in range(self.num_lons) ]
        job_result = map_func(_get_automutual_info, job_args)
        del job_args

        self.automutual_info = np.zeros((endpoint - 1, self.num_lats, self.num_lons))

        for i, j, res in job_result:
            self.automutual_info[:, i, j] = res

        del job_result


    def get_phase_fluctuations(self, rewrite = True, pool = None):
        """
        Gets phase fluctuations.
        """

        self.get_continuous_phase(pool = pool)

        if self.sampling == 'monthly':
           omega = 2 * np.pi / 12
        elif self.sampling == 'daily':
            omega = 2 * np.pi / 365.25

        if pool is None:
            map_func = map
        elif pool is not None:
            map_func = pool.map

        job_args = [ (i, j, omega, self.phase[:, i, j]) for i in range(self.num_lats) for j in range(self.num_lons) ]
        job_result = map_func(_get_phase_fluctuations, job_args)
        del job_args

        self.phase_fluctuations = np.zeros_like(self.data)
        
        for i, j, res in job_result:
            self.phase_fluctuations[:, i, j] = res

        if rewrite:
            self.phase = self.phase_fluctuations.copy()

        del job_result



    def _process_matrix(self, jobq, resq):
        
        while True:
            a = jobq.get() # get queued input

            if a is None: # if it is None, we are finished, put poison pill to resq
                break # break infinity cycle
            else:
                i, j, ph1, ph2, method = a # compute stuff
                if method == "MPC":
                    # get phase diff
                    diff = ph1 - ph2
                    # compute mean phase coherence
                    coh = np.power(np.mean(np.cos(diff)), 2) + np.power(np.mean(np.sin(diff)), 2)
                    resq.put((i, j, coh))

                elif method == "MIEQQ":
                    resq.put((i, j, mutual_information(ph1, ph2, algorithm = 'EQQ2', bins = 4, log2 = False)))

                elif method == "MIGAU":
                    corr = np.corrcoef([ph1, ph2])[0, 1]                    
                    mi = -0.5 * np.log(1 - np.power(corr, 2)) if corr < 1. else 0
                    resq.put((i, j, mi)) 

                elif method == "COV":
                    resq.put((i, j, np.cov(ph1, ph2, ddof = 1)[0,1]))

                elif method == "WCOH":
                    # input field must be wave from wavelet!!!!
                    w1 = np.complex(0, 0)
                    w2 = w1; w3 = w1
                    for t in range(0, self.time.shape[0]):
                        w1 += ph1[t] * np.conjugate(ph2[t])
                        w2 += ph1[t] * np.conjugate(ph1[t])
                        w3 += ph2[t] * np.conjugate(ph2[t])
                    w1 /= np.sqrt(np.abs(w2) * np.abs(w3))
                    resq.put((i, j, np.abs(w1)))

                elif method[0] == 'L':
                    p = int(method[1])
                    res = 0
                    for t in range(ph1.shape[0]):
                        res += np.power(np.abs(ph1[t] - ph2[t]), p)
                    resq.put((i, j, res))


    def _process_matrix_cond(self, jobq, resq):
        
        while True:
            a = jobq.get() # get queued input

            if a is None: # if it is None, we are finished, put poison pill to resq
                break # break infinity cycle
            else:
                i, j, ph1, ph2, cond_ts = a # compute stuff
                resq.put((i, j, cond_mutual_information(ph1, ph2, cond_ts, algorithm = 'EQQ2', bins = 4, log2 = False)))



    def get_adjacency_matrix(self, field, method = "MPC", pool = None, use_queue = True, num_workers = 0):
        """
        Gets the matrix of mean phase coherence between each two grid-points.
        Methods for adjacency matrix:
            MPC - mean phase coherence
            MIEQQ - mutual information - equiquantal algorithm
            MIGAU - mutual information - Gauss algorithm
            COV - covariance matrix
            WCOH - wavelet coherence
            L1 or L2 - Lp difference
        """
        
        if method == "MPC":
            self.get_continuous_phase(pool = pool)
        
        field = self.flatten_field(field)

        start = datetime.now()

        if not use_queue:
            self.adjacency_matrix = np.zeros((field.shape[1], field.shape[1]))

            if pool is None:
                map_func = map
            elif pool is not None:
                map_func = pool.map
            job_args = [ (i, j, field[:, i], field[:, j]) for i in range(field.shape[1]) for j in range(i, field.shape[1]) ]
            if method == 'MPC':
                job_results = map_func(_get_phase_coherence, job_args)
            elif method == 'MIGAU':
                job_results = map_func(_get_mutual_inf_gauss, job_args)
            elif method == 'MIEQQ':
                job_results = map_func(_get_mutual_inf_EQQ, job_args)
            del job_args

            for i, j, coh in job_results:
                self.adjacency_matrix[i, j] = coh
                self.adjacency_matrix[j, i] = coh

            for i in range(self.adjacency_matrix.shape[0]):
                self.adjacency_matrix[i, i] = 0.

            del job_results

        else:

            if method == "WCOH" and field.dtype != np.complex64:
                raise Exception("Wavelet coherence requires input field to be wave data from wavelet!")
            if method[0] == 'L' and int(method[1]) not in [1,2]:
                raise Exception("Lp method shoud use p = 1 or 2")

            jobs = mp.Queue()
            results = mp.Queue()

            # start workers - BEFORE filling the queue as they are simultaneously computing while filling the queue
            workers = [mp.Process(target = self._process_matrix, args = (jobs, results)) for i in range(num_workers)]
            for w in workers:
                w.start()

            # fill queue with actual inputs
            cnt_results = 0
            for i in range(field.shape[1]):
                for j in range(i, field.shape[1]):
                    cnt_results += 1
                    jobs.put([i, j, field[:, i], field[:, j], method])
            
            # fill queue with None for workers to finish
            for i in range(num_workers):
                jobs.put(None)

            self.adjacency_matrix = np.zeros((field.shape[1], field.shape[1]))

            cnt = 0
            while cnt < cnt_results:
                i, j, val = results.get()
                self.adjacency_matrix[i, j] = val
                self.adjacency_matrix[j, i] = val
                cnt += 1

            # finally, finish workers
            for w in workers:
                w.join()

            # nullify the diagonal 
            for i in range(self.adjacency_matrix.shape[0]):
                self.adjacency_matrix[i, i] = 0.

        print datetime.now()-start
        
        field = self.reshape_flat_field(field)


    def get_adjacency_matrix_conditioned(self, cond_ts, use_queue = True, num_workers = 0):


        self.phase = self.flatten_field(self.phase)

        start = datetime.now()

        jobs = mp.Queue()
        results = mp.Queue()

        # start workers - BEFORE filling the queue as they are simultaneously computing while filling the queue
        workers = [mp.Process(target = self._process_matrix_cond, args = (jobs, results)) for i in range(num_workers)]
        for w in workers:
            w.start()

        # fill queue with actual inputs
        cnt_results = 0
        for i in range(self.phase.shape[1]):
            for j in range(i, self.phase.shape[1]):
                cnt_results += 1
                jobs.put([i, j, self.phase[:, i], self.phase[:, j], cond_ts])
        
        # fill queue with None for workers to finish
        for i in range(num_workers):
            jobs.put(None)

        self.adjacency_matrix = np.zeros((self.phase.shape[1], self.phase.shape[1]))

        cnt = 0
        while cnt < cnt_results:
            i, j, val = results.get()
            self.adjacency_matrix[i, j] = val
            cnt += 1

        # finally, finish workers
        for w in workers:
            w.join()

        # nullify the diagonal 
        for i in range(self.adjacency_matrix.shape[0]):
            self.adjacency_matrix[i, i] = 0.

        print datetime.now()-start
        
        self.phase = self.reshape_flat_field(self.phase)




    def save_net(self, fname, only_matrix = True):
        """
        Saves the scale specific network.
        If only_matrix is True, saves only adjacency_matrix, else saves the whole class.
        """

        import cPickle

        with open(fname, 'wb') as f:
            if only_matrix:
                cPickle.dump({'adjacency_matrix' : self.adjacency_matrix.astype(np.float16)}, f, protocol = cPickle.HIGHEST_PROTOCOL)
            else:
                cPickle.dump(self.__dict__, f, protocol = cPickle.HIGHEST_PROTOCOL)


    def load_net(self, fname):
        """
        Loads the network into the class.
        """

        import cPickle

        with open(fname, 'rb') as f:
            data = cPickle.load(f)

        self.__dict__ = data

