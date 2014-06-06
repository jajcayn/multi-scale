"""
created on June 6, 2014

@author: Nikola Jajcay
"""

from src import wavelet_analysis as wvlt
from src.data_class import load_ECA_D_data_daily
from surrogates.surrogates import SurrogateField
import numpy as np
from datetime import datetime, date
import cPickle
from multiprocessing import Process, Queue, Pool


def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))


ANOMALISE = True # if True, data will be anomalised hence SAT -> SATA
WORKERS = 3 # number of threads, if 0, all computations will be run single-thread
PERIOD = 8 # years; central period of wavelet used
START_DATE = date(1950,1,1)
MEANS = True # if True conditional means will be evaluated, if False conditional variance
NUM_SURR = 1000 # number of surrogates to be evaluated
SURR_TYPE = [1, 1, 1] # which types of surrogates to be evaluated as [MF, FT, AR(1)]


## load data and prepare data
g = load_ECA_D_data_daily('tg_0.25deg_reg_v10.0.nc', 'tg', date(1950,1,1), date(2014,1,1), None, None, ANOMALISE)
g.get_data_of_precise_length('16k', START_DATE, None, True) # get 2^n data because of MF surrogates
END_DATE = g.get_date_from_ndx(-1)
sg = SurrogateField()
mean, var, trend = g.get_seasonality(True) # subtract mean, divide by std and subtract trend from data
sg.copy_field(g) # copy standartised data to SurrogateField
g.return_seasonality(mean, var, trend) # return seasonality to data for analysis


## wavelet data
print("[%s] Running wavelet analysis on data using %d workers..." % (str(datetime.now()), WORKERS))
k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale

phase_data = np.zeros_like(g.data)
cond_means_data = np.zeros([8] + g.get_spatial_dims())


def _get_oscillatory_modes(a):
    """
    Gets oscillatory modes in terms of phase and amplitude from wavelet analysis for given data.
    """
    i, j, s0, data = a
    wave, _, _, _ = wvlt.continous_wavelet(data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = 6.)
    phase = np.arctan2(np.imag(wave), np.real(wave))
    
    return i, j, phase


if WORKERS == 0:
    pool = None
    map_func = map
else:
    pool = Pool(WORKERS)
    map_func = pool.map
    
job_args = [ (i, j, s0, g.data[:, i, j]) for i in range(g.lats.shape[0]) for j in range(g.lons.shape[0]) ]
job_result = map_func(_get_oscillatory_modes, job_args)
del job_args
# map results
for i, j, ph in job_result:
    phase_data[:, i, j] = ph
del job_result

# select interor of wavelet to suppress unwanted edge effects
IDX = g.select_date(date(START_DATE.year + 4, START_DATE.month, START_DATE.day), 
                    date(END_DATE.year - 4, END_DATE.month, END_DATE.day))

phase_data = phase_data[IDX, ...]

print("[%s] Wavelet on data done. Computing conditional %s on data..." % (str(datetime.now()), 
      'means' if MEANS else 'variance'))

## conditional means/variance data
difference_data = np.zeros(g.get_spatial_dims())
mean_data = np.zeros(g.get_spatial_dims())
phase_bins = get_equidistant_bins()

for lat in range(g.lats.shape[0]):
    for lon in range(g.lons.shape[0]):
        for i in range(cond_means_data.shape[0]):
            ndx = ((phase_data[:, lat, lon] >= phase_bins[i]) & (phase_data[:, lat, lon] <= phase_bins[i+1]))
            if MEANS:
                cond_means_data[i, lat, lon] = np.mean(g.data[ndx, lat, lon])
            else:
                cond_means_data[i, lat, lon] = np.var(g.data[ndx, lat, lon], ddof = 1)
        
        difference_data[lat, lon] = cond_means_data[:, lat, lon].max() - cond_means_data[:, lat, lon].min()
        mean_data[lat, lon] = np.mean(cond_means_data[:, lat, lon])
        
print("[%s] Analysis on data done. Starting surrogates..." % (str(datetime.now())))

## save file in case something will go wrong with surrogates..
fname = ('ECA-D_cond_%s_%s' % ('means' if MEANS else 'std', str(START_DATE)))
with open(fname, 'w') as f:
    cPickle.dump({'difference_data' : difference_data, 'mean_data' : mean_data}, f)



def _analysis_surrogates(a):
    sf, seasonality, idx, jobq, req = a
    mean, var, trend = seasonality
    while jobq.get() is not None:
        if SURR_TYPE[0]: # MF surrs
            sf.construct_multifractal_surrogates()
            sf.add_seasonality(mean, var, trend)
            wave, _, _, _ = wvlt.continous_wavelet(sf.surr_data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0)
            phase = np.arctan2(np.imag(wave), np.real(wave))
            # subselect surr_data and phase
            sf.surr_data = sf.surr_data[idx, ...]
    


## wavelet surrogates
surr_completed = 0
jobQ = Queue()
resQ = Queue()
for i in range(NUM_SURR):
    jobQ.put(1)
for i in range(WORKERS):
    jobQ.put(None)
    
seasonality = (mean, var, trend)
workers = [Process(target = _analysis_surrogates, args = (sg, seasonality, IDX, jobQ, resQ)) for iota in range(WORKERS)]
for w in workers:
    w.start()











