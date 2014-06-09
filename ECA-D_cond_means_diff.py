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


log_file = open('ECA-D_cond_%s_%s' % ('means' if MEANS else 'std', datetime.now().strftime('%Y%m%d-%H%M')), 'w')
def log(msg):
    log_file.write('[%s] %s\n' % (str(datetime.now()), msg))
    log_file.flush()


## load and prepare data
g = load_ECA_D_data_daily('tg_0.25deg_reg_v10.0.nc', 'tg', date(1950,1,1), date(2014,1,1), None, None, ANOMALISE)
g.get_data_of_precise_length('16k', START_DATE, None, True) # get 2^n data because of MF surrogates
END_DATE = g.get_date_from_ndx(-1)
sg = SurrogateField()
mean, var, trend = g.get_seasonality(True) # subtract mean, divide by std and subtract trend from data
sg.copy_field(g) # copy standartised data to SurrogateField
g.return_seasonality(mean, var, trend) # return seasonality to data for analysis


## wavelet data
log("Running wavelet analysis on data using %d workers..." % (WORKERS))
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

if pool is not None:
    pool.close()

# select interor of wavelet to suppress unwanted edge effects
IDX = g.select_date(date(START_DATE.year + 4, START_DATE.month, START_DATE.day), 
                    date(END_DATE.year - 4, END_DATE.month, END_DATE.day))

phase_data = phase_data[IDX, ...]

log("Wavelet on data done. Computing conditional %s on data..." % ('means' if MEANS else 'variance'))

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
        
log("Analysis on data done. Starting surrogates...")


## save file in case something will go wrong with surrogates..
if not MEANS: # from variance to standard deviation
    difference_data = np.sqrt(difference_data)
    mean_data = np.sqrt(mean_data)
fname = ('ECA-D_cond_%s_%s' % ('means' if MEANS else 'std', str(START_DATE)))
with open(fname, 'w') as f:
    cPickle.dump({'difference_data' : difference_data, 'mean_data' : mean_data}, f)



def _analysis_surrogates(a):
    sf, seasonality, idx, jobq, resq = a
    mean, var, trend = seasonality
    phase_bins = get_equidistant_bins()
    surr_diffs = []
    surr_means = []
    while jobq.get() is not None:
        if SURR_TYPE[0]: # MF surrs
            mf_diffs = np.zeros((sf.data.shape[1], sf.data.shape[2]))
            mf_mean = np.zeros_like(mf_diffs)
            sf.construct_multifractal_surrogates()
            sf.add_seasonality(mean, var, trend)
            for lat in range(sf.lats.shape[0]):
                for lon in range(sf.lons.shape[0]):
                    wave, _, _, _ = wvlt.continous_wavelet(sf.surr_data[:, lat, lon], 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0)
                    phase = np.arctan2(np.imag(wave), np.real(wave))
                    # subselect surr_data and phase
                    sf.surr_data = sf.surr_data[idx, ...]
                    phase = phase[0, idx]
                    cond_means_temp = np.zeros((8,))
                    for i in range(cond_means_temp.shape[0]):
                        ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
                        if MEANS:
                            cond_means_temp[i] = np.mean(sf.surr_data[ndx])
                        else:
                            cond_means_temp[i] = np.var(sf.surr_data[ndx], ddof = 1)
                    mf_diffs[lat, lon] = cond_means_temp.max() - cond_means_temp.min()
                    mf_mean[lat, lon] = np.mean(cond_means_temp)
            surr_diffs.append(mf_diffs)
            surr_means.append(mf_mean)

        if SURR_TYPE[1]: # FT surrs
            ft_diffs = np.zeros((sf.data.shape[1], sf.data.shape[2]))
            ft_mean = np.zeros_like(ft_diffs)
            sf.construct_fourier_surrogates()
            sf.add_seasonality(mean, var, trend)
            for lat in range(sf.lats.shape[0]):
                for lon in range(sf.lons.shape[0]):
                    wave, _, _, _ = wvlt.continous_wavelet(sf.surr_data[:, lat, lon], 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0)
                    phase = np.arctan2(np.imag(wave), np.real(wave))
                    # subselect surr_data and phase
                    sf.surr_data = sf.surr_data[idx, ...]
                    phase = phase[0, idx]
                    cond_means_temp = np.zeros((8,))
                    for i in range(cond_means_temp.shape[0]):
                        ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
                        if MEANS:
                            cond_means_temp[i] = np.mean(sf.surr_data[ndx])
                        else:
                            cond_means_temp[i] = np.var(sf.surr_data[ndx], ddof = 1)
                    ft_diffs[lat, lon] = cond_means_temp.max() - cond_means_temp.min()
                    ft_mean[lat, lon] = np.mean(cond_means_temp)
            surr_diffs.append(ft_diffs)
            surr_means.append(ft_mean)

        if SURR_TYPE[2]: # AR(1) surrs
            ar_diffs = np.zeros((sf.data.shape[1], sf.data.shape[2]))
            ar_mean = np.zeros_like(ar_diffs)
            sf.data += trend
            sf.data *= var
            sf.data += mean
            sf.prepare_AR_surrogates()
            sf.construct_surrogates_with_residuals()
            for lat in range(sf.lats.shape[0]):
                for lon in range(sf.lons.shape[0]):
                    wave, _, _, _ = wvlt.continous_wavelet(sf.surr_data[:, lat, lon], 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0)
                    phase = np.arctan2(np.imag(wave), np.real(wave))
                    # subselect surr_data and phase
                    sf.surr_data = sf.surr_data[idx, ...]
                    phase = phase[0, idx]
                    cond_means_temp = np.zeros((8,))
                    for i in range(cond_means_temp.shape[0]):
                        ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
                        if MEANS:
                            cond_means_temp[i] = np.mean(sf.surr_data[ndx])
                        else:
                            cond_means_temp[i] = np.var(sf.surr_data[ndx], ddof = 1)
                    ar_diffs[lat, lon] = cond_means_temp.max() - cond_means_temp.min()
                    ar_mean[lat, lon] = np.mean(cond_means_temp)
                    
            surr_diffs.append(ar_diffs)
            surr_means.append(ar_mean)
                    
        resq.put((surr_diffs, surr_means))
        

    


## wavelet surrogates
log("Computing %d MF/FT/AR(1) surrogates in parallel using %d workers..." % (NUM_SURR, WORKERS))
surr_MIX_diff = np.zeros([np.sum(SURR_TYPE), NUM_SURR] + g.get_spatial_dims())
surr_MIX_mean = np.zeros([np.sum(SURR_TYPE), NUM_SURR] + g.get_spatial_dims())
surr_completed = 0
jobQ = Queue()
resQ = Queue()
for i in range(NUM_SURR):
    jobQ.put(1)
for i in range(WORKERS):
    jobQ.put(None)
    
seasonality = (mean, var, trend)
workers = [Process(target = _analysis_surrogates, args = (sg, seasonality, IDX, jobQ, resQ)) for iota in range(WORKERS)]

t_start = datetime.now()

log("Starting workers...")
for w in workers:
    w.start()
    
t_last = t_start

while surr_completed < NUM_SURR:
    # get result
    surr_diff, surr_mean = resQ.get()
    for surr_type in range(len(surr_diff)):
        surr_MIX_diff[surr_type, surr_completed, ...] = surr_diff[surr_type]
        surr_MIX_mean[surr_type, surr_completed, ...] = surr_mean[surr_type]
    surr_completed += 1
    
    # time to go
    t_now = datetime.now()
    
    if (t_now - t_last).total_seconds() > 600:
        t_last = t_now
        dt = (t_now - t_start) / surr_completed * (NUM_SURR - surr_completed)
        log("PROGRESS: %d/%d complete, predicted completition at %s..."
               % (surr_completed, NUM_SURR, str(t_now+dt)))
               
for w in workers:
    w.join()
    
log("Analysis on surrogates done after %s. Now saving data..." % (str(datetime.now - t_start)))

## save file with surrogates
if not MEANS: # from variance to standard deviation
    difference_data = np.sqrt(difference_data)
    mean_data = np.sqrt(mean_data)
    surr_MIX_diff = np.sqrt(surr_MIX_diff)
    surr_MIX_mean = np.sqrt(surr_MIX_mean)
fname = ('ECA-D_cond_%s_%s' % ('means' if MEANS else 'std', str(START_DATE)))
with open(fname, 'w') as f:
    cPickle.dump({'difference_data' : difference_data, 'mean_data' : mean_data, 
                  'difference_surrogates' : surr_MIX_diff, 'mean surrogates' : surr_MIX_mean,
                  'surrogates_type' : SURR_TYPE}, f)











