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
import os


def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))
    


ANOMALISE = True # if True, data will be anomalised hence SAT -> SATA
WORKERS = 16 # number of threads, if 0, all computations will be run single-thread
PERIOD = 8 # years; central period of wavelet used
START_DATE = date(1960,1,1)
LATS = [25.375, 75.375] # lats ECA: 25.375 -- 75.375 = 201 grid points
LONS = [-40.375, -11.375] #lons ECA: -40.375 -- 75.375 = 464 grid points
MEANS = True # if True conditional means will be evaluated, if False conditional variance
NUM_SURR = 1000 # number of surrogates to be evaluated
SURR_TYPE = [0, 1, 0] # which types of surrogates to be evaluated as [MF, FT, AR(1)]
LOG = True # if True, output will be written to log defined in log_file, otherwise printed to screen
# warning: logging into log file will suppress printing warnings handled by modules e.g. numpy's warnings


if LOG:
    log_file = open('result/ECA-D_%s_cond_%s_%s.log' % ('SATA' if ANOMALISE else 'SAT', 
                'means' if MEANS else 'std', datetime.now().strftime('%Y%m%d-%H%M')), 'w')

def log(msg):
    if LOG:
        log_file.write('[%s] %s\n' % (str(datetime.now()), msg))
        log_file.flush()
    else:
        print("[%s] %s" % (str(datetime.now()), msg))

## load and prepare data
g = load_ECA_D_data_daily('tg_0.25deg_reg_v10.0.nc', 'tg', date(1950,1,1), date(2014,1,1), 
                            LATS, LONS, ANOMALISE, logger_function = log)
g.get_data_of_precise_length('16k', START_DATE, None, True) # get 2^n data because of MF surrogates
END_DATE = g.get_date_from_ndx(-1)
if np.sum(SURR_TYPE) != 0:
    log("Creating surrogate fields...")
    sg = SurrogateField() # for MF and FT surrs
    if SURR_TYPE[2]:
        sgAR = SurrogateField() # for AR(1) surrs
        sgAR.copy_field(g)
    log("De-seasonalising the data and copying to surrogate field...")
    mean, var, trend = g.get_seasonality(True) # subtract mean, divide by std and subtract trend from data
    sg.copy_field(g) # copy standartised data to SurrogateField
    g.return_seasonality(mean, var, trend) # return seasonality to data for analysis
    log("Surrogate fields created.")


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
    if np.all(np.isnan(data)) == False:
        wave, _, _, _ = wvlt.continous_wavelet(data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = 6.)
        phase = np.arctan2(np.imag(wave), np.real(wave))
    else:
        phase = np.nan
    
    return i, j, phase


def _get_cond_means(a):
    """
    Gets either conditional means or variance according to phase.
    """
    i, j, phase, data, phase_bins = a
    cond_means_temp = np.zeros((8,))
    if np.all(np.isnan(phase)) == False:
        for i in range(cond_means_data.shape[0]):
            ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
            if MEANS:
                cond_means_temp[i] = np.mean(data[ndx])
            else:
                cond_means_temp[i] = np.var(data[ndx], ddof = 1)
        diff_temp = cond_means_temp.max() - cond_means_temp.min()
        mean_temp = np.mean(cond_means_temp)
    else:
        diff_temp = np.nan
        mean_temp = np.nan

    return i, j, diff_temp, mean_temp


if WORKERS == 0:
    pool = None
    map_func = map
else:
    pool = Pool(WORKERS)
    map_func = pool.map

if SURR_TYPE[2]:
    # prepare AR(1) surrogates in parallel
    sgAR.prepare_AR_surrogates(pool)
    
job_args = [ (i, j, s0, g.data[:, i, j]) for i in range(g.lats.shape[0]) for j in range(g.lons.shape[0]) ]
job_result = map_func(_get_oscillatory_modes, job_args)
del job_args
# map results
for i, j, ph in job_result:
    phase_data[:, i, j] = ph
del job_result

# select interior of wavelet to suppress unwanted edge effects
IDX = g.select_date(date(START_DATE.year + 4, START_DATE.month, START_DATE.day), 
                    date(END_DATE.year - 4, END_DATE.month, END_DATE.day))

phase_data = phase_data[IDX, ...]

log("Wavelet on data done. Computing conditional %s on data..." % ('means' if MEANS else 'variance'))

## conditional means / variance data
difference_data = np.zeros(g.get_spatial_dims())
mean_data = np.zeros(g.get_spatial_dims())
phase_bins = get_equidistant_bins()

job_args = [ (i, j, phase_data[:, i, j], g.data[:, i, j], phase_bins) for i in range(g.lats.shape[0]) for j in range(g.lons.shape[0]) ]
job_result = map_func(_get_cond_means, job_args)
del job_args, phase_data
# map results
for i, j, diff_t, mean_t in job_result:
    difference_data[i, j] = diff_t
    mean_data[i, j] = mean_t
del job_result

if pool is not None:
    pool.close()
    pool.join()
    del pool
        
log("Analysis on data done. Saving file...")


## save file in case something will go wrong with surrogates..
if not MEANS: # from variance to standard deviation
    difference_data = np.sqrt(difference_data)
    mean_data = np.sqrt(mean_data)
fname = ('result/ECA-D_%s_cond_%s_data_%s_areaof_%s.bin' % ('SATA' if ANOMALISE else 'SAT', 
         'means' if MEANS else 'std', str(START_DATE), str(LONS[0])+'-'+str(LONS[1])))
with open(fname, 'w') as f:
    cPickle.dump({'difference_data' : difference_data, 'mean_data' : mean_data}, f)

del g

def _analysis_surrogates(sf, sfAR, seasonality, idx, jobq, resq):
    mean, var, trend = seasonality
    phase_bins = get_equidistant_bins()
    surr_diffs = np.zeros((3, sf.data.shape[1], sf.data.shape[2]))
    surr_means = np.zeros_like(surr_diffs)
    while jobq.get() is not None:
        if SURR_TYPE[0]: # MF surrs
            mf_diffs = np.zeros((sf.data.shape[1], sf.data.shape[2]))
            mf_mean = np.zeros_like(mf_diffs)
            sf.construct_multifractal_surrogates()
            sf.add_seasonality(mean, var, trend)
            for lat in range(sf.lats.shape[0]):
                for lon in range(sf.lons.shape[0]):
                    if np.all(np.isnan(sf.surr_data[:, lat, lon])) == False:
                        wave, _, _, _ = wvlt.continous_wavelet(sf.surr_data[:, lat, lon], 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0)
                        phase = np.arctan2(np.imag(wave), np.real(wave))
                        # subselect surr_data and phase
                        d = sf.surr_data[idx, lat, lon].copy()
                        phase = phase[0, idx]
                        cond_means_temp = np.zeros((8,))
                        for i in range(cond_means_temp.shape[0]):
                            ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
                            if MEANS:
                                cond_means_temp[i] = np.mean(d[ndx])
                            else:
                                cond_means_temp[i] = np.var(d[ndx], ddof = 1)
                        mf_diffs[lat, lon] = cond_means_temp.max() - cond_means_temp.min()
                        mf_mean[lat, lon] = np.mean(cond_means_temp)
                        del cond_means_temp, phase, wave, d
                    else:
                        mf_diffs[lat, lon] = np.nan
                        mf_mean[lat, lon] = np.nan
            surr_diffs[0, ...] = mf_diffs
            surr_means[0, ...] = mf_mean

        if SURR_TYPE[1]: # FT surrs
            ft_diffs = np.zeros((sf.data.shape[1], sf.data.shape[2]))
            ft_mean = np.zeros_like(ft_diffs)
            sf.construct_fourier_surrogates()
            sf.add_seasonality(mean, var, trend)
            for lat in range(sf.lats.shape[0]):
                for lon in range(sf.lons.shape[0]):
                    if np.all(np.isnan(sf.surr_data[:, lat, lon])) == False:
                        wave, _, _, _ = wvlt.continous_wavelet(sf.surr_data[:, lat, lon], 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0)
                        phase = np.arctan2(np.imag(wave), np.real(wave))
                        # subselect surr_data and phase
                        d = sf.surr_data[idx, lat, lon].copy()
                        phase = phase[0, idx]
                        cond_means_temp = np.zeros((8,))
                        for i in range(cond_means_temp.shape[0]):
                            ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
                            if MEANS:
                                cond_means_temp[i] = np.mean(d[ndx])
                            else:
                                cond_means_temp[i] = np.var(d[ndx], ddof = 1)
                        ft_diffs[lat, lon] = cond_means_temp.max() - cond_means_temp.min()
                        ft_mean[lat, lon] = np.mean(cond_means_temp)
                        del cond_means_temp, phase, wave, d
                    else:
                        ft_diffs[lat, lon] = np.nan
                        ft_mean[lat, lon] = np.nan
            surr_diffs[1, ...] = ft_diffs
            surr_means[1, ...] = ft_mean

        if SURR_TYPE[2]: # AR(1) surrs
            ar_diffs = np.zeros((sf.data.shape[1], sf.data.shape[2]))
            ar_mean = np.zeros_like(ar_diffs)
            sfAR.construct_surrogates_with_residuals()
            for lat in range(sfAR.lats.shape[0]):
                for lon in range(sfAR.lons.shape[0]):
                    if np.all(np.isnan(sfAR.surr_data[:, lat, lon])) == False: 
                        wave, _, _, _ = wvlt.continous_wavelet(sfAR.surr_data[:, lat, lon], 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0)
                        phase = np.arctan2(np.imag(wave), np.real(wave))
                        # subselect surr_data and phase
                        d = sfAR.surr_data[idx, lat, lon].copy()
                        phase = phase[0, idx]
                        cond_means_temp = np.zeros((8,))
                        for i in range(cond_means_temp.shape[0]):
                            ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
                            if MEANS:
                                cond_means_temp[i] = np.mean(d[ndx])
                            else:
                                cond_means_temp[i] = np.var(d[ndx], ddof = 1)
                        ar_diffs[lat, lon] = cond_means_temp.max() - cond_means_temp.min()
                        ar_mean[lat, lon] = np.mean(cond_means_temp)
                        del cond_means_temp, phase, wave, d
                    else:
                        ar_diffs[lat, lon] = np.nan
                        ar_mean[lat, lon] = np.nan
            surr_diffs[2, ...] = ar_diffs
            surr_means[2, ...] = ar_mean
                    
        resq.put((surr_diffs, surr_means))
        

## wavelet surrogates
if np.sum(SURR_TYPE) != 0:
    log("Computing %d MF/FT/AR(1) surrogates in parallel using %d workers..." % (NUM_SURR, WORKERS))
    surr_MIX_diff = np.zeros((3, NUM_SURR, sg.data.shape[1], sg.data.shape[2]))
    surr_MIX_mean = np.zeros_like(surr_MIX_diff)
    surr_completed = 0
    jobQ = Queue()
    resQ = Queue()
    for i in range(NUM_SURR):
        jobQ.put(1)
    for i in range(WORKERS):
        jobQ.put(None)
    
    seasonality = mean, var, trend
    if SURR_TYPE[2]:
        sgAR = sgAR
    else:
        sgAR = None
    workers = [Process(target = _analysis_surrogates, args = (sg, sgAR, seasonality, IDX, jobQ, resQ)) for iota in range(WORKERS)]

    del seasonality
    
    t_start = datetime.now()

    log("Starting workers...")
    for w in workers:
        w.start()
        
    t_last = t_start

    while surr_completed < NUM_SURR:
        # get result
        surr_diff, surr_mean = resQ.get()
        for surr_type in range(surr_diff.shape[0]):
            surr_MIX_diff[surr_type, surr_completed, ...] = surr_diff[surr_type]
            surr_MIX_mean[surr_type, surr_completed, ...] = surr_mean[surr_type]
        surr_completed += 1
        
        # time to go
        t_now = datetime.now()
        
        if (t_now - t_last).total_seconds() > 60:
            t_last = t_now
            dt = (t_now - t_start) / surr_completed * (NUM_SURR - surr_completed)
            log("PROGRESS: %d/%d complete, predicted completition at %s..."
                   % (surr_completed, NUM_SURR, str(t_now+dt)))
                   
    for w in workers:
        w.join()
        
    log("Analysis on surrogates done after %s. Now saving data..." % (str(datetime.now() - t_start)))

    ## save file with surrogates
    if not MEANS: # from variance to standard deviation
        difference_data = np.sqrt(difference_data)
        mean_data = np.sqrt(mean_data)
        surr_MIX_diff = np.sqrt(surr_MIX_diff)
        surr_MIX_mean = np.sqrt(surr_MIX_mean)
    fname = ('result/ECA-D_%s_cond_%s_FTsurrogates_%s_areaof_%s.bin' % ('SATA' if ANOMALISE else 'SAT', 
             'means' if MEANS else 'std', str(START_DATE), str(LONS[0])+'-'+str(LONS[1])))
    with open(fname, 'w') as f:
        cPickle.dump({'difference_surrogates' : surr_MIX_diff, 'mean surrogates' : surr_MIX_mean,
                      'surrogates_type' : SURR_TYPE}, f)

os.remove('ECA-D_temp_file_seasonality.bin')
log("Saved.")
if LOG:
    log_file.close()
