"""
created on June 13, 2014

@author: Nikola Jajcay
"""

from src import wavelet_analysis as wvlt
from src.data_class import load_ECA_D_data_daily, load_ERA_data_daily
from surrogates.surrogates import SurrogateField
import numpy as np
from datetime import datetime, date
import cPickle
from multiprocessing import Pool
#import hickle as hkl



## functions
def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))
    
    
def _get_oscillatory_modes(a):
    """
    Gets oscillatory modes in terms of phase and amplitude from wavelet analysis for given data.
    """
    i, j, s0, data = a
    if not np.all(np.isnan(data)):
        wave, _, _, _ = wvlt.continous_wavelet(data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = 6.)
        phase = np.arctan2(np.imag(wave), np.real(wave))
    else:
        phase = np.nan
    
    return i, j, phase


def _get_amplitude(a):
    """
    Gets amplitude of yearly cycle from SAT data.
    """
    i, j, s0_amp, data = a
    if not np.any(np.isnan(data)):
        wave, _, _, _ = wvlt.continous_wavelet(data, 1, False, wvlt.morlet, dj = 0, s0 = s0_amp, j1 = 0, k0 = 6.) # perform wavelet
        amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))
        amplitude = amplitude[0, :]
        phase_amp = np.arctan2(np.imag(wave), np.real(wave))
        phase_amp = phase_amp[0, :]
        # fitting oscillatory phase / amplitude to actual SAT
        reconstruction = amplitude * np.cos(phase_amp)
        fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
        m, c = np.linalg.lstsq(fit_x, data)[0]
        amplitude = m * amplitude + c
    else:
        amplitude = np.nan

    return i, j, amplitude
        
    
def _get_cond_means(a):
    """
    Gets either conditional means or variance according to phase.
    """
    i, j, phase, data, phase_bins = a
    cond_means_temp_m = np.zeros((8,))
    cond_means_temp_v = np.zeros((8,))
    if not np.all(np.isnan(phase)):
        for iota in range(cond_means_temp_m.shape[0]):
            ndx = ((phase >= phase_bins[iota]) & (phase <= phase_bins[iota+1]))
            cond_means_temp_m[iota] = np.mean(data[ndx])
            cond_means_temp_v[iota] = np.std(data[ndx], ddof = 1)
    else:
        cond_means_temp_m = [np.nan for ii in range(8)]
        cond_means_temp_v = [np.nan for ii in range(8)]

    return i, j, cond_means_temp_m, cond_means_temp_v 
    
    

ECA = True # if False ERA-40 will be evaluated instead
ANOMALISE = True # if True, data will be anomalised hence SAT -> SATA
WORKERS = 10 # number of threads, if 0, all computations will be run single-thread
PERIOD = 8 # years; central period of wavelet used
START_DATE = date(1958,1,1)
LATS = None #[25.375, 75.375] # lats ECA: 25.375 -- 75.375 = 201 grid points
LONS = None #[-40.375, -11.375] #lons ECA: -40.375 -- 75.375 = 464 grid points
SURR_TYPE = 'MF' # None, for data, MF, FT, AR or ALL (use only with ERA reanalysis, not ECA&D)
NUM_SURR = 100 # number of surrogates to be evaluated
NUM_FILES = 1
LOG = True # if True, output will be written to log defined in log_file, otherwise printed to screen
SEASON = None
AMPLITUDE = True # season cannot be used with amplitude, it does not make any sense
# warning: logging into log file will suppress printing warnings handled by modules e.g. numpy's warnings


if LOG and ECA:
    log_file = open('result/ECA-D_%s_cond_mean_var_%s.log' % ('SATA' if ANOMALISE else 'SAT', datetime.now().strftime('%Y%m%d-%H%M')), 'w')
elif LOG and not ECA:
    log_file = open('result/ERA_%s_cond_mean_var_%s.log' % ('SATA' if ANOMALISE else 'SAT', datetime.now().strftime('%Y%m%d-%H%M')), 'w')

def log(msg):
    if LOG:
        log_file.write('[%s] %s\n' % (str(datetime.now()), msg))
        log_file.flush()
    else:
        print("[%s] %s" % (str(datetime.now()), msg))

## load and prepare data
# ECA&D
if ECA:
    g = load_ECA_D_data_daily('tg_0.25deg_reg_v10.0.nc', 'tg', date(1950,1,1), date(2014,1,1), 
                                LATS, LONS, False, logger_function = log)
    if AMPLITUDE:
        log("Evaluating amplitude of the yearly cycle instead of total SAT(A) variability...")
        g_amp = g.copy_data()
        log("SAT data copied...")
                            
# ERA-40 + ERA-Interim
else:
    g = load_ERA_data_daily('ERA40_EU', 't2m', date(1958,1,1), date(2004,1,1), LATS, LONS, False, 
                             parts = 3, logger_function = log)
                             
idx = g.get_data_of_precise_length('16k', START_DATE, None, True) # get 2^n data because of MF surrogates
if AMPLITUDE:
    g_amp = g_amp[idx[0] : idx[1], ...]
END_DATE = g.get_date_from_ndx(-1)

if SURR_TYPE is not None:
    log("Creating surrogate fields...")
    sg = SurrogateField() # for MF and FT surrs
    log("De-seasonalising the data and copying to surrogate field...")
    mean, var, trend = g.get_seasonality(True) # subtract mean, divide by std and subtract trend from data
    sg.copy_field(g) # copy standartised data to SurrogateField
    g.return_seasonality(mean, var, trend) # return seasonality to data
    log("Surrogate fields created.")

if ANOMALISE:
    g.anomalise()



## wavelet data
log("Running wavelet analysis on data using %d workers..." % (WORKERS))
k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale

phase_data = np.zeros_like(g.data)

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

if AMPLITUDE:
    log("Computing amplitude from SAT data using %d workers..." % (WORKERS))
    s0_amp = 1 * y / fourier_factor
    amp_data = np.zeros_like(g_amp)

    job_args = [ (i, j, s0_amp, g_amp[:, i, j]) for i in range(g.lats.shape[0]) for j in range(g.lons.shape[0]) ]
    job_results = map_func(_get_amplitude, job_args)
    del job_args
    # map results
    for i, j, amp in job_results:
        amp_data[:, i, j] = amp
    del job_results

# select interior of wavelet to suppress unwanted edge effects
IDX = g.select_date(date(START_DATE.year + 4, START_DATE.month, START_DATE.day), 
                    date(END_DATE.year - 4, END_DATE.month, END_DATE.day))

phase_data = phase_data[IDX, ...]
if AMPLITUDE:
    amp_data = amp_data[IDX, ...]

log("Wavelet on data done. Computing conditional mean and variance on data...")

# subselect season if desired
if SEASON != None:
    season_list = [str(mon) for mon in SEASON]
    log("Selecting only %s months..." % (', '.join(season_list)))
    NDX_SEASON = g.select_months(SEASON)
    phase_data = phase_data[NDX_SEASON, ...]


bins_data = np.zeros(g.get_spatial_dims() + [8])
bins_data_var = np.zeros(g.get_spatial_dims() + [8])
phase_bins = get_equidistant_bins()

if AMPLITUDE:
    job_args = [ (i, j, phase_data[:, i, j], amp_data[:, i, j], phase_bins) for i in range(g.lats.shape[0]) for j in range(g.lons.shape[0]) ]
else:
    job_args = [ (i, j, phase_data[:, i, j], g.data[:, i, j], phase_bins) for i in range(g.lats.shape[0]) for j in range(g.lons.shape[0]) ]
job_result = map_func(_get_cond_means, job_args)
del job_args, phase_data
# map results
for i, j, cmm, cmv in job_result:
    bins_data[i, j, :] = cmm
    bins_data_var[i, j, :] = cmv
del job_result


# if surrogate type is AR, exploit the pool to prepare the AR model
if SURR_TYPE == 'AR' or SURR_TYPE == 'ALL':
    log("Creating AR(1) model and computing residuals...")
    sg.prepare_AR_surrogates(pool)

if pool is not None:
    pool.close()
    pool.join()
    del pool
        
log("Analysis on data done. Saving file...")
## save file in case something will go wrong with surrogates..
# from variance to standard deviation
if ECA:
    fname = ('result/ECA-D_%s%s_cond_mean_var_data_from_%s_16k' % ('SATamplitude_' if AMPLITUDE else '', 'SATA' if ANOMALISE else 'SAT', str(START_DATE)))
else:
    fname = ('result/ERA_%s_cond_mean_var_data_from_%s_16k' % ('SATA' if ANOMALISE else 'SAT', str(START_DATE)))    
with open(fname + '.bin', 'wb') as f:
#    cPickle.dump({'difference_data' : difference_data, 'mean_data' : mean_data, 
#                   'difference_data_var' : difference_data_var, 'mean_data_var' : mean_data_var, 
#                   'lats' : g.lats, 'lons' : g.lons}, f, protocol = cPickle.HIGHEST_PROTOCOL)
    cPickle.dump({'bins_data' : bins_data, 'bins_data_var' : bins_data_var, 'season' : SEASON, 
                   'lats' : g.lats, 'lons' : g.lons}, f, protocol = cPickle.HIGHEST_PROTOCOL)
# hkl.dump({'difference_data' : difference_data, 'mean_data' : mean_data, 
#           'difference_data_var' : difference_data_var, 'mean_data_var' : mean_data_var,
#           'lats' : g.lats, 'lons' : g.lons}, fname + '.hkl', mode = 'w')
# hkl.dump({'bins_data' : bins_data, 'bins_data_var' : bins_data_var, 'season' : SEASON,
#           'lats' : g.lats, 'lons' : g.lons}, fname + '.hkl', mode = 'w')
    
# release the g object 
del g

## surrogates
if SURR_TYPE is not None:
    log("Computing %d %s surrogates in parallel using %d workers..." % (NUM_SURR, SURR_TYPE, WORKERS))
    log("Result will be saved in %d files..." % (NUM_FILES))
    SU = 3 if SURR_TYPE == 'ALL' else 1
#    surr_diff = np.zeros((SU, NUM_SURR, sg.data.shape[1], sg.data.shape[2]))
#    surr_mean = np.zeros_like(surr_diff)
#    surr_diff_var = np.zeros_like(surr_diff)
#    surr_mean_var = np.zeros_like(surr_diff)
    # bins_surrogates = np.zeros((SU, NUM_SURR, sg.data.shape[1], sg.data.shape[2], 8))
    # bins_surrogates_var = np.zeros_like(bins_surrogates)
    phase_bins = get_equidistant_bins()
    t_start = datetime.now()
    t_last = t_start
    pool = Pool(WORKERS)
    
    for su_type in range(SU):
        for file_num in range(NUM_FILES):
            bins_surrogates = np.zeros((SU, NUM_SURR/NUM_FILES, sg.data.shape[1], sg.data.shape[2], 8))
            bins_surrogates_var = np.zeros_like(bins_surrogates)
            for surr_completed in range(NUM_SURR/NUM_FILES):
                # create surrogates field
                if (SURR_TYPE == 'MF') or (SURR_TYPE == 'ALL' and su_type == 0):
                    sg.construct_multifractal_surrogates(pool = pool)
                    sg.add_seasonality(0, var, trend)
                elif (SURR_TYPE == 'FT') or (SURR_TYPE == 'ALL' and su_type == 1):
                    sg.construct_fourier_surrogates_spatial(pool = pool)
                    sg.add_seasonality(0, var, trend)
                elif (SURR_TYPE == 'AR') or (SURR_TYPE == 'ALL' and su_type == 2):
                    sg.construct_surrogates_with_residuals(pool = pool)
                    sg.add_seasonality(0, var[:-1, ...], trend[:-1, ...])
            
                # oscillatory modes
                phase_surrs = np.zeros_like(sg.surr_data)
                job_args = [ (i, j, s0, sg.surr_data[:, i, j]) for i in range(sg.lats.shape[0]) for j in range(sg.lons.shape[0]) ]
                job_result = pool.map(_get_oscillatory_modes, job_args)
                del job_args
                # map results
                for i, j, ph in job_result:
                    phase_surrs[:, i, j] = ph
                del job_result

                if AMPLITUDE:
                    # surrogates are now SATA, for amplitude we need SAT so plus mean, times 1 var, plus 0 trend
                    if SURR_TYPE == 'MF' or SURR_TYPE == 'FT':
                        sg.add_seasonality(mean, 1, 0)
                    elif SURR_TYPE == 'AR':
                        sg.add_seasonality(mean[:-1, ...], 1, 0)

                    amp_surrs = np.zeros_like(sg.surr_data)
                    log("surr data shape %s" % (str(sg.surr_data.shape)))
                    job_args = [ (i, j, s0_amp, sg.surr_data[:, i, j]) for i in range(sg.lats.shape[0]) for j in range(sg.lons.shape[0]) ]
                    log("arg created with len %d" % len(job_args))
                    log(map_func)
                    job_results = map_func(_get_amplitude, job_args)
                    del job_args
                    # map results
                    for i, j, amp in job_results:
                        amp_surrs[:, i, j] = amp
                    del job_results
        
                sg.surr_data = sg.surr_data[IDX, ...]
                phase_surrs = phase_surrs[IDX, ...]
                if AMPLITUDE:
                    amp_surrs = amp_surrs[IDX, ...]
                
                if SEASON != None:
                    sg.surr_data = sg.surr_data[NDX_SEASON, ...]
                    phase_surrs = phase_surrs[NDX_SEASON, ...]
                
                if AMPLITUDE:
                    job_args = [ (i, j, phase_surrs[:, i, j], amp_surrs[:, i, j], phase_bins) for i in range(sg.lats.shape[0]) for j in range(sg.lons.shape[0]) ]
                else:
                    job_args = [ (i, j, phase_surrs[:, i, j], sg.surr_data[:, i, j], phase_bins) for i in range(sg.lats.shape[0]) for j in range(sg.lons.shape[0]) ]
                job_result = pool.map(_get_cond_means, job_args)
                del job_args, phase_surrs
                # map results
                for i, j, cmm, cmv in job_result:
                    bins_surrogates[su_type, surr_completed, i, j, :] = cmm
                    bins_surrogates_var[su_type, surr_completed, i, j, :] = cmv
                del job_result
        
                # time to go
                t_now = datetime.now()
                if ((t_now - t_last).total_seconds() > 600) and surr_completed > 0:
                    t_last = t_now
                    dt = (t_now - t_start) / surr_completed * (NUM_SURR - surr_completed)
                    log("PROGRESS: %d/%d surrogate done, predicted completition at %s" % (surr_completed, NUM_SURR, 
                        str(t_now + dt)))

            if ECA:
                fname = ('result/ECA-D_%s%s_cond_mean_var_%ssurrogates_from_%s_16k_%d' % ('SATamplitude_' if AMPLITUDE else '', 
                            'SATA' if ANOMALISE else 'SAT', SURR_TYPE, str(START_DATE), file_num))
            else:
                fname = ('result/ERA_%s_cond_mean_var_%ssurrogates_from_%s_16k_%d' % ('SATA' if ANOMALISE else 'SAT', 
                            SURR_TYPE, str(START_DATE), file_num))
            with open(fname + '.bin', 'wb') as f:
                cPickle.dump({'bins_surrogates' : bins_surrogates, 'bins_surrogates_var' : bins_surrogates_var, 'season' : SEASON,
                      'surrogates_type' : SURR_TYPE}, f, protocol = cPickle.HIGHEST_PROTOCOL)
            log("Saved %d/%d file" % (file_num+1, NUM_FILES))

    if pool is not None:
        pool.close()
        pool.join()
        del pool

    log("Analysis on surrogates done after %s. Now saving data..." % (str(datetime.now() - t_start)))
    
    ## save file with surrogates

#==============================================================================
#     surr_diff_var = np.sqrt(surr_diff_var)
#     surr_mean_var = np.sqrt(surr_mean_var)
#==============================================================================
#     bins_surrogates_var = np.sqrt(bins_surrogates_var)
#     if ECA:
#         fname = ('result/ECA-D_%s_cond_mean_var_%ssurrogates_from_%s_16k' % ('SATA' if ANOMALISE else 'SAT', 
#                     SURR_TYPE, str(START_DATE)))
#     else:
#         fname = ('result/ERA_%s_cond_mean_var_%ssurrogates_from_%s_16k' % ('SATA' if ANOMALISE else 'SAT', 
#                     SURR_TYPE, str(START_DATE)))
#     with open(fname + '.bin', 'wb') as f:
# #        cPickle.dump({'difference_surrogates' : surr_diff, 'mean_surrogates' : surr_mean,
# #                      'difference_surrogates_var' : surr_diff_var, 'mean_surrogates_var' : surr_mean_var,
# #                      'surrogates_type' : SURR_TYPE}, f, protocol = cPickle.HIGHEST_PROTOCOL)
#         cPickle.dump({'bins_surrogates' : bins_surrogates, 'bins_surrogates_var' : bins_surrogates_var, 'season' : SEASON,
#                       'surrogates_type' : SURR_TYPE}, f, protocol = cPickle.HIGHEST_PROTOCOL)
   # hkl.dump({'difference_surrogates' : surr_diff, 'mean surrogates' : surr_mean,
   #             'difference_surrogates_var' : surr_diff_var, 'mean_surrogates_var' : surr_mean_var,
   #             'surrogates_type' : SURR_TYPE}, fname + '.hkl', mode = 'w')
    # hkl.dump({'bins_surrogates' : bins_surrogates, 'bins_surrogates_var' : bins_surrogates_var, 'season' : SEASON,
    #            'surrogates_type' : SURR_TYPE}, fname + '.hkl', mode = 'w')
                      
log("Saved all %d files." % (NUM_FILES))
if LOG:
    log_file.close()
