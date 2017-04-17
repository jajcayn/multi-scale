import numpy as np
from src.data_class import load_station_data
from datetime import date
import matplotlib.pyplot as plt
import scipy.stats as sts
from src.surrogates import SurrogateField
from multiprocessing import Pool


def get_equidistant_bins(bins = 8):
    return np.array(np.linspace(-np.pi, np.pi, bins+1))

mons = {1: 'J', 2: 'F', 3: 'M', 4: 'A', 5: 'M', 6: 'J', 7: 'J', 8: 'A', 9: 'S', 10: 'O', 11: 'N', 12: 'D'}

NUM_SURRS = 10000
WINDOW_LENGTH = 36 # years
# SEASON = [9,10,11]
SEASON = None
# param_window = 32 # years

# prg = load_station_data('../data/ECAstation-TG/TG_STAID000027.txt', date(1775, 1, 1), date(2016, 5, 1), 
    # anom = False, offset = 1)
prg = load_station_data('../data/TG_STAID000027.txt', date(1775, 1, 1), date(2016, 5, 1), 
    anom = False, offset = 1)
# get ready for surrs
mean, var, trend = prg.get_seasonality(detrend = True)
prg_surr = SurrogateField()
prg_surr.copy_field(prg)
prg.return_seasonality(mean, var, trend)

def _get_surrs_stats(a):
    sg, ndxs, mean, var, trend, SEASON = a
    # create surrs
    sg.construct_fourier_surrogates_spatial()
    sg.add_seasonality(mean, var, trend)
    time_copy = sg.time.copy()

    ## COMPUTE FOR SURRS
    annual_phase, annual_amp = sg.wavelet(1, 'y', cut = 4, ts = sg.get_surr(), cut_time = False, cut_data = False, 
        regress_amp_to_data = True)

    sg.anomalise(ts = sg.surr_data)
    phase, amplitude = sg.wavelet(8, 'y', cut = 4, ts = sg.get_surr(), cut_time = False, cut_data = False, 
        regress_amp_to_data = True, continuous_phase = False)
    _, amplitudeAACreg = sg.wavelet(8, 'y', cut = 4, ts = sg.get_surr(), cut_time = True, cut_data = True, 
        regress_amp_to_data = False, continuous_phase = False)
    sg.surr_data = sg.surr_data[int(4*365.25):-int(4*365.25)]
    sg.time = sg.time[int(4*365.25):-int(4*365.25)]

    m, c, r, p, std_err = sts.linregress(amplitudeAACreg*np.cos(phase), annual_amp*np.cos(annual_phase))
    amplitudeAACreg = m*amplitudeAACreg + c

    if SEASON is not None:
        ndx_season = sg.select_months(SEASON, apply_to_data = True)
        annual_amp = annual_amp[ndx_season]
        amplitude = amplitude[ndx_season]
        amplitudeAACreg = amplitudeAACreg[ndx_season]
        phase = phase[ndx_season]
        sg.surr_data = sg.surr_data[ndx_season]
    bins = get_equidistant_bins()

    n_windows = len(ndxs)
    amp_windows = np.zeros((n_windows))
    effect_windows = np.zeros((n_windows))
    mean_amp_windows = np.zeros((n_windows))
    mean_ampAAC_windows = np.zeros((n_windows))

    for i, ndx in zip(range(len(ndxs)), ndxs):
        cond_means_temp = np.zeros((8,2))
        data_temp = sg.surr_data[ndx].copy()
        amp_temp = annual_amp[ndx].copy()
        phase_temp = sg.phase[ndx].copy()
        for j in range(cond_means_temp.shape[0]): # get conditional means for current phase range
            effect_ndx = ((phase_temp >= bins[j]) & (phase_temp <= bins[j+1]))
            cond_means_temp[j, 0] = np.mean(data_temp[effect_ndx])
            cond_means_temp[j, 1] = np.mean(amp_temp[effect_ndx])
        amp_windows[i] = cond_means_temp[:, 1].max() - cond_means_temp[:, 1].min()
        effect_windows[i] = cond_means_temp[:, 0].max() - cond_means_temp[:, 0].min()
        mean_amp_windows[i] = np.mean(amplitude[ndx])
        mean_ampAAC_windows[i] = np.mean(amplitudeAACreg[ndx])

    sg.time = time_copy.copy()

    return (amp_windows, effect_windows, mean_amp_windows, mean_ampAAC_windows)


## COMPUTE FOR DATA
prg.wavelet(1, 'y', cut = 4, cut_time = False, cut_data = False, regress_amp_to_data = True)
annual_amp = prg.amplitude.copy()
annual_phase = prg.phase.copy()

prg.anomalise()
prg.wavelet(8, 'y', cut = 4, cut_time = False, cut_data = False, regress_amp_to_data = True, continuous_phase = False)
amplitude = prg.amplitude.copy()
prg.wavelet(8, 'y', cut = 4, cut_time = True, cut_data = True, regress_amp_to_data = False, continuous_phase = False)
amplitudeAACreg = prg.amplitude.copy()

m, c, r, p, std_err = sts.linregress(amplitudeAACreg*np.cos(prg.phase), annual_amp*np.cos(annual_phase))
amplitudeAACreg = m*amplitudeAACreg + c

if SEASON is not None:
    ndx_season = prg.select_months(SEASON, apply_to_data = True)
    annual_amp = annual_amp[ndx_season]
    amplitude = amplitude[ndx_season]
    amplitudeAACreg = amplitudeAACreg[ndx_season]
    prg.phase = prg.phase[ndx_season]
bins = get_equidistant_bins()

ndxs, dates = prg.get_sliding_window_indexes(window_length = WINDOW_LENGTH, window_shift = 1, unit = 'y', return_half_dates = True)
n_windows = len(ndxs)
amp_windows = np.zeros((n_windows))
effect_windows = np.zeros((n_windows))
mean_amp_windows = np.zeros((n_windows))
mean_ampAAC_windows = np.zeros((n_windows))

for i, ndx in zip(range(len(ndxs)), ndxs):
    cond_means_temp = np.zeros((8,2))
    data_temp = prg.data[ndx].copy()
    amp_temp = annual_amp[ndx].copy()
    phase_temp = prg.phase[ndx].copy()
    for j in range(cond_means_temp.shape[0]): # get conditional means for current phase range
        effect_ndx = ((phase_temp >= bins[j]) & (phase_temp <= bins[j+1]))
        cond_means_temp[j, 0] = np.mean(data_temp[effect_ndx])
        cond_means_temp[j, 1] = np.mean(amp_temp[effect_ndx])
    amp_windows[i] = cond_means_temp[:, 1].max() - cond_means_temp[:, 1].min()
    effect_windows[i] = cond_means_temp[:, 0].max() - cond_means_temp[:, 0].min()
    mean_amp_windows[i] = np.mean(amplitude[ndx])
    mean_ampAAC_windows[i] = np.mean(amplitudeAACreg[ndx])


## COMPUTE FOR SURRS
pool = Pool(20)
args = [(prg_surr, ndxs, mean, var, trend, SEASON) for i in range(NUM_SURRS)]
results = pool.map(_get_surrs_stats, args)
amp_windows_surrs = np.zeros((NUM_SURRS, n_windows))
effect_windows_surrs = np.zeros((NUM_SURRS, n_windows))
mean_amp_windows_surrs = np.zeros((NUM_SURRS, n_windows))
mean_ampAAC_windows_surrs = np.zeros((NUM_SURRS, n_windows))

for res,i in zip(results, range(len(results))):
    amp_windows_surrs[i, :] = res[0]
    effect_windows_surrs[i, :] = res[1]
    mean_amp_windows_surrs[i, :] = res[2]
    mean_ampAAC_windows_surrs[i, :] = res[3]

import cPickle

with open("PRG-8yr-effect-linear-nonlinear-%d-FTsurrs.bin" % (NUM_SURRS), 'wb') as f:
    cPickle.dump({'amp_windows' : amp_windows, 'effect_windows' : effect_windows,
        'mean_amp_windows' : mean_amp_windows, 'mean_ampAAC_windows' : mean_ampAAC_windows,
        'amp_windows_surrs' : amp_windows_surrs, 'effect_windows_surrs' : effect_windows_surrs,
        'mean_amp_windows_surrs' : mean_amp_windows_surrs, 'mean_ampAAC_windows_surrs' : mean_ampAAC_windows_surrs},
         f, protocol = cPickle.HIGHEST_PROTOCOL)


# # l1, = plt.plot(effect_windows, linewidth = 2., color = "#1f77b4")
# plt.ylabel("TEMP", size = 20)
# plt.xticks(np.arange(0, len(ndxs), 10), [d.year for d in dates[0::10]], rotation = 30)
# if SEASON is None:
#     plt.ylim([-0.5, 2.5])
# else:
#     plt.ylim([-0.5, 2.5])
# # plt.gca().twinx()
# l2, = plt.plot(mean_amp_windows, linewidth = 1.2, color = "#ff7f0e")
# l3, = plt.plot(amp_windows, linewidth = 1.7, color = "#2ca02c")
# # plt.legend([l1, l2], ["8yr effect on SATA", "mean 8yr amplitude"])
# plt.legend([l2, l3], ["mean 8yr amplitude in AAC", "8yr effect on AAC"])
# plt.xticks(np.arange(0, len(ndxs), 10), [d.year for d in dates[0::10]], rotation = 30)
# plt.xlabel("YEAR", size = 20)
# plt.ylabel("TEMP", size = 20)
# # plt.ylim([0, 2.5])
# if SEASON is None:
#     # plt.title("PRG station: full year \n sliding window: %d years, shift: %d year | param. est. window: %dyrs" 
#         # % (WINDOW_LENGTH, 1, param_window), size = 28)
#     plt.title("PRG station: full year \n sliding window: %d years, shift: %d year" 
#         % (WINDOW_LENGTH, 1), size = 28)
# else:
#     # plt.title("PRG station: %s \n sliding window: %d years, shift: %d year | param. est. window: %dyrs" 
#         # % (''.join([mons[s] for s in SEASON]), WINDOW_LENGTH, 1, param_window), size = 28)
#     plt.title("PRG station: %s \n sliding window: %d years, shift: %d year" 
#         % (''.join([mons[s] for s in SEASON]), WINDOW_LENGTH, 1), size = 28)
# plt.show()






