import numpy as np
from src.data_class import load_station_data
from datetime import date
# import matplotlib.pyplot as plt
import scipy.stats as sts
from src.surrogates import SurrogateField
from multiprocessing import Pool


def get_equidistant_bins(bins = 8):
    return np.array(np.linspace(-np.pi, np.pi, bins+1))

def _get_surrs_stats(a):
    sg, ndx, mean, var, trend, SEASON = a
    # create surrs
    sg.construct_fourier_surrogates(algorithm = 'AAFT')
    sg.add_seasonality(mean, var, trend)
    time_copy = sg.time.copy()

    ## COMPUTE FOR SURRS
    annual_phase, annual_amp = sg.wavelet(1, 'y', cut = 1, ts = sg.get_surr(), cut_time = False, cut_data = False, 
        regress_amp_to_data = True)

    sg.anomalise()
    phase, amplitude = sg.wavelet(8, 'y', cut = 1, ts = sg.get_surr(), cut_time = False, cut_data = False, 
        regress_amp_to_data = True, continuous_phase = False)
    _, amplitudeAACreg = sg.wavelet(8, 'y', cut = 1, ts = sg.get_surr(), cut_time = True, cut_data = True, 
        regress_amp_to_data = False, continuous_phase = False)
    # sg.surr_data = sg.surr_data[int(1*365.25):-int(1*365.25)]
    sg.time = sg.time[int(1*365.25):-int(1*365.25)]

    m, c, r, p, std_err = sts.linregress(amplitudeAACreg*np.cos(phase), annual_amp*np.cos(annual_phase))
    amplitudeAACreg = m*amplitudeAACreg + c

    if SEASON is not None:
        ndx_season = sg.select_months(SEASON, apply_to_data = False)
        annual_amp = annual_amp[ndx_season]
        amplitude = amplitude[ndx_season]
        amplitudeAACreg = amplitudeAACreg[ndx_season]
        phase = phase[ndx_season]
        sg.surr_data = sg.surr_data[ndx_season]
    bins = get_equidistant_bins()

    cond_means_temp = np.zeros((8,4))
    for j in range(cond_means_temp.shape[0]): # get conditional means for current phase range
        effect_ndx = ((phase >= bins[j]) & (phase <= bins[j+1]))
        cond_means_temp[j, 0] = np.mean(sg.surr_data[effect_ndx])
        cond_means_temp[j, 1] = np.mean(annual_amp[effect_ndx])
        cond_means_temp[j, 2] = np.mean(amplitude[effect_ndx])
        cond_means_temp[j, 3] = np.mean(amplitudeAACreg[effect_ndx])
    amp_windows = cond_means_temp[:, 1].max() - cond_means_temp[:, 1].min()
    effect_windows = cond_means_temp[:, 0].max() - cond_means_temp[:, 0].min()
    mean_amp_windows = cond_means_temp[:, 2].max() - cond_means_temp[:, 2].min()
    mean_ampAAC_windows = cond_means_temp[:, 3].max() - cond_means_temp[:, 3].min()

    sg.time = time_copy.copy()

    return (amp_windows, effect_windows, mean_amp_windows, mean_ampAAC_windows)

mons = {1: 'J', 2: 'F', 3: 'M', 4: 'A', 5: 'M', 6: 'J', 7: 'J', 8: 'A', 9: 'S', 10: 'O', 11: 'N', 12: 'D'}

NUM_SURRS = 1000
WINDOW_LENGTH = 36 # years
# SEASON = [12,1,2]
# SEASON = None
# param_window = 32 # years
SEASONS = [None, [3,4,5], [6,7,8], [9,10,11], [12, 1, 2]]

for SEASON in SEASONS:

    print ''.join([mons[s] for s in SEASON]) if SEASON is not None else 'overall'

    # prg = load_station_data('../data/ECAstation-TG/TG_STAID000027.txt', date(1775, 1, 1), date(2016, 5, 1), 
        # anom = False, offset = 1)
    prg = load_station_data('../data/TG_STAID000027.txt', date(1775, 1, 1), date(2016, 5, 1), 
        anom = False, offset = 1)

    bins = get_equidistant_bins()

    ndxs, dates = prg.get_sliding_window_indexes(window_length = WINDOW_LENGTH, window_shift = 1, unit = 'y', return_half_dates = True)
    n_windows = len(ndxs)
    amp_windows = np.zeros((n_windows))
    effect_windows = np.zeros((n_windows))
    mean_amp_windows = np.zeros((n_windows))
    mean_ampAAC_windows = np.zeros((n_windows))
    amp_windows_surrs = np.zeros((NUM_SURRS, n_windows))
    effect_windows_surrs = np.zeros((NUM_SURRS, n_windows))
    mean_amp_windows_surrs = np.zeros((NUM_SURRS, n_windows))
    mean_ampAAC_windows_surrs = np.zeros((NUM_SURRS, n_windows))

    for i, ndx in zip(range(len(ndxs)), ndxs):
        # copy part of data
        prg_temp = prg.copy(temporal_ndx = ndx)

        # get ready for surrs
        mean, var, trend = prg_temp.get_seasonality(detrend = True)
        prg_surr = SurrogateField()
        prg_surr.copy_field(prg_temp)
        prg_temp.return_seasonality(mean, var, trend)

        ## COMPUTE FOR DATA
        prg_temp.wavelet(1, 'y', cut = 1, cut_time = False, cut_data = False, regress_amp_to_data = True)
        annual_amp = prg_temp.amplitude.copy()
        annual_phase = prg_temp.phase.copy()

        prg_temp.anomalise()
        prg_temp.wavelet(8, 'y', cut = 1, cut_time = False, cut_data = False, regress_amp_to_data = True, continuous_phase = False)
        amplitude = prg_temp.amplitude.copy()
        prg_temp.wavelet(8, 'y', cut = 1, cut_time = True, cut_data = True, regress_amp_to_data = False, continuous_phase = False)
        amplitudeAACreg = prg_temp.amplitude.copy()

        m, c, r, p, std_err = sts.linregress(amplitudeAACreg*np.cos(prg_temp.phase), annual_amp*np.cos(annual_phase))
        amplitudeAACreg = m*amplitudeAACreg + c

        if SEASON is not None:
            ndx_season = prg_temp.select_months(SEASON, apply_to_data = True)
            annual_amp = annual_amp[ndx_season]
            amplitude = amplitude[ndx_season]
            amplitudeAACreg = amplitudeAACreg[ndx_season]
            prg_temp.phase = prg_temp.phase[ndx_season]

        cond_means_temp = np.zeros((8,4))
        for j in range(cond_means_temp.shape[0]): # get conditional means for current phase range
            effect_ndx = ((prg_temp.phase >= bins[j]) & (prg_temp.phase <= bins[j+1]))
            cond_means_temp[j, 0] = np.mean(prg_temp.data[effect_ndx])
            cond_means_temp[j, 1] = np.mean(annual_amp[effect_ndx])
            cond_means_temp[j, 2] = np.mean(amplitude[effect_ndx])
            cond_means_temp[j, 3] = np.mean(amplitudeAACreg[effect_ndx])
        amp_windows[i] = cond_means_temp[:, 1].max() - cond_means_temp[:, 1].min()
        effect_windows[i] = cond_means_temp[:, 0].max() - cond_means_temp[:, 0].min()
        mean_amp_windows[i] = cond_means_temp[:, 2].max() - cond_means_temp[:, 2].min()
        mean_ampAAC_windows[i] = cond_means_temp[:, 3].max() - cond_means_temp[:, 3].min()

        ## COMPUTE FOR SURRS
        pool = Pool(20)
        args = [(prg_surr, ndx, mean, var, trend, SEASON) for _ in range(NUM_SURRS)]
        results = pool.map(_get_surrs_stats, args)
        for res, j in zip(results, range(len(results))):
            amp_windows_surrs[j, i] = res[0]
            effect_windows_surrs[j, i] = res[1]
            mean_amp_windows_surrs[j, i] = res[2]
            mean_ampAAC_windows_surrs[j, i] = res[3]

        pool.close()
        pool.join()


    ## SAVE RESULTS
    import cPickle
    if SEASON is None:
        fname = "PRG-8yr-effect-linear-nonlinear-%d-AAFTsurrs-all-windows.bin" % (NUM_SURRS)
    else:
        fname = "PRG-8yr-effect-linear-nonlinear%s-%d-AAFTsurrs-all-windows.bin" % (''.join([mons[s] for s in SEASON]), NUM_SURRS)
    with open(fname, 'wb') as f:
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






