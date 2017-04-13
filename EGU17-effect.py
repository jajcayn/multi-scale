import numpy as np
from src.data_class import load_station_data
from datetime import date
import matplotlib.pyplot as plt


def get_equidistant_bins(bins = 8):
    return np.array(np.linspace(-np.pi, np.pi, bins+1))

mons = {1: 'J', 2: 'F', 3: 'M', 4: 'A', 5: 'M', 6: 'J', 7: 'J', 8: 'A', 9: 'S', 10: 'O', 11: 'N', 12: 'D'}

WINDOW_LENGTH = 36 # years
SEASON = [6, 7, 8]
# SEASON = None
param_window = 32 # years

prg = load_station_data('../data/ECAstation-TG/TG_STAID000027.txt', date(1775, 1, 1), date(2016, 5, 1), 
    anom = False, offset = 1)

prg.wavelet(1, 'y', cut = 4, cut_time = False, cut_data = False, regress_amp_to_data = True)
annual_amp = prg.amplitude.copy()

prg.anomalise()
prg.wavelet(8, 'y', cut = 4, cut_time = False, cut_data = False, regress_amp_to_data = True, continuous_phase = False)
amplitude = prg.amplitude.copy()
prg.get_parametric_phase(8, param_window, 'y', cut = 4, cut_time = True, cut_data = True, continuous_phase = False)

if SEASON is not None:
    ndx_season = prg.select_months(SEASON, apply_to_data = True)
    annual_amp = annual_amp[ndx_season]
    # prg.data = prg.data[ndx_season]
    amplitude = amplitude[ndx_season]
    prg.phase = prg.phase[ndx_season]
    # prg.time = prg.time[ndx_season]
bins = get_equidistant_bins()

ndxs, dates = prg.get_sliding_window_indexes(window_length = WINDOW_LENGTH, window_shift = 1, unit = 'y', return_half_dates = True)
n_windows = len(ndxs)
amp_windows = np.zeros((n_windows))
effect_windows = np.zeros((n_windows))
mean_amp_windows = np.zeros((n_windows))

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


l1, = plt.plot(effect_windows, linewidth = 2., color = "#1f77b4")
plt.ylabel("TEMP", size = 20, color = l1.get_color())
plt.xticks(np.arange(0, len(ndxs), 10), [d.year for d in dates[0::10]], rotation = 30)
if SEASON is None:
    plt.ylim([0, 2.5])
else:
    plt.ylim([0, 6])
plt.gca().twinx()
l2, = plt.plot(mean_amp_windows, linewidth = 1.2, color = "#ff7f0e")
l3, = plt.plot(amp_windows, linewidth = 1.7, color = "#2ca02c")
plt.legend([l1, l2, l3], ["8yr effect on SATA", "mean 8yr amplitude", "8yr effect on AAC"])
plt.xticks(np.arange(0, len(ndxs), 10), [d.year for d in dates[0::10]], rotation = 30)
plt.xlabel("YEAR", size = 20)
plt.ylabel("TEMP", size = 20, color = l2.get_color())
plt.ylim([0, 1.5])
if SEASON is None:
    plt.title("PRG station: full year \n sliding window: %d years, shift: %d year | param. est. window: %dyrs" 
        % (WINDOW_LENGTH, 1, param_window), size = 28)
else:
    plt.title("PRG station: %s \n sliding window: %d years, shift: %d year | param. est. window: %dyrs" 
        % (''.join([mons[s] for s in SEASON]), WINDOW_LENGTH, 1, param_window), size = 28)
plt.show()






