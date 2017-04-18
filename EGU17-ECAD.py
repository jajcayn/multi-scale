import numpy as np
from src.data_class import load_station_data, load_ECA_D_data_daily
from datetime import date
import matplotlib.pyplot as plt
import scipy.stats as sts
from src.surrogates import SurrogateField
from pathos.multiprocessing import Pool


def get_equidistant_bins(bins = 8):
    return np.array(np.linspace(-np.pi, np.pi, bins+1))

mons = {1: 'J', 2: 'F', 3: 'M', 4: 'A', 5: 'M', 6: 'J', 7: 'J', 8: 'A', 9: 'S', 10: 'O', 11: 'N', 12: 'D'}

NUM_SURRS = 10000
WINDOW_LENGTH = 36 # years
# SEASON = [9,10,11]
SEASON = None
# param_window = 32 # years

g = load_ECA_D_data_daily('../data/ECAD.tg.daily.0.50deg.nc', 'tg', date(1950,1,1), date(2016,8,31), 
                              None, None, False)
g.get_monthly_data()

pool = Pool(5)
bins = get_equidistant_bins()

g.wavelet(1, 'y', cut = 1, cut_time = False, cut_data = False, regress_amp_to_data = True, pool = pool)
annual_amp = g.amplitude.copy()
annual_phase = g.phase.copy()

g.anomalise()
g.wavelet(8, 'y', cut = 1, cut_time = False, cut_data = False, regress_amp_to_data = True, continuous_phase = False, 
    pool = pool)
amplitude = g.amplitude.copy()
g.wavelet(8, 'y', cut = 1, cut_time = True, cut_data = True, regress_amp_to_data = False, continuous_phase = False, 
    pool = pool)
amplitudeAACreg = g.amplitude.copy()

for lat in range(g.lats.shape[0]):
    for lon in range(g.lons.shape[0]):
        if not np.any(np.isnan(amplitudeAACreg[:, lat, lon])):
            m, c, r, p, std_err = sts.linregress(amplitudeAACreg[:, lat, lon]*np.cos(g.phase[:, lat, lon]), 
                annual_amp[:, lat, lon]*np.cos(annual_phase[:, lat, lon]))
            amplitudeAACreg[:, lat, lon] = m*amplitudeAACreg[:, lat, lon] + c

pool.close()
pool.join()

if SEASON is not None:
    ndx_season = g.select_months(SEASON, apply_to_data = True)
    annual_amp = annual_amp[ndx_season, ...]
    amplitude = amplitude[ndx_season, ...]
    amplitudeAACreg = amplitudeAACreg[ndx_season, ...]
    g.phase = g.phase[ndx_season, ...]

amp = np.zeros(g.get_spatial_dims())
effect = np.zeros(g.get_spatial_dims())
mean_amp = np.zeros(g.get_spatial_dims())
mean_ampAAC = np.zeros(g.get_spatial_dims())

cond_means_temp = np.zeros([8, 2] + g.get_spatial_dims())
for lat in range(g.lats.shape[0]):
    for lon in range(g.lons.shape[0]):
        if not np.any(np.isnan(g.data[:, lat, lon])):
            for j in range(cond_means_temp.shape[0]): # get conditional means for current phase range
                effect_ndx = ((g.phase[:, lat, lon] >= bins[j]) & (g.phase[:, lat, lon] <= bins[j+1]))
                cond_means_temp[j, 0, lat, lon] = np.mean(g.data[effect_ndx, lat, lon])
                cond_means_temp[j, 1, lat, lon] = np.mean(annual_amp[effect_ndx, lat, lon])
            amp[lat, lon] = cond_means_temp[:, 1, lat, lon].max() - cond_means_temp[:, 1, lat, lon].min()
            effect[lat, lon] = cond_means_temp[:, 0, lat, lon].max() - cond_means_temp[:, 0, lat, lon].min()
            mean_amp[lat, lon] = np.mean(amplitude[:, lat, lon])
            mean_ampAAC[lat, lon] = np.mean(amplitudeAACreg[:, lat, lon])
        else:
            amp[lat, lon] = np.nan
            effect[lat, lon] = np.nan
            mean_amp[lat, lon] = np.nan
            mean_ampAAC[lat, lon] = np.nan


g.quick_render(field_to_plot = mean_ampAAC, whole_world = False, symm = False, tit = "LINEAR: 8yr mean AAC amplitude", 
    fname = "plots/egu17/ECAD-8yr-AAC-amp.png")
g.quick_render(field_to_plot = amp, whole_world = False, symm = False, tit = "NONLINEAR: 8yr cond. bins on AAC",
    fname = "plots/egu17/ECAD-8yr-AAC-cond-bins.png")
g.quick_render(field_to_plot = amp - mean_ampAAC, whole_world = False, symm = False, tit = "NONLINEAR - LINEAR",
    fname = "plots/egu17/ECAD-8yr-AAC-diff.png")
