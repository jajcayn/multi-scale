"""
created on August 27, 2014

@author: Nikola Jajcay
"""

import numpy as np
from src.data_class import load_station_data, DataField
from datetime import date
from src import wavelet_analysis as wvlt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import scipy.stats as sts


def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))
    
    
    
    
PERIOD = 8
WINDOW_LENGTH = 13462 # 13462, 16384
MIDDLE_YEAR = 1965 # year around which the window will be deployed
DIFFS_LIM = [10,25]
SAT = False

# load whole data - load SAT data
g = load_station_data('TG_STAID000027.txt', date(1775, 1, 1), date(2014, 1, 1), False)

# save SAT data
tg_sat = g.copy_data()
# anomalise to obtain SATA data
g.anomalise()


g_temp = DataField()

# starting month and day
sm = 7
sd = 28
# starting year of final window
y = 365.25
sy = int(MIDDLE_YEAR - (WINDOW_LENGTH/y)/2)

# get wvlt window
start = g.find_date_ndx(date(sy - 4, sm, sd))
end = start + 16384 if WINDOW_LENGTH < 16000 else start + 32768
g.data = g.data[start : end]
g.time = g.time[start : end]
tg_sat = tg_sat[start : end]


# wavelet
k0 = 6. # wavenumber of Morlet wavelet used in analysis
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale
wave, _, _, _ = wvlt.continous_wavelet(g.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes

# get final window
idx = g.get_data_of_precise_length(WINDOW_LENGTH, date(sy, sm, sd), None, True)
phase = phase[0, idx[0] : idx[1]]
# get window for non-anomalised data
tg_sat = tg_sat[idx[0] : idx[1]]

phase_bins = get_equidistant_bins()
scaling = []

for i in range(phase_bins.shape[0] - 1):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    if SAT:
        data_temp = tg_sat[ndx].copy()
    else:
        data_temp = g.data[ndx].copy()
    time_temp = g.time[ndx].copy()
    tg_sat_temp = tg_sat[ndx].copy()
    
    scaling_bin = []
    for diff in range(DIFFS_LIM[0],DIFFS_LIM[1]+1):
        difs = []
        for day in range(data_temp.shape[0]-diff):
            if (time_temp[day+diff] - time_temp[day]) == diff:
                difs.append(np.abs(data_temp[day+diff] - data_temp[day]))
        difs = np.array(difs)
        scaling_bin.append(difs)
    scaling.append(scaling_bin) 
    
scaling = np.array(scaling)

# plotting
day_diff = 0
fig = plt.figure(figsize = (16,8), frameon = False)
gs1 = gridspec.GridSpec(1, 8)
gs1.update(left = 0.05, right = 0.95, top = 0.88, bottom = 0.52, wspace = 0.4)
for i in range(8):
    ax = plt.Subplot(fig, gs1[0, i])
    fig.add_subplot(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(top = 'off', right = 'off', which = 'both', color = '#6A4A3C')
    to_draw = []
    for ii in range(scaling.shape[1]):
        for jj in range(scaling[i, ii].shape[0]):
            to_draw.append(scaling[i, ii][jj])
    n = ax.hist(to_draw, bins = 20, range = (0, 25), cumulative = True, normed = True, 
            fc = '#E86E52', ec = '#434552')
    ax.set_xticks(np.linspace(5,30,6))
    ax.set_xbound(5,30)
    ax.set_xlabel('$\Delta$ T [$^{\circ}$C]')
    ax.set_yticks(np.linspace(0.95,1.,6))
    ax.set_yticks(np.linspace(0.95,1.,11), minor = True)
    ax.set_ybound(lower = 0.95, upper = 1.0)
#    ax.set_yscale('log')
    

gs2 = gridspec.GridSpec(1,4)
gs2.update(left = 0.05, right = 0.95, top = 0.41, bottom = 0.08, wspace = 0.3)
#func = [np.amax, np.amin, np.mean, np.median]
func = [np.mean, np.std, sts.skew, sts.kurtosis]
#titl = ['Maximum', 'Minimum', 'Mean', 'Median']
titl = ['Mean', 'STD', 'skewness', 'kurtosis']
for i in range(4):
    ax = plt.Subplot(fig, gs2[0, i])
    fig.add_subplot(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(top = 'off', right = 'off', which = 'both', color = '#6A4A3C')
    c = np.zeros((scaling.shape[0], scaling.shape[1]))
    for ii in range(c.shape[0]):
        for jj in range(c.shape[1]):
            if i != 1:
                c[ii,jj] = func[i](scaling[ii, jj])
            else:
                c[ii,jj] = func[i](scaling[ii, jj], ddof = 1)
    ma, mi = c.max(), c.min()
    if ma-mi > 1:
        vmax, vmin = np.ceil(ma), np.floor(mi)
    elif ma-mi > 0.1:
        vmax, vmin = np.ceil(ma*10)/10., np.floor(mi*10)/10.
    else:
        vmax, vmin = np.ceil(ma*100)/100., np.floor(mi*100)/100.
    p = ax.pcolor(np.linspace(0.5,16.5,17), np.linspace(0.5,8.5,9), c, cmap = plt.cm.RdBu_r, vmin = vmin, vmax = vmax)
    ax.set_xbound(lower = 0.5, upper = 16.5)
    ax.set_ybound(lower=0.5, upper=8.5)
    ax.set_xticks(np.linspace(1,16,16), minor = True)
    ax.set_title(titl[i])
    ax.set_xlabel("$\Delta$ days")
    ax.set_ylabel("bin no.")
    cbar = plt.colorbar(p, ax=ax)
    cbar.set_label("T [$^{\circ}$C]")
fig.text(0.5, 0.92, 'Cumulative normed histograms of scaling', ha = 'center', va = 'center', size = 14)
plt.suptitle('%s - %d point / %s window: %s -- %s' % (g.location, MIDDLE_YEAR, '14k' if WINDOW_LENGTH < 16000 else '16k', 
                  str(g.get_date_from_ndx(0)), str(g.get_date_from_ndx(-1))), size = 16)

plt.savefig('debug/scaling_hist_moments_%s%d_%s_window.png' % ('SAT' if SAT else 'SATA', MIDDLE_YEAR, '14k' if WINDOW_LENGTH < 16000 else '16k'))
