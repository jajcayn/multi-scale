"""
created on September 5, 2014

@author: Nikola Jajcay
"""

import numpy as np
from src.data_class import load_station_data
from datetime import date
from src import wavelet_analysis as wvlt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))



g = load_station_data('TG_STAID000027.txt', date(1775, 1, 1), date(2014, 1, 1), False)
tg = g.copy_data()
g.anomalise()

# tg is SAT, g.data is SATA

k0 = 6. # wavenumber of Morlet wavelet used in analysis
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = 8 * 365.25 # frequency of interest
s0 = period / fourier_factor # get scale
wave, _, _, _ = wvlt.continous_wavelet(g.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes

# cut because of wavelet
wvlt_cut = g.select_date(date(1779,8,1), date(2009,11,1)) # first and last incomplete winters are omitted
tg = tg[wvlt_cut]
phase = phase[0, wvlt_cut]

djf_ndx = g.select_months([12,1,2])
tg = tg[djf_ndx]
phase = phase[djf_ndx]

d, m, y = g.extract_day_month_year()

phase_bins = get_equidistant_bins()

djf_mean_std = np.zeros((8, 2))

for i in range(phase_bins.shape[0] - 1):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    djf_mean_std[i, 0] = np.mean(tg[ndx], axis = 0)
    djf_mean_std[i, 1] = np.std(tg[ndx], axis = 0, ddof = 1)
    
#plt.bar(phase_bins[:-1], djf_mean_std[:, 0], yerr = djf_mean_std[:, 1])
#plt.show()
    
season_avg = []
phase_avg = []
season = 1779
while season < 2009: # 2009
    this_djf = filter(lambda i: (m[i] == 12 and y[i] == season) or (m[i] < 3 and y[i] == season + 1), range(tg.shape[0]))
    season_avg.append([season, np.mean(tg[this_djf], axis = 0), np.std(tg[this_djf], axis = 0, ddof = 1)])
    phase_avg.append([phase[this_djf].min(), phase[this_djf].max(), np.mean(phase[this_djf]), np.median(phase[this_djf])])
    season += 1
    
season_avg = np.array(season_avg)
phase_avg = np.array(phase_avg)

# plotting
fig = plt.figure(figsize = (16,8), frameon = False)
#gs = gridspec.GridSpec(1, 2, width_ratios = [0.5, 1])
#gs.update(left = 0.05, right = 0.95, top = 0.9, bottom = 0.1, wspace = 0.25)

#ax = plt.Subplot(fig, gs[0, 0])
#fig.add_subplot(ax)
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#ax.spines['left'].set_visible(False)
#ax.tick_params(top = 'off', right = 'off', color = '#6A4A3C')
#diff = (phase_bins[1]-phase_bins[0])
#ax.bar(phase_bins[:-1]+0.1*diff, djf_mean_std[:, 0], width = 0.8*diff, fc = '#C5D76C', ec = '#C5D76C', 
#       ecolor = '#4AD8C4', yerr = djf_mean_std[:, 1])
#ax.axis([-np.pi, np.pi, -6, 6])
#ax.set_ylabel("conditional means SAT [$^{\circ}$C]")
#ax.set_xlabel("phase [rad]")
#ax.set_title("Conditional means and std SAT -- DJF only")

#ax = plt.Subplot(fig, gs[0, 1])
ax = plt.subplot(111)
fig.add_subplot(ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(top = 'off', right = 'off', color = '#6A4A3C')
ax.plot(season_avg[:, 0], season_avg[:, 1], color = '#7F417C', linewidth = 2)
ax.fill_between(season_avg[:, 0], season_avg[:, 1] + season_avg[:, 2], season_avg[:, 1] - season_avg[:, 2], color = '#93739E', alpha = 0.8)
ax.axis([season_avg[0,0], season_avg[-1,0], -15, 10])
ax.set_ylabel("seasonal SAT mean with std [$^{\circ}$C]")
ax.set_xlabel("season (Dec + Jan & Feb next year)")
#ax.set_title("DJF means of SAT temperature")
ax.set_xticks(np.arange(season_avg[0,0], season_avg[-1,0]+5, 15))

ax2 = ax.twinx()
ax2.axis([season_avg[0,0], season_avg[-1,0], -np.pi, 9*np.pi])
for i in range(season_avg.shape[0]):
    if (phase_avg[i, 3] <= phase_bins[1]) or (phase_avg[i, 3] >= phase_bins[7]):
        ax2.plot(season_avg[i, 0], phase_avg[i, 3], 'o', markersize = 8, color = '#C5D76C')
        ax.plot(season_avg[i, 0], season_avg[i, 1], 'o', markersize = 8, color = '#7F417C')
    elif (phase_avg[i, 3] <= phase_bins[2]) or (phase_avg[i, 3] >= phase_bins[6]):
        ax2.plot(season_avg[i, 0], phase_avg[i, 3], 'o', markersize = 6, color = '#4AD8C4')
        ax.plot(season_avg[i, 0], season_avg[i, 1], 'o', markersize = 6, color = '#7F417C')
    else:
        ax2.plot(season_avg[i, 0], phase_avg[i, 3], 'o', markersize = 3, color = '#000000')
        
ax2.yaxis.set_ticks(np.linspace(-np.pi, np.pi, 5))
fig.text(0.97, 0.2, 'seasonal median of phase [rad]', ha = 'center', va = 'center', rotation = -90)
plt.suptitle("%s -- DJF SAT analysis" % (g.location), size = 20)
plt.savefig('debug/Praha_SAT_djf.png')
    