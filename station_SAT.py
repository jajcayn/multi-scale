"""
created on September 5, 2014

@author: Nikola Jajcay
"""

import numpy as np
from src.data_class import load_station_data, load_sunspot_data
from datetime import date
from src import wavelet_analysis as wvlt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import argrelextrema


def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))


PLOT = True
SEASON = 'JJA'
PERCENTIL = 80


g = load_station_data('TG_STAID000027.txt', date(1775, 1, 1), date(2014, 1, 1), False)
g_amp = load_station_data('TG_STAID000027.txt', date(1775, 1, 1), date(2014, 1, 1), False)
# g_sun = load_sunspot_data('monthssn.dat', date(1775, 1, 1), date(2014, 1, 1), False)
tg = g.copy_data()
g.anomalise()

# tg is SAT, g.data is SATA

k0 = 6. # wavenumber of Morlet wavelet used in analysis
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = 8 * 365.25 # frequency of interest
s0 = period / fourier_factor # get scale
wave, _, _, _ = wvlt.continous_wavelet(g.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes

period = 1 * 365.25 # frequency of interest
s0 = period / fourier_factor # get scale
wave, _, _, _ = wvlt.continous_wavelet(g_amp.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))

# cut because of wavelet
wvlt_cut = g.select_date(date(1779,8,1), date(2009,11,1)) # first and last incomplete winters are omitted
tg = tg[wvlt_cut]
phase = phase[0, wvlt_cut]
amplitude = amplitude[0, wvlt_cut]

# if extremes from total sigma / mean not just DJF
sigma_djf = np.std(tg, axis = 0, ddof = 1)
mean_djf = np.mean(tg, axis = 0)

months_ndx = [12,1,2] if SEASON == 'DJF' else [6,7,8]
djf_ndx = g.select_months(months_ndx)
# g_sun.select_months(months_ndx)
tg = tg[djf_ndx]
phase = phase[djf_ndx]
amplitude = amplitude[djf_ndx]


d, m, y = g.extract_day_month_year()
# _, m_s, y_s = g_sun.extract_day_month_year()

phase_bins = get_equidistant_bins()

djf_mean_std = np.zeros((8, 2))

for i in range(phase_bins.shape[0] - 1):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    djf_mean_std[i, 0] = np.mean(tg[ndx], axis = 0)
    djf_mean_std[i, 1] = np.std(tg[ndx], axis = 0, ddof = 1)
    
#plt.bar(phase_bins[:-1], djf_mean_std[:, 0], yerr = djf_mean_std[:, 1])
#plt.show()
    
#sigma_djf = np.std(tg, axis = 0, ddof = 1)
#mean_djf = np.mean(tg, axis = 0)
season_avg = []
phase_avg = []
amplitude_avg = []
sunspot_avg = []
extremes = []
extr_ttl = []
season = 1779
while season < 2009: # 2009
    if SEASON == 'DJF':
        this_djf = filter(lambda i: (m[i] == 12 and y[i] == season) or (m[i] < 3 and y[i] == season + 1), range(tg.shape[0]))
        # this_djf_sspot = filter(lambda i: (m_s[i] == 12 and y_s[i] == season) or (m_s[i] < 3 and y_s[i] == season + 1), range(g_sun.data.shape[0]))
        this_year_extr = filter(lambda i: (m[i] == 12 and y[i] == season) or (m[i] < 12 and y[i] == season + 1), range(tg.shape[0])) 
    elif SEASON == 'JJA':
        this_djf = filter(lambda i: (m[i] > 5 and m[i] < 9 and y[i] == season), range(tg.shape[0]))
        # this_djf_sspot = filter(lambda i: (m_s[i] > 5 and m_s[i] < 9 and y_s[i] == season), range(g_sun.data.shape[0]))    
        this_year_extr = filter(lambda i: y[i] == season, range(tg.shape[0]))
    
    season_avg.append([season, np.mean(tg[this_djf], axis = 0), np.std(tg[this_djf], axis = 0, ddof = 1)])
    phase_avg.append([phase[this_djf].min(), phase[this_djf].max(), np.mean(phase[this_djf]), np.median(phase[this_djf])])
    # sunspot_avg.append([np.mean(g_sun.data[this_djf_sspot], axis = 0), np.std(g_sun.data[this_djf_sspot], axis = 0, ddof = 1)])
    amplitude_avg.append([np.mean(amplitude[this_djf], axis = 0), np.std(amplitude[this_djf], axis = 0, ddof = 1)])
    
    sigma_season = np.std(tg[this_year_extr], axis = 0, ddof = 1)
    if SEASON == 'DJF':
        sigma_2ex = np.less_equal(tg[this_djf], np.mean(tg[this_year_extr]) - 2*sigma_season)
        sigma_3ex = np.less_equal(tg[this_djf], np.mean(tg[this_year_extr]) - 3*sigma_season)
    elif SEASON == 'JJA':
        sigma_2ex = np.greater_equal(tg[this_djf], np.mean(tg[this_year_extr]) + 2*sigma_season)
        sigma_3ex = np.greater_equal(tg[this_djf], np.mean(tg[this_year_extr]) + 3*sigma_season)
    extremes.append([np.sum(sigma_2ex), np.sum(sigma_3ex)])
    
    if SEASON == 'DJF':
        sigma_2ex = np.less_equal(tg[this_djf], mean_djf - 2*sigma_djf)
        sigma_3ex = np.less_equal(tg[this_djf], mean_djf - 3*sigma_djf)
    elif SEASON == 'JJA':
        sigma_2ex = np.greater_equal(tg[this_djf], mean_djf + 2*sigma_djf)
        sigma_3ex = np.greater_equal(tg[this_djf], mean_djf + 3*sigma_djf)
    extr_ttl.append([np.sum(sigma_2ex), np.sum(sigma_3ex)])
    
    season += 1
    
season_avg = np.array(season_avg)
phase_avg = np.array(phase_avg)
sunspot_avg = np.array(sunspot_avg)
amplitude_avg = np.array(amplitude_avg)
extremes = np.array(extremes)
extr_ttl = np.array(extr_ttl)

# plotting
if PLOT:
    fig = plt.figure(figsize = (16,10), frameon = False)
    gs = gridspec.GridSpec(4, 1, height_ratios = [0.3, 1, 0.3, 0.3])
    gs.update(left = 0.05, right = 0.95, top = 0.9, bottom = 0.1, hspace = 0.35, wspace = 0.25)
            
            
    axs = plt.Subplot(fig, gs[0, 0])
    fig.add_subplot(axs)
    axs.plot(season_avg[:, 0], amplitude_avg[:, 0], color = '#FFE545', linewidth = 2)
    axs.fill_between(season_avg[:, 0], amplitude_avg[:, 0] - amplitude_avg[:, 1], amplitude_avg[:, 0] + amplitude_avg[:, 1], 
                        color = '#FFF091', alpha = 0.8)
    # minima = argrelextrema(sunspot_avg[:, 0], np.less, order = 3)[0]
    # minima = np.append(minima, [31, sunspot_avg.shape[0]-1])
    # for mi in range(minima.shape[0]):
        # axs.plot(season_avg[minima[mi], 0], sunspot_avg[minima[mi], 0], 'o', markersize = 6, color = '#FFE545')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.tick_params(top = 'off', right = 'off', color = '#6A4A3C')
    axs.axis([season_avg[0,0], season_avg[-1,0], 100, 250])
    axs.set_xticks(np.arange(season_avg[0,0], season_avg[-1,0]+5, 15))
    axs.set_title("seasonal amplitude of SAT mean with std")
    axs.set_ylabel("amplitude of SAT")
    
    axe = plt.Subplot(fig, gs[2, 0])
    fig.add_subplot(axe)    
    axe.spines['top'].set_visible(False)
    axe.spines['right'].set_visible(False)
    axe.spines['left'].set_visible(False)
    axe.spines['bottom'].set_visible(False)
    axe.tick_params(right = 'off', color = '#6A4A3C')
    axe.plot(season_avg[:, 0], extr_ttl[:, 0], linewidth = 1.5, color = "#6EB1C2")
    axe.plot(season_avg[:, 0], extr_ttl[:, 1], linewidth = 1.5, color = '#026475')
    if SEASON == 'DJF':
        axe.axis([season_avg[0,0], season_avg[-1,0], 0, 40])
    elif SEASON == 'JJA':
        axe.axis([season_avg[0,0], season_avg[-1,0], 0, 20])
    axe.set_xticks(np.arange(season_avg[0,0], season_avg[-1,0]+5, 15))
    axe.set_title("total extremes (among data %s - %s)" % (str(g.get_date_from_ndx(0)), str(g.get_date_from_ndx(-1))))
#    axe.set_xlabel("season (Dec + Jan & Feb next year)")
    
    axe2 = plt.Subplot(fig, gs[3, 0])
    fig.add_subplot(axe2)    
    axe2.spines['top'].set_visible(False)
    axe2.spines['right'].set_visible(False)
    axe2.spines['left'].set_visible(False)
    axe2.spines['bottom'].set_visible(False)
    axe2.tick_params(right = 'off', color = '#6A4A3C')
    axe2.plot(season_avg[:, 0], extremes[:, 0], linewidth = 1.5, color = "#8F9C28")
    axe2.plot(season_avg[:, 0], extremes[:, 1], linewidth = 1.5, color = '#143425')
    axe2.set_xticks(np.arange(season_avg[0,0], season_avg[-1,0]+5, 15))
    axe2.set_title("seasonal extremes (among data in one season)")
    if SEASON == 'DJF':
        axe2.set_xlabel("season (Dec + Jan & Feb next year)")
        axe2.axis([season_avg[0,0], season_avg[-1,0], 0, 10])
    elif SEASON == 'JJA':
        axe2.set_xlabel("season (Jun, Jul, Aug)")
        axe2.axis([season_avg[0,0], season_avg[-1,0], 0, 8])
    
    
    ax = plt.Subplot(fig, gs[1, 0])
    fig.add_subplot(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(right = 'off', color = '#6A4A3C')
    ax.plot(season_avg[:, 0], season_avg[:, 1], color = '#7F417C', linewidth = 2)
    ax.fill_between(season_avg[:, 0], season_avg[:, 1] + season_avg[:, 2], season_avg[:, 1] - season_avg[:, 2], color = '#93739E', alpha = 0.8)
    if SEASON == 'DJF':
        ax.axis([season_avg[0,0], season_avg[-1,0], -15, 10])
    elif SEASON == 'JJA':
        ax.axis([season_avg[0,0], season_avg[-1,0], 10, 30])
    ax.set_ylabel("seasonal SAT mean with std [$^{\circ}$C]")
#    ax.set_xlabel("season (Dec + Jan & Feb next year)")
    #ax.set_title("DJF means of SAT temperature")
    ax.set_xticks(np.arange(season_avg[0,0], season_avg[-1,0]+5, 15))
    
    spheres_cnt = np.zeros((2,))
    in_phase = []
    nearly_in_phase = []
    ax2 = ax.twinx()
    cnt = 0
    dist_of_in_phase = np.zeros((3,))
    dist_of_nearly_in_phase = np.zeros((3,))
    ax2.axis([season_avg[0,0], season_avg[-1,0], -np.pi, 9*np.pi])
    for i in range(season_avg.shape[0]):
        if SEASON == 'JJA':
            if (phase_avg[i, 3] <= phase_bins[5]) and (phase_avg[i, 3] >= phase_bins[3]):
                ax2.plot(season_avg[i, 0], phase_avg[i, 3], 'o', markersize = 8, color = '#C5D76C')
                perc = np.greater(season_avg[i, 1], season_avg[:, 1])
                if np.sum(perc) >= season_avg.shape[0]*(PERCENTIL/100.):
                    ax.plot(season_avg[i, 0], season_avg[i, 1], 'o', markersize = 8, color = '#8F9C28')
                else:
                    ax.plot(season_avg[i, 0], season_avg[i, 1], 'o', markersize = 8, color = '#7F417C')
                axe.plot(season_avg[i, 0], extr_ttl[i, 0], 'o', markersize = 8, color = '#6EB1C2')
                axe2.plot(season_avg[i, 0], extremes[i, 0], 'o', markersize = 8, color = '#8F9C28')
                in_phase.append([season_avg[i, 1], extr_ttl[i, 0], extremes[i, 0]])
            elif (phase_avg[i, 3] <= phase_bins[3] and phase_avg[i, 3] >= phase_bins[2]) or (phase_avg[i, 3] >= phase_bins[5] and phase_avg[i, 3] <= phase_bins[6]):
                ax2.plot(season_avg[i, 0], phase_avg[i, 3], 'o', markersize = 6, color = '#4AD8C4')
                ax.plot(season_avg[i, 0], season_avg[i, 1], 'o', markersize = 6, color = '#7F417C')
                axe.plot(season_avg[i, 0], extr_ttl[i, 0], 'o', markersize = 6, color = '#6EB1C2')
                axe2.plot(season_avg[i, 0], extremes[i, 0], 'o', markersize = 6, color = '#8F9C28')
                nearly_in_phase.append([season_avg[i, 1], extr_ttl[i, 0], extremes[i, 0]])
            else:
                ax2.plot(season_avg[i, 0], phase_avg[i, 3], 'o', markersize = 3, color = '#000000')
        elif SEASON == 'DJF':
            if (phase_avg[i, 3] <= phase_bins[1]) or (phase_avg[i, 3] >= phase_bins[7]):
                ax2.plot(season_avg[i, 0], phase_avg[i, 3], 'o', markersize = 8, color = '#C5D76C')
                perc = np.less(season_avg[i, 1], season_avg[:, 1])
                # percentil from below
                if np.sum(perc) >= season_avg.shape[0]*(PERCENTIL/100.):
                    ax.plot(season_avg[i, 0], season_avg[i, 1], 'o', markersize = 8, color = '#6EB1C2')
                    dist_of_in_phase[0] += 1
                # percentil from top
                elif np.sum(perc) <= season_avg.shape[0]*((100-PERCENTIL)/100.):
                    dist_of_in_phase[2] += 1
                else:
                    ax.plot(season_avg[i, 0], season_avg[i, 1], 'o', markersize = 8, color = '#7F417C')
                    dist_of_in_phase[1] += 1
                axe.plot(season_avg[i, 0], extr_ttl[i, 0], 'o', markersize = 8, color = '#6EB1C2')
                axe2.plot(season_avg[i, 0], extremes[i, 0], 'o', markersize = 8, color = '#8F9C28')
                in_phase.append([season_avg[i, 1], extr_ttl[i, 0], extremes[i, 0]])
                spheres_cnt[0] += 1
            elif (phase_avg[i, 3] <= phase_bins[2]) or (phase_avg[i, 3] >= phase_bins[6]):
                perc = np.less(season_avg[i, 1], season_avg[:, 1])
                if np.sum(perc) >= season_avg.shape[0]*(PERCENTIL/100.):
                    dist_of_nearly_in_phase[0] += 1
                elif np.sum(perc) <= season_avg.shape[0]*((100-PERCENTIL)/100.):
                    dist_of_nearly_in_phase[2] += 1
                else:
                    dist_of_nearly_in_phase[1] += 1
                ax2.plot(season_avg[i, 0], phase_avg[i, 3], 'o', markersize = 6, color = '#4AD8C4')
                ax.plot(season_avg[i, 0], season_avg[i, 1], 'o', markersize = 6, color = '#7F417C')
                axe.plot(season_avg[i, 0], extr_ttl[i, 0], 'o', markersize = 6, color = '#6EB1C2')
                axe2.plot(season_avg[i, 0], extremes[i, 0], 'o', markersize = 6, color = '#8F9C28')
                nearly_in_phase.append([season_avg[i, 1], extr_ttl[i, 0], extremes[i, 0]])
                spheres_cnt[1] += 1
            else:
                ax2.plot(season_avg[i, 0], phase_avg[i, 3], 'o', markersize = 3, color = '#000000')
        perc = np.less(season_avg[i,1], season_avg[:,1])
        if np.sum(perc) >= season_avg.shape[0]*(PERCENTIL/100.):
            cnt += 1
        
                
    in_phase = np.array(in_phase)
    nearly_in_phase = np.array(nearly_in_phase)
            
    ax2.yaxis.set_ticks(np.linspace(-np.pi, np.pi, 5))
    
    
    print PERCENTIL, dist_of_in_phase, spheres_cnt[0]
    print PERCENTIL, dist_of_nearly_in_phase, spheres_cnt[1]
    print cnt

    
    
    plt.suptitle("%s -- %s SAT analysis" % (g.location, SEASON), size = 20)
    plt.savefig('debug/Praha_SAT_%s_%dperc.png' % (SEASON.lower(), PERCENTIL))
    