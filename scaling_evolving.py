"""
created on August 14, 2014

@author: Nikola Jajcay
"""


import numpy as np
from src.data_class import load_station_data, DataField
from datetime import date
from src import wavelet_analysis as wvlt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker 
import scipy.stats as sst

def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))
    
    
    
PERIOD = 8
PAST_UNTIL = 1930 # from which segment the average extremes should be computed
WINDOW_LENGTH = 16384 # 13462, 16384




# load whole data - load SAT data
g = load_station_data('TG_STAID000027.txt', date(1775, 1, 1), date(2014, 1, 1), False)
g_for_avg = load_station_data('TG_STAID000027.txt', date(1775, 1, 1), date(2014, 1, 1), False)

# save SAT data
tg_sat = g.copy_data()
tg_avg_sat = g_for_avg.copy_data()
# anomalise to obtain SATA data
g.anomalise()
g_for_avg.anomalise()


g_for_avg.select_date(date(1775,1,1), date(PAST_UNTIL, 1, 1))

year = 365.25
# get average extremes
k0 = 6. # wavenumber of Morlet wavelet used in analysis
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * year # frequency of interest
s0 = period / fourier_factor # get scale
wave, _, _, _ = wvlt.continous_wavelet(g_for_avg.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes

avg_ndx = g_for_avg.select_date(date(1779,1,1), date(PAST_UNTIL-4,1,1))
phase = phase[0, avg_ndx]
tg_avg_sat = tg_avg_sat[avg_ndx]

# sigma
sigma = np.std(tg_avg_sat, axis = 0, ddof = 1)

avg_bins = np.zeros((8, 2)) # bin no. x result no. (hot / cold extremes)

phase_bins = get_equidistant_bins()

for i in range(phase_bins.shape[0] - 1):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    data_temp = g_for_avg.data[ndx].copy()
    time_temp = g_for_avg.time[ndx].copy()
    tg_sat_temp = tg_avg_sat[ndx].copy()
    
    # positive extremes
    g_e = np.greater_equal(tg_sat_temp, np.mean(tg_avg_sat, axis = 0) + 2 * sigma)
    avg_bins[i, 0] = np.sum(g_e)
    
    # negative extremes
    l_e = np.less_equal(tg_sat_temp, np.mean(tg_avg_sat, axis = 0) - 2 * sigma)
    avg_bins[i, 1] = np.sum(l_e)


sm = 7
sd = 28
evolve = []
# evolving
for MIDDLE_YEAR in range(1802, 1988):
    g_temp = DataField()
    tg_temp = tg_sat.copy()
    sy = int(MIDDLE_YEAR - (WINDOW_LENGTH/year)/2)
    g_temp.data = g.data.copy()
    g_temp.time = g.time.copy()
    start = g_temp.find_date_ndx(date(sy - 4, sm, sd))
    end = start + 16384 if WINDOW_LENGTH < 16000 else start + 32768
   
    g_temp.data = g_temp.data[start : end]
    g_temp.time = g_temp.time[start : end]
    tg_temp = tg_temp[start : end]
    
    k0 = 6. # wavenumber of Morlet wavelet used in analysis
    fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
    period = PERIOD * year # frequency of interest
    s0 = period / fourier_factor # get scale
    wave, _, _, _ = wvlt.continous_wavelet(g_temp.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
    phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes
    
    idx = g_temp.get_data_of_precise_length(WINDOW_LENGTH, date(sy, sm, sd), None, True)
    phase = phase[0, idx[0] : idx[1]]
    tg_temp = tg_temp[idx[0] : idx[1]]
    
    sigma = np.std(tg_temp, axis = 0, ddof = 1)
    
    result_temp = np.zeros((8,2))
    
    for i in range(phase_bins.shape[0] - 1):
        ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
        data_temp = g_temp.data[ndx].copy()
        time_temp = g_temp.time[ndx].copy()
        tg_sat_temp = tg_temp[ndx].copy()
        
        # positive extremes
        g_e = np.greater_equal(tg_sat_temp, np.mean(tg_temp, axis = 0) + 2 * sigma)
        result_temp[i, 0] = np.sum(g_e)
        
        # negative extremes
        l_e = np.less_equal(tg_sat_temp, np.mean(tg_temp, axis = 0) - 2 * sigma)
        result_temp[i, 1] = np.sum(l_e)
    
    evolve.append(result_temp)
    
# plotting
x = np.linspace(0., np.pi, 8)
y = np.sin(x)
hot = []
cold = []
hot_sp = []
cold_sp = []
hsin = []
csin = []
hsin_sp = []
csin_sp = []
for bar in evolve:
    hot.append(np.corrcoef(avg_bins[:, 0], bar[:, 0])[0, 1])
    cold.append(np.corrcoef(avg_bins[:, 1], bar[:, 1])[0, 1])
    
    hot_sp.append(sst.spearmanr(avg_bins[:, 0], bar[:, 0])[0])
    cold_sp.append(sst.spearmanr(avg_bins[:, 1], bar[:, 1])[0])
    
    hsin.append(np.corrcoef(y, bar[:, 0])[0, 1])
    csin.append(np.corrcoef(-y, bar[:, 1])[0, 1])
    
    hsin_sp.append(sst.spearmanr(y, bar[:, 0])[0])
    csin_sp.append(sst.spearmanr(-y, bar[:, 1])[0])
    
hot = np.array(hot)
cold = np.array(cold)
hot_sp = np.array(hot_sp)
cold_sp = np.array(cold_sp)
hsin = np.array(hsin)
csin = np.array(csin)
hsin_sp = np.array(hsin_sp)
csin_sp = np.array(csin_sp)


fig = plt.figure(figsize = (16,8), frameon = False)
gs = gridspec.GridSpec(2, 3, width_ratios = [5,1,1])
gs.update(left = 0.05, right = 0.95, top = 0.9, bottom = 0.1, wspace = 0.25, hspace = 0.4)
colours = ['#F38630', '#69D2E7']
colours_sp = ['#FEF215', '#6A009D']
colours_sin_sp = ['#FAD900', '#431341']
colours_sin = ['#91842C', '#22AC27']
plots = [hot, cold]
plots_sp = [hot_sp, cold_sp]
sins = [hsin, csin]
sins_sp = [hsin_sp, csin_sp]
titles = ['hot extremes  >2$\sigma$', 'cold extremes  <-2$\sigma$']
plt.suptitle('Evolving of correlation of extremes barplots with average 1775-%d, %s window' % (PAST_UNTIL, '16k' if WINDOW_LENGTH > 16000 else '14k'), size = 16)
for i in range(2):
    # evolving
    ax = plt.Subplot(fig, gs[i, 0])
    fig.add_subplot(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color = '#6A4A3C')
    ax.plot(plots[i], color = colours[i], linewidth = 2, label = 'Pearson')
    ax.plot(plots_sp[i], color = colours_sp[i], linewidth = 2, label = 'Spearman')
    ax.plot(sins[i], "--", color = colours_sin[i], linewidth = 1.5, label = 'Pearson sin')
    ax.plot(sins_sp[i], "--", color = colours_sin_sp[i], linewidth = 1.5, label = 'Spearman sin')
    ax.legend(loc = 3, prop = {'size' : 10}, ncol = 2)
    ax.set_ylabel('correlation with past average')
    ax.set_xlabel('middle year of %.2f-year window' % (WINDOW_LENGTH/year))
    ax.set_xlim(0, plots[i].shape[0])
    ax.set_ylim(-1,1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.set_xticks(np.arange(0, plots[i].shape[0]+10, 12), minor = False)
    ax.set_xticks(np.arange(0, plots[i].shape[0]+4, 3), minor = True)
    ax.set_xticklabels(np.arange(1802,1998,12))
    ax.set_title(titles[i])
    
    # average bins past
    ax = plt.Subplot(fig, gs[i, 1])
    fig.add_subplot(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color = '#6A4A3C')
    diff = (phase_bins[1]-phase_bins[0])
    rects = ax.bar(phase_bins[:-1]+0.1*diff, avg_bins[:, i], width = 0.8*diff, bottom = None, fc = colours[i], ec = colours[i])
    maximum = avg_bins[:, i].argmax()
    ax.text(rects[maximum].get_x() + rects[maximum].get_width()/2., 0, 
             '%d'%int(rects[maximum].get_height()), ha = 'center', va = 'bottom', color = '#6A4A3C')
    if i == 0:
        ax.plot(phase_bins[:-1]+0.5*diff, avg_bins[:, i].min()*y+avg_bins[:, i].min(), color = '#FEF215', linewidth = 1.5)
    elif i == 1:
        ax.plot(phase_bins[:-1]+0.5*diff, -avg_bins[:, i].min()*y+2*avg_bins[:, i].min(), color = '#6A009D', linewidth = 1.5)
    ax.axis([-np.pi, np.pi, 0, avg_bins[:, i].max() + 1])
    ax.tick_params(top = 'off', right = 'off', color = '#6A4A3C')
    ax.set_xlabel('phase [rad]')
    
    # typical histo
    ax = plt.Subplot(fig, gs[i, 2])
    fig.add_subplot(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color = '#6A4A3C')
    ax.tick_params(top = 'off', right = 'off', color = '#6A4A3C')
    for bar in evolve:
        rects = ax.bar(phase_bins[:-1]+0.1*diff, bar[:, i], width = 0.8*diff, bottom = None, fc = colours[i], ec = colours[i], alpha = 0.15)
    ax.axis([-np.pi, np.pi, 0, avg_bins[:, i].max() + 1])
    ax.set_xlabel('phase [rad]')
    
fig.text(0.72, 0.47, 'average extremes \n 1775-%d' % PAST_UNTIL, va = 'center', ha = 'center', size = 13, weight = 'heavy')    
fig.text(0.9, 0.47, 'collage of bar plots', va = 'center', ha = 'center', size = 13, weight = 'heavy')    
plt.savefig('debug/extremes_evolving_%d_%s_window.png' % (PAST_UNTIL, '16k' if WINDOW_LENGTH > 16000 else '14k'))

    
    
    