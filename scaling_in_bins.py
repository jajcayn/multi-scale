"""
created on July 3, 2014

@author: Nikola Jajcay
"""

import numpy as np
from src.data_class import load_station_data, DataField
from datetime import date
from src import wavelet_analysis as wvlt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 


def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))
    
    
def render_extremes_and_scaling_in_bins(res, heat_w, cold_w, fname = None):
    fig = plt.figure(figsize = (16,8), frameon = False)
    
    # extremes
    gs1 = gridspec.GridSpec(1, 6)
    gs1.update(left = 0.1, right = 0.95, top = 0.91, bottom = 0.55, wspace = 0.25)
    titles = ['> 2$\cdot\sigma$', '> 3$\cdot\sigma$', 
              '< -2$\cdot\sigma$', '< -3$\cdot\sigma$',
              '5 days > 0.8$\cdot$max T', '5 days < 0.8$\cdot$min T']
    colours = ['#F38630', '#FA6900', '#69D2E7', '#A7DBD8', '#EB6841', '#00A0B0']
    hatches = ['/', '+', 'x', '.']
    labels = ['DJF', 'MAM', 'JJA', 'SON']
    
    for i in range(6):
        ax = plt.Subplot(fig, gs1[0, i])
        fig.add_subplot(ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(top = 'off', right = 'off', color = '#6A4A3C')
        diff = (phase_bins[1]-phase_bins[0])
        if i < 4:
            for j in range(4):
                if j == 0:
                    rects = ax.bar(phase_bins[:-1]+0.1*diff, res[:, 4*i], width = 0.8*diff, bottom = None, fc = colours[i], 
                                    ec = '#6A4A3C', hatch = hatches[j], linewidth = 0.1, label = labels[j])
                else:
                    rects = ax.bar(phase_bins[:-1]+0.1*diff, res[:, 4*i+j], width = 0.8*diff, bottom = np.sum(res[:, 4*i:4*i+j], axis = 1), 
                                    fc = colours[i], ec = '#6A4A3C', hatch = hatches[j], linewidth = 0.1, label = labels[j])
            ax.set_xbound(lower = -np.pi, upper = np.pi)
            if i == 0:
                ax.legend(bbox_to_anchor = (-0.3, 0.72), prop = {'size' : 11})
            maximum = np.sum(res[:, 4*i:4*i+4], axis = 1).argmax()
            ax.text(rects[maximum].get_x() + rects[maximum].get_width()/2., 0, 
                     '%d'%int(np.sum(res[maximum, 4*i:4*i+4])), ha = 'center', va = 'bottom', color = '#6A4A3C')
        else:
            rects = ax.bar(phase_bins[:-1]+0.1*diff, res[:, i+12], width = 0.8*diff, bottom = None, fc = colours[i], ec = colours[i])
            ax.axis([-np.pi, np.pi, 0, res[:, i+12].max() + 1])
            maximum = res[:, i+12].argmax()
            ax.text(rects[maximum].get_x() + rects[maximum].get_width()/2., 0, 
                     '%d'%int(rects[maximum].get_height()), ha = 'center', va = 'bottom', color = '#6A4A3C')
#        if res[:, i].max() < 5:
#            ax.yaxis.set_ticks(np.arange(0, 5, 1))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.set_xlabel('phase [rad]')
        ax.set_title(titles[i])
    fig.text(0.07, 0.75, 'count', ha = 'center', va = 'center', rotation = 'vertical')
    fig.text(0.97, 0.75, 'count', ha = 'center', va = 'center', rotation = -90)
    
    # scaling
    gs2 = gridspec.GridSpec(1, 8)
    gs2.update(left = 0.05, right = 0.95, top = 0.42, bottom = 0.12, wspace = 0.25)
    for i in range(8):
        ax = plt.Subplot(fig, gs2[0, i])
        fig.add_subplot(ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(top = 'off', right = 'off', color = '#6A4A3C')
        ax.tick_params(which = 'minor', top = 'off', right = 'off', color = '#6A4A3C')
        max_d = max(cold_w[i].keys()[-1], heat_w[i].keys()[-1])
        ax.bar(np.arange(3, max_d+1,1)+0.1, [heat_w[i][j] if (j in heat_w[i]) else 0 for j in range(3,max_d+1)], width = 0.8, 
               bottom = None, fc = '#EB6841', ec = '#EB6841')
        ax.bar(np.arange(3, max_d+1,1)+0.1, [-cold_w[i][j] if (j in cold_w[i]) else 0 for j in range(3,max_d+1)], width = 0.8, 
               bottom = None, fc = '#00A0B0', ec = '#00A0B0')
        bound = max(np.array([cold_w[k][j] for k in range(8) for j in cold_w[k].keys() if j >= 3]).max(), np.array([heat_w[k][j] for k in range(8) for j in heat_w[k].keys() if j >= 3]).max())
        ax.axis([3, 25, -bound, bound])
#        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel('wave duration [days]')
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        pos = gs2[0,i].get_position(fig).get_points()
        fig.text(np.mean(pos[:,0]), 0.45, '(%.2f, %.2f)' % (phase_bins[i], phase_bins[i+1]), ha = 'center', va = 'center')
    fig.text(0.015, 0.27, 'number of occurence \n cold | heat', ha = 'center', va = 'center', rotation = 'vertical')
    fig.text(0.965, 0.277, 'number of occurence \n heat | cold', ha = 'center', va = 'center', rotation = -90)
    fig.text(0.5, 0.02, 'heat/cold waves (80percentil) with duration at least 3 days', ha = 'center', va = 'center', size = 16)    
    plt.suptitle('%s - %d point / %s window: %s -- %s' % (g.location, MIDDLE_YEAR, '14k' if WINDOW_LENGTH < 16000 else '16k', 
                  str(g.get_date_from_ndx(0)), str(g.get_date_from_ndx(-1))), size = 16)
    if fname != None:
        plt.savefig(fname)
    else:
        plt.show()
        
        
        
def render_scaling_min_max(scaling, min_scaling, max_scaling, fname = None):
    fig = plt.figure(figsize = (14,9), frameon = False)
    
    colours = ["#E3D2B4", "#AC9F7F", "#ACECC9", 
               "#FFBF17", "#FF4F01", "#F32645", 
               "#C42366", "#A92477"]
    
    gs = gridspec.GridSpec(1, 3)
    gs.update(left = 0.05, right = 0.95, top = 0.85, bottom = 0.3, wspace = 0.2)
    ax1 = plt.Subplot(fig, gs[0,1])
    fig.add_subplot(ax1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.tick_params(top = 'off', right = 'off', color = '#6A4A3C')
    ax1.tick_params(which = 'minor', top = 'off', right = 'off', color = '#6A4A3C')
    lab = {}
    for i in range(scaling.shape[0]):
        lab[i], = ax1.loglog(scaling[i, 0:], linewidth = 0.75, color = colours[i])
#    ax1.axis([0, scaling.shape[1], 0, 6])
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.set_ylim(ymax = 6)
    ax1.set_xlim(xmax = 80)
    ax1.set_xlabel('log $\Delta$time [days]')
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax1.set_title("scaling mean temperature TG")
    fig.legend([lab[i] for i in range(scaling.shape[0])], 
                ["bin %d: %.2f - %.2f" % (i+1, phase_bins[i], phase_bins[i+1]) for i in range(scaling.shape[0])], 
                 loc = 'lower center')
    
    ax2 = plt.Subplot(fig, gs[0,0])
    fig.add_subplot(ax2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(top = 'off', right = 'off', color = '#6A4A3C')
    ax2.tick_params(which = 'minor', top = 'off', right = 'off', color = '#6A4A3C')
    for i in range(scaling.shape[0]):
        ax2.loglog(min_scaling[i, 0:], linewidth = 0.75, color = colours[i])
#    ax2.axis([0, scaling.shape[1], 0, 6])
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.set_ylim(ymax = 6)
    ax2.set_xlim(xmax = 80)
    ax2.set_xlabel('log $\Delta$time [days]')
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax2.set_title("scaling min temperature TN")
    
    ax3 = plt.Subplot(fig, gs[0,2])
    fig.add_subplot(ax3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.tick_params(top = 'off', right = 'off', color = '#6A4A3C')
    ax3.tick_params(which = 'minor', top = 'off', right = 'off', color = '#6A4A3C')
    for i in range(scaling.shape[0]):
        ax3.loglog(max_scaling[i, 0:], linewidth = 0.75, color = colours[i])
#    ax3.axis([0, scaling.shape[1], 0, 6])
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax3.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3.set_ylim(ymax = 6)
    ax3.set_xlim(xmax = 80)
    ax3.set_xlabel('log $\Delta$time [days]')
    ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax3.set_title("scaling max temperature TX")
    
    fig.text(0.02, 0.575, 'log $\Delta$difference [$^{\circ}$C]', ha = 'center', va = 'center', rotation = 'vertical')
    fig.text(0.97, 0.575, 'log $\Delta$difference [$^{\circ}$C]', ha = 'center', va = 'center', rotation = -90)
    
    plt.suptitle('%s - %d point / %s window: %s -- %s' % (g.location, MIDDLE_YEAR, '14k' if WINDOW_LENGTH < 16000 else '16k', 
                  str(g.get_date_from_ndx(0)), str(g.get_date_from_ndx(-1))), size = 16)
                  
    if fname != None:
        plt.savefig(fname)
    else:
        plt.show()
    


PERIOD = 8
WINDOW_LENGTH = 13462 # 13462, 16384
MIDDLE_YEAR = 1965 # year around which the window will be deployed
JUST_SCALING = False
PLOT = True


# load whole data - load SAT data
g = load_station_data('TG_STAID000027.txt', date(1775, 1, 1), date(2014, 1, 1), False)
# save SAT data
tg_sat = g.copy_data()
# anomalise to obtain SATA data
g.anomalise()
# the same with TX and TN

g_max = load_station_data('TX_STAID000027.txt', date(1775, 1, 1), date(2014, 1, 1), False)
tx_sat = g_max.copy_data()
g_max.anomalise()

g_min = load_station_data('TN_STAID000027.txt', date(1775, 1, 1), date(2014, 1, 1), False)
tn_sat = g_min.copy_data()
g_min.anomalise()

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
g_max.data = g_max.data[start : end]
g_max.time = g_max.time[start : end]
g_min.data = g_min.data[start : end]
g_min.time = g_min.time[start : end]



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
g_max.data = g_max.data[idx[0] : idx[1]]
g_max.time = g_max.time[idx[0] : idx[1]]
g_min.data = g_min.data[idx[0] : idx[1]]
g_min.time = g_min.time[idx[0] : idx[1]]
sigma_max = np.nanstd(g.data, axis = 0, ddof = 1)
sigma_min = np.nanstd(g.data, axis = 0, ddof = 1)
temp_max = g_max.data.copy()

# get sigma for extremes

# prepare result matrix
result = np.zeros((8, 18)) # bin no. x result no. (DJF, MAM, JJA, SON)

def add_value_dict(dic, key, val = 1):
    if key in dic:
        dic[key] += val
    else:
        dic[key] = val
        
def get_percentil_exceedance(val, set_of_values, percentil, plus_minus = True):
    if plus_minus:
        return np.sum(np.greater(val, set_of_values)) > (percentil/100.)*set_of_values.shape[0]
    else:
        return np.sum(np.less(val, set_of_values)) > (percentil/100.)*set_of_values.shape[0]

            
# binning
phase_bins = get_equidistant_bins()
scaling = []
hw = []
cw = []
if JUST_SCALING:
    scaling_min = []
    scaling_max = []
for i in range(phase_bins.shape[0] - 1):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    data_temp = g.data[ndx].copy()
    time_temp = g.time[ndx].copy()
    max_temp = g_max.data[ndx].copy()
    min_temp = g_min.data[ndx].copy()
    
    if not JUST_SCALING:
        g_temp.time = time_temp.copy()
        _, m, _ = g_temp.extract_day_month_year()
        # positive extremes - 2sigma
        g_e = np.greater_equal(data_temp, 2 * sigma_max)
        result[i, 0] = np.sum((m[g_e] == 12) | (m[g_e] <= 2)) # DJF
        result[i, 1] = np.sum((m[g_e] > 2) & (m[g_e] <= 5)) # MAM
        result[i, 2] = np.sum((m[g_e] > 5) & (m[g_e] <= 8)) # JJA
        result[i, 3] = np.sum((m[g_e] > 8) & (m[g_e] <= 11)) # SON
        # positive extremes - 3sigma
        g_e = np.greater_equal(data_temp, 3 * sigma_max)
        result[i, 4] = np.sum((m[g_e] == 12) | (m[g_e] <= 2)) # DJF
        result[i, 5] = np.sum((m[g_e] > 2) & (m[g_e] <= 5)) # MAM
        result[i, 6] = np.sum((m[g_e] > 5) & (m[g_e] <= 8)) # JJA
        result[i, 7] = np.sum((m[g_e] > 8) & (m[g_e] <= 11)) # SON
        # negative extremes - 2sigma
        l_e = np.less_equal(data_temp, -2 * sigma_min)
        result[i, 8] = np.sum((m[l_e] == 12) | (m[l_e] <= 2)) # DJF
        result[i, 9] = np.sum((m[l_e] > 2) & (m[l_e] <= 5)) # MAM
        result[i, 10] = np.sum((m[l_e] > 5) & (m[l_e] <= 8)) # JJA
        result[i, 11] = np.sum((m[l_e] > 8) & (m[l_e] <= 11)) # SON
        # negative extremes - 3sigma
        l_e = np.less_equal(data_temp, -3 * sigma_min)
        result[i, 12] = np.sum((m[l_e] == 12) | (m[l_e] <= 2)) # DJF
        result[i, 13] = np.sum((m[l_e] > 2) & (m[l_e] <= 5)) # MAM
        result[i, 14] = np.sum((m[l_e] > 5) & (m[l_e] <= 8)) # JJA
        result[i, 15] = np.sum((m[l_e] > 8) & (m[l_e] <= 11)) # SON
        for iota in range(data_temp.shape[0]-5):
            # heat waves
            if np.all(data_temp[iota : iota+5] > 0.8*max_temp[iota : iota+5]):
                result[i, 16] += 1
            # cold waves
            if np.all(data_temp[iota : iota+5] < 0.8*min_temp[iota : iota+5]):
                result[i, 17] += 1
                
        # histo of HW/CW
        iota = 0
        heat_w = {}
        cold_w = {}
        while iota < data_temp.shape[0]:
            if get_percentil_exceedance(data_temp[iota], g_max.data, 80, True):
                lag = 0
                while get_percentil_exceedance(data_temp[iota+lag], g_max.data, 80, True):
                    if time_temp[iota+lag] - time_temp[iota+lag-1] == 1:
                        if iota+lag+1 < data_temp.shape[0]:
                            lag += 1
                        else:
                            break
                    else:
                        iota += 1
                        break
                if lag != 0:
                    add_value_dict(heat_w, lag)
                iota += lag
                
            elif get_percentil_exceedance(data_temp[iota], g_min.data, 80, False):
                lag = 0
                while get_percentil_exceedance(data_temp[iota+lag], g_min.data, 80, False):
                    if time_temp[iota+lag] - time_temp[iota+lag-1] == 1:
                        if iota+lag+1 < data_temp.shape[0]:
                            lag += 1
                        else:
                            break
                    else:
                        iota += 1
                        break
                if lag != 0:
                    add_value_dict(cold_w, lag)
                iota += lag
                
            else:
                iota += 1
            if iota+1 < data_temp.shape[0]:
                continue
            else:
                break
            
        hw.append(heat_w)
        cw.append(cold_w)
            
    #scaling
    if JUST_SCALING:
        scaling_bin = [0]
        for diff in range(1,80):
            difs = []
            for day in range(data_temp.shape[0]-diff):
                if (time_temp[day+diff] - time_temp[day]) == diff:
                    difs.append(np.abs(data_temp[day+diff] - data_temp[day]))
            difs = np.array(difs)
            scaling_bin.append(np.mean(difs))
        scaling.append(scaling_bin)        
        
        scaling_bin = [0]
        for diff in range(1,80):
            difs = []
            for day in range(min_temp.shape[0]-diff):
                if (time_temp[day+diff] - time_temp[day]) == diff:
                    difs.append(np.abs(min_temp[day+diff] - min_temp[day]))
            difs = np.array(difs)
            scaling_bin.append(np.mean(difs))
        scaling_min.append(scaling_bin)
        
        scaling_bin = [0]
        for diff in range(1,80):
            difs = []
            for day in range(max_temp.shape[0]-diff):
                if (time_temp[day+diff] - time_temp[day]) == diff:
                    difs.append(np.abs(max_temp[day+diff] - max_temp[day]))
            difs = np.array(difs)
            scaling_bin.append(np.mean(difs))
        scaling_max.append(scaling_bin)
        

scaling = np.array(scaling)
if JUST_SCALING:
    scaling_min = np.array(scaling_min)
    scaling_max = np.array(scaling_max)

if PLOT:
    if not JUST_SCALING:
        fname = ('debug/scaling_extremes_%d_%sk_window.png' % (MIDDLE_YEAR, '14' if WINDOW_LENGTH < 16000 else '16'))
        render_extremes_and_scaling_in_bins(result, hw, cw, fname)
    else:
        fname = ('debug/scaling_min_max_%d_%sk_window.png' % (MIDDLE_YEAR, '14' if WINDOW_LENGTH < 16000 else '16'))
        render_scaling_min_max(scaling, scaling_min, scaling_max, fname)
            


