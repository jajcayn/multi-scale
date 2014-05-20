"""
created on May 20, 2014

@author: Nikola Jajcay
"""

from src import wavelet_analysis
from src.data_class import load_station_data
from surrogates.surrogates import SurrogateField
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue


def render(diffs, meanvars, stds = None, subtit = '', fname = None):
    fig, ax1 = plt.subplots(figsize=(11,8))
    if len(diffs) > 3:
        ax1.plot(diffs, color = '#403A37', linewidth = 2, figure = fig)
    else:
        p2, = ax1.plot(diffs[1], color = '#899591', linewidth = 1.5, figure = fig)
        if stds is not None:
            ax1.plot(diffs[1] + stds[0], color = '#899591', linewidth = 0.7, figure = fig)
            ax1.plot(diffs[1] - stds[0], color = '#899591', linewidth = 0.7, figure = fig)
            ax1.fill_between(np.arange(0,diffs[1].shape[0],1), diffs[1] + stds[0], diffs[1] - stds[0], 
                             facecolor = "#899591", alpha = 0.5)
        p1, = ax1.plot(diffs[0], color = '#403A37', linewidth = 2, figure = fig)
    #ax1.plot(total_diffs[0], np.arange(0,len(total_diffs[0])), total_diffs[1], np.arange(0, cnt))
    if not ANOMALISE and MEANS:
        ax1.axis([0, cnt-1, 0, 3])
    if not ANOMALISE and not MEANS:
        ax1.axis([0, cnt-1, 0, 25])
    if ANOMALISE and MEANS:
        ax1.axis([0, cnt-1, 0, 2])
    if ANOMALISE and not MEANS:
        ax1.axis([0, cnt-1, 0, 6])
    if np.int(WINDOW_LENGTH) == WINDOW_LENGTH:
        ax1.set_xlabel('start year of %d-year wide window' % WINDOW_LENGTH, size = 14)
    else:
        ax1.set_xlabel('start year of %.2f-year wide window' % WINDOW_LENGTH, size = 14)
    if MEANS:
        ax1.set_ylabel('difference in cond mean in temperature [$^{\circ}$C]', size = 14)
    elif not MEANS:
        ax1.set_ylabel('difference in cond variance in temperature [$^{\circ}$C$^2$]', size = 14)
    plt.xticks(np.arange(0,cnt,15), np.arange(start_date.year, end_date.year, 15), rotation = 30)
    ax2 = ax1.twinx()
    if len(meanvars) > 3:
        ax2.plot(meanvars, color = '#CA4F17', linewidth = 2, figure = fig) # color = '#CA4F17'
    else:
        p4, = ax2.plot(meanvars[1], color = '#64C4A0', linewidth = 1.5, figure = fig)
        if stds is not None:
            ax2.plot(meanvars[1] + stds[1], color = '#64C4A0', linewidth = 0.7, figure = fig)
            ax2.plot(meanvars[1] - stds[1], color = '#64C4A0', linewidth = 0.7, figure = fig)
            ax2.fill_between(np.arange(0,diffs[1].shape[0],1), meanvars[1] + stds[1], meanvars[1] - stds[1],
                             facecolor = "#64C4A0", alpha = 0.5)
        p3, = ax2.plot(meanvars[0], color = '#CA4F17', linewidth = 2, figure = fig)
    if MEANS:
        ax2.set_ylabel('mean of cond means in temperature [$^{\circ}$C]', size = 14)
    elif not MEANS:
        ax2.set_ylabel('mean of cond variance in temperature [$^{\circ}$C$^2$]', size = 14)
    ax2.axis([0, cnt-1, -1, 1.5])
    for tl in ax2.get_yticklabels():
        tl.set_color('#CA4F17')
    if len(diffs) < 3:
        plt.legend([p1, p2, p3, p4], ["difference DATA", "difference SURROGATE mean", "mean DATA", "mean SURROGATE mean"])
    tit = 'SURR: Evolution of difference in cond'
    if MEANS:
        tit += ' mean in temp, '
    else:
        tit += ' variance in temp, '
    if not ANOMALISE:
        tit += 'SAT, '
    else:
        tit += 'SATA, '
    if np.int(WINDOW_LENGTH) == WINDOW_LENGTH:
        tit += ('%d-year window, %d-year shift' % (WINDOW_LENGTH, WINDOW_SHIFT))
    else:
        tit += ('%.2f-year window, %d-year shift' % (WINDOW_LENGTH, WINDOW_SHIFT))
    #plt.title(tit)
    tit = ('Evolution of difference in cond mean in temp SATA -- %s \n' % g_copy.location)
    tit += subtit
    plt.text(0.5, 1.05, tit, horizontalalignment = 'center', size = 16, transform = ax2.transAxes)
    #ax2.set_xticks(np.arange(start_date.year, end_date.year, 20))
    
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
        
        
        
ANOMALISE = True
PERIOD = 8 # years, period of wavelet
WINDOW_LENGTH = 16384 / 365.25
WINDOW_SHIFT = 1 # years, delta in the sliding window analysis
MEANS = True # if True, compute conditional means, if False, compute conditional variance
WORKERS = 3
NUM_SURR = 1000 # how many surrs will be used to evaluate


## loading data
g = load_station_data('TG_STAID000027.txt', date(1800,1,1), date(2014,1,1), ANOMALISE)
g.get_data_of_precise_length('32k', None, date(2012,5,8), True)


