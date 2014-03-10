"""
created on Feb 6, 2014

@author: Nikola Jajcay
"""

from src import wavelet_analysis
from src.data_class import DataField
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt


ANOMALISE = False
PERIOD = 10 # years, period of wavelet
#WINDOW_LENGTH = 32 # years, should be at least PERIOD of wavelet
WINDOW_LENGTH = 16384 / 365.25
WINDOW_SHIFT = 1 # years, delta in the sliding window analysis
PLOT = True
PAD = False # whether padding is used in wavelet analysis (see src/wavelet_analysis)
debug_plot = True # partial
MEANS = True # if True, compute conditional means, if False, compute conditional variance


## loading data ##
print("[%s] Loading station data..." % (str(datetime.now())))
g = DataField()
g.load_station_data('TG_STAID000027.txt', dataset = "ECA-station")
print("** loaded")
start_date = date(1834,7,28)
end_date = date(2014, 1, 1) # exclusive
# length of the time series with date(1954,6,8) with start date(1775,1,1) = 65536 - power of 2
# the same when end date(2014,1,1) than start date(1834,7,28)
g.select_date(start_date, end_date)
if ANOMALISE:
    print("** anomalising")
    g.anomalise()
day, month, year = g.extract_day_month_year()
print("[%s] Data from %s loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
        % (str(datetime.now()), g.location, str(g.data.shape), day[0], month[0], 
           year[0], day[-1], month[-1], year[-1]))
           

## analysis ##
# wavelet
print("[%s] Wavelet analysis in progress with %d year window shifted by %d year(s)..." % (str(datetime.now()), WINDOW_LENGTH, WINDOW_SHIFT))
k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale 
wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, PAD, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes


cond_means = np.zeros((8,))

# solution where all bins are edited
def get_equiquantal_bins(phase_part):
    # use iterative algorythm
    volume = int(phase_part.shape[0] / 8.)
    start = phase_part.min()
    phase_bins = [start]
    end = phase_part.max()
    for i in range(7):
        ndx = np.zeros_like(phase_part, dtype = np.bool)
        idx = phase_bins[-1]
        while (ndx[ndx == True].shape[0] < volume):
            ndx = ((phase_part >= phase_bins[-1]) & (phase_temp <= idx))
            idx += 0.001
        phase_bins.append(idx)
    phase_bins.append(end)
        
    return np.array(phase_bins)
    
def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))

# time evolution of sliding window
d, m, year = g.extract_day_month_year()
data_temp = np.zeros((WINDOW_LENGTH * y))
phase_temp = np.zeros((WINDOW_LENGTH * y))
difference = []
mean_var = []
start_idx = g.find_date_ndx(start_date) # set to first date
end_idx = start_idx + data_temp.shape[0] # first date plus WINDOW_LENGTH years (since year is 365.25, leap years are counted)
cnt = 0
while end_idx < g.data.shape[0]: # while still in the correct range
    cnt += 1
    bin_cnt = []
    data_temp = g.data[start_idx : end_idx] # subselect data
    phase_temp = phase[0,start_idx : end_idx]
    for i in range(cond_means.shape[0]): # get conditional means for current phase range
        #phase_bins = get_equiquantal_bins(phase_temp) # equiquantal bins
        phase_bins = get_equidistant_bins() # equidistant bins
        ndx = ((phase_temp >= phase_bins[i]) & (phase_temp <= phase_bins[i+1]))
        bin_cnt.append(ndx[ndx == True].shape)
        if MEANS:
            cond_means[i] = np.mean(data_temp[ndx])
        else:
            cond_means[i] = np.var(data_temp[ndx], ddof = 1)
    #if (cond_means.max() > 0. and cond_means.min() > 0.):
    difference.append(cond_means.max() - cond_means.min()) # append difference to list    
    mean_var.append(np.mean(cond_means))
    #else:
    #    difference.append(0)
    if debug_plot:
        if (year[start_idx] > 1830 and year[end_idx] < 2014):
            fig = plt.figure(figsize=(7,14), dpi = 300)
            plt.subplot(211)
            plt.plot(phase[0,start_idx:end_idx], linewidth = 1.5)
            for i in range(len(phase_bins)):
                plt.axhline(y = phase_bins[i], color = 'red')
            plt.axis([0, WINDOW_LENGTH*y, -np.pi, np.pi])
            plt.xticks(np.linspace(0, len(phase[0,start_idx:end_idx]), 9), year[start_idx:end_idx:int((end_idx-start_idx)/9)])
            if not ANOMALISE and MEANS:
                plt.title('SAT cond mean \n %d.%d.%d - %d.%d.%d' % (d[start_idx], m[start_idx], year[start_idx], d[end_idx], 
                                                  m[end_idx], year[end_idx]), size = 20)
            elif not ANOMALISE and not MEANS:
                plt.title('SAT cond variance \n %d.%d.%d - %d.%d.%d' % (d[start_idx], m[start_idx], year[start_idx], d[end_idx], 
                                                  m[end_idx], year[end_idx]), size = 20)
            elif ANOMALISE and MEANS:
                plt.title('SATA cond mean \n %d.%d.%d - %d.%d.%d' % (d[start_idx], m[start_idx], year[start_idx], d[end_idx], 
                                                  m[end_idx], year[end_idx]), size = 20)
            elif ANOMALISE and not MEANS:
                plt.title('SATA cond variance \n %d.%d.%d - %d.%d.%d' % (d[start_idx], m[start_idx], year[start_idx], d[end_idx], 
                                                  m[end_idx], year[end_idx]), size = 20)
            plt.xlabel('years')
            plt.ylabel('phase [rad]')
            plt.subplot(212)
            #diff = (phase_bins[1]-phase_bins[0])
            rects = plt.bar(phase_bins[:-1], cond_means, width = 0.5, bottom = None, fc = '#403A37')
            k = 0
            for rect in rects: 
               height = rect.get_height()
               if height > 0. and height < 30.:
                   plt.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%g' % bin_cnt[k], ha = 'center', va = 'bottom')
               k += 1
            plt.xlabel('phase [rad]')
            if MEANS:
                plt.ylabel('cond mean in temperature [$^{\circ}$C]')
            elif not MEANS:
                plt.ylabel('cond variance in temperature [$^{\circ}$C$^2$]')
            plt.axis([-np.pi, np.pi, 5, 15])
            plt.title('Difference is %g \n Mean is %g' % (difference[-1], mean_var[-1]))
            if not ANOMALISE:
                fname = 'debug/SAT_'
            else:
                fname = 'debug/SATA_'
            if MEANS:
                fname += 'means_'
            else:
                fname += 'var_'
            fname += ('%dyears/%dperiod_plot_%s-%s.png' % (WINDOW_LENGTH, PERIOD, str(date(year[start_idx], m[start_idx], d[start_idx])), 
                                               str(date(year[end_idx], m[end_idx], d[end_idx]))))
            plt.savefig(fname)
    start_idx = g.find_date_ndx(date(start_date.year + WINDOW_SHIFT * cnt, start_date.month, start_date.day)) # shift start index by WINDOW_SHIFT years
    end_idx = start_idx + data_temp.shape[0] # shift end index
print("[%s] Wavelet analysis done. Now plotting.." % (str(datetime.now())))
#difference[difference == np.nan] = 0
print len(difference)
    
## plotting ##
if PLOT:
    fig, ax1 = plt.subplots(figsize=(11,8))
    #fig = plt.figure(figsize=(10,7))
    ax1.plot(difference, color = '#403A37', linewidth = 2, figure = fig)
    if not ANOMALISE and MEANS:
        ax1.axis([0, cnt-1, 0, 3.5])
    if not ANOMALISE and not MEANS:
        ax1.axis([0, cnt-1, 0, 35])
    if ANOMALISE and MEANS:
        ax1.axis([0, cnt-1, 0, 2])
    if ANOMALISE and not MEANS:
        ax1.axis([0, cnt-1, 0, 10])
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
    ax2.plot(mean_var, color = '#CA4F17', linewidth = 2, figure = fig)
    if MEANS:
        ax2.set_ylabel('mean of cond means in temperature [$^{\circ}$C]', size = 14)
    elif not MEANS:
        ax2.set_ylabel('mean of cond variance in temperature [$^{\circ}$C$^2$]', size = 14)
    ax2.axis([0, cnt-1, 8.5, 11.5])
    for tl in ax2.get_yticklabels():
        tl.set_color('#CA4F17')
    tit = 'Evolution of difference in cond'
    if MEANS:
        tit += ' mean in temp, '
    else:
        tit += ' variance in temp, '
    if not ANOMALISE:
        tit += 'SAT, '
    else:
        tit += 'SATA, '
    if np.int(WINDOW_LENGTH) == WINDOW_LENGTH:
        tit += ('%d-year window, %d-year shift,\n %d-year wavelet' % (WINDOW_LENGTH, WINDOW_SHIFT, PERIOD))
    else:
        tit += ('%.2f-year window, %d-year shift,\n %d-year wavelet' % (WINDOW_LENGTH, WINDOW_SHIFT, PERIOD))
    #plt.title(tit)
    plt.text(0.5, 1.05, tit, horizontalalignment = 'center', size = 16, transform = ax2.transAxes)
    #ax2.set_xticks(np.arange(start_date.year, end_date.year, 20))
    if not ANOMALISE:
        fname = 'SAT_'
    else:
        fname = 'SATA_'
    if MEANS:
        fname += 'means_'
    else:
        fname += 'var_'
    fname += ('%dyears_%dperiod.png' % (WINDOW_LENGTH, PERIOD))
    plt.savefig('debug/' + fname)
  

    
    
    
    
    
    
    
    
    
    
