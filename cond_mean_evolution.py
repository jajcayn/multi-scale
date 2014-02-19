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
PERIOD = 8 # years, period of wavelet
WINDOW_LENGTH = 8 # years, should be at least PERIOD of wavelet
WINDOW_SHIFT = 1 # years, delta in the sliding window analysis
PLOT = True
PAD = False # whether padding is used in wavelet analysis (see src/wavelet_analysis)
debug_plot = True


## loading data ##
print("[%s] Loading station data..." % (str(datetime.now())))
g = DataField()
g.load_station_data('Klemday07.raw')
print("** loaded")
start_date = date(1958,1,1)
end_date = date(2002, 11, 10)
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


phase_bins = np.linspace(-np.pi, np.pi, 9)
cond_means = np.zeros((len(phase_bins) - 1,))

# time evolution of sliding window
d, m, year = g.extract_day_month_year()
data_temp = np.zeros((WINDOW_LENGTH * y))
phase_temp = np.zeros((WINDOW_LENGTH * y))
difference = []
start_idx = g.find_date_ndx(start_date) # set to first date
end_idx = start_idx + data_temp.shape[0] # first date plus WINDOW_LENGTH years (since year is 365.25, leap years are counted)
cnt = 0
while end_idx < g.data.shape[0]: # while still in the correct range
    cnt += 1
    bin_cnt = []
    data_temp = g.data[start_idx : end_idx] # subselect data
    phase_temp = phase[0,start_idx : end_idx]
    for i in range(cond_means.shape[0]): # get conditional means for current phase range
        ndx = ((phase_temp >= phase_bins[i]) & (phase_temp <= phase_bins[i+1]))
        bin_cnt.append(ndx[ndx == True].shape)
        cond_means[i] = np.mean(data_temp[ndx])
    #if (cond_means.max() > 0. and cond_means.min() > 0.):
    difference.append(cond_means.max() - cond_means.min()) # append difference to list
    #else:
    #    difference.append(0)
    if debug_plot:
        fig = plt.figure(figsize=(7,14), dpi = 300)
        plt.subplot(211)
        plt.plot(phase[0,start_idx:end_idx], linewidth = 1.5)
        for i in range(len(phase_bins)):
            plt.axhline(y = phase_bins[i], color = 'red')
        plt.axis([0, WINDOW_LENGTH*y, -np.pi, np.pi])
        plt.title('%d.%d.%d - %d.%d.%d' % (d[start_idx], m[start_idx], year[start_idx], d[end_idx], m[end_idx], year[end_idx]), size = 20)
        plt.subplot(212)
        diff = (phase_bins[1]-phase_bins[0])
        rects = plt.bar(phase_bins[:-1]+diff*0.05, cond_means, width = diff*0.9, bottom = None, fc = '#403A37')
        k = 0
        for rect in rects: 
           height = rect.get_height()
           if height > 0. and height < 20.:
               plt.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%g' % bin_cnt[k], ha = 'center', va = 'bottom')
           k += 1
        plt.xlabel('phase [rad]')
        plt.ylabel('cond mean temperature [$^{\circ}$C]')
        plt.axis([-np.pi, np.pi, -5, 5])
        plt.title('Difference is %g' % (difference[-1]))
        plt.savefig('debug/plot%s' % str(cnt))
    start_idx = g.find_date_ndx(date(start_date.year + WINDOW_SHIFT * cnt, start_date.month, start_date.day)) # shift start index by WINDOW_SHIFT years
    end_idx = start_idx + data_temp.shape[0] # shift end index
print("[%s] Wavelet analysis done. Now plotting.." % (str(datetime.now())))
print cnt
#difference[difference == np.nan] = 0
#print difference
    
## plotting ##
if PLOT:
    fig = plt.figure(figsize=(10,7))
    plt.plot(difference, color = '#403A37', linewidth = 2, figure = fig)
    plt.axis([0, cnt-1, -5, 5])
    plt.xlabel('start year of %d-year wide window' % WINDOW_LENGTH)
    plt.xticks(np.linspace(0, cnt-1, 7), [i for i in range(start_date.year, end_date.year, 6)], rotation = 30)
    plt.ylabel('difference in cond mean temperature [$^{\circ}$C]')
    if not ANOMALISE:
        plt.title('Evolution of difference in cond mean temp, SAT, %d-year window, %d-year shift' % (WINDOW_LENGTH, WINDOW_SHIFT))
    else:
        plt.title('Evolution of difference in cond mean temp, SATA, %d-year window, %d-year shift' % (WINDOW_LENGTH, WINDOW_SHIFT))
    plt.savefig('debug/total.png')
  

    
    
    
    
    
    
    
    
    
    
