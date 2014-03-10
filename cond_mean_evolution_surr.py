"""
created on Mar 8, 2014

@author: Nikola Jajcay
"""

from src import wavelet_analysis
from src.data_class import DataField
from surrogates.surrogates import construct_fourier_surrogates
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt


ANOMALISE = False
PERIOD = 6 # years, period of wavelet
#WINDOW_LENGTH = 32 # years, should be at least PERIOD of wavelet
WINDOW_LENGTH = 16384 / 365.25
WINDOW_SHIFT = 1 # years, delta in the sliding window analysis
PLOT = True
PAD = False # whether padding is used in wavelet analysis (see src/wavelet_analysis)
debug_plot = False # partial
MEANS = False # if True, compute conditional means, if False, compute conditional variance
num_surr = 10 # how many surrs will be used to evaluate



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
           
           
print("** Using surrogate data..")
mean, var = g.get_seasonality()
data_copy = g.data.copy()
    

## analysis ##
# wavelet
print("[%s] Wavelet analysis in progress with %d year window shifted by %d year(s)..." % (str(datetime.now()), WINDOW_LENGTH, WINDOW_SHIFT))
k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale 
cond_means = np.zeros((8,))

d, m, year = g.extract_day_month_year()


def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))


total_diffs = []
total_meanvars = []

for iota in range(num_surr):
    # prepare surrogates
    surr = construct_fourier_surrogates(data_copy) # generate surrogates from deseasonalised data
    surr *= var # add deviation to surrogates
    g.data = surr + mean # add mean to surrogates
    
    # wavelet
    wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, PAD, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
    phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes
    
    
    difference = []
    mean_var = []
    data_temp = np.zeros((WINDOW_LENGTH * y))
    phase_temp = np.zeros((WINDOW_LENGTH * y))
    start_idx = g.find_date_ndx(start_date) # set to first date
    end_idx = start_idx + data_temp.shape[0] # first date plus WINDOW_LENGTH years (since year is 365.25, leap years are counted)
    cnt = 0
    while end_idx < g.data.shape[0]: # while still in the correct range
        cnt += 1
        data_temp = g.data[start_idx : end_idx] # subselect data
        phase_temp = phase[0,start_idx : end_idx]
        for i in range(cond_means.shape[0]): # get conditional means for current phase range
            #phase_bins = get_equiquantal_bins(phase_temp) # equiquantal bins
            phase_bins = get_equidistant_bins() # equidistant bins
            ndx = ((phase_temp >= phase_bins[i]) & (phase_temp <= phase_bins[i+1]))
            if MEANS:
                cond_means[i] = np.mean(data_temp[ndx])
            else:
                cond_means[i] = np.var(data_temp[ndx], ddof = 1)
        difference.append(cond_means.max() - cond_means.min()) # append difference to list    
        mean_var.append(np.mean(cond_means))
        start_idx = g.find_date_ndx(date(start_date.year + WINDOW_SHIFT * cnt, start_date.month, start_date.day)) # shift start index by WINDOW_SHIFT years
        end_idx = start_idx + data_temp.shape[0] # shift end index
    total_diffs.append(np.array(difference))
    total_meanvars.append(np.array(mean_var))

    print("[%s] Wavelet analysis done. Now plotting.." % (str(datetime.now())))
    
   
    if PLOT:
        fig, ax1 = plt.subplots(figsize=(11,8))
        ax1.plot(total_diffs[-1], color = '#403A37', linewidth = 2, figure = fig)
        #ax1.plot(total_diffs[0], np.arange(0,len(total_diffs[0])), total_diffs[1], np.arange(0, cnt))
        if not ANOMALISE and MEANS:
            ax1.axis([0, cnt-1, 0, 6])
        if not ANOMALISE and not MEANS:
            ax1.axis([0, cnt-1, 0, 30])
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
        ax2.plot(total_meanvars[-1], color = '#CA4F17', linewidth = 2, figure = fig) # color = '#CA4F17'
        if MEANS:
            ax2.set_ylabel('mean of cond means in temperature [$^{\circ}$C]', size = 14)
        elif not MEANS:
            ax2.set_ylabel('mean of cond variance in temperature [$^{\circ}$C$^2$]', size = 14)
        ax2.axis([0, cnt-1, 60, 75])
        for tl in ax2.get_yticklabels():
            tl.set_color('#CA4F17')
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
        plt.text(0.5, 1.05, tit, horizontalalignment = 'center', size = 16, transform = ax2.transAxes)
        #ax2.set_xticks(np.arange(start_date.year, end_date.year, 20))
        if not ANOMALISE:
            fname = 'SURR_SAT_'
        else:
            fname = 'SURR_SATA_'
        if MEANS:
            fname += 'means_'
        else:
            fname += 'var_'
        fname += ('%dyears_%dperiod_%d.png' % (WINDOW_LENGTH, PERIOD, iota))
        plt.savefig('debug/' + fname)