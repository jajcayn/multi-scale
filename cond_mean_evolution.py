"""
created on Feb 6, 2014

@author: Nikola Jajcay
"""

from src import data_class, wavelet_analysis
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt


ANOMALISE = False
PERIOD = 8 # years, period of wavelet
WINDOW_LENGTH = 8 # years, should be at least PERIOD of wavelet
WINDOW_SHIFT = 2 # years, delta in the sliding window analysis
PLOT = True


## loading data ##
print("[%s] Loading station data..." % (str(datetime.now())))
g = data_class.DataField()
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
print("[%s] Wavelet analysis in progress..." % (str(datetime.now())))
k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale 
phase_bins = np.linspace(-np.pi, np.pi, 9)
cond_means = np.zeros((len(phase_bins) - 1,))

# time evolution of sliding window
data_temp = np.zeros((WINDOW_LENGTH * y))
difference = []
start_idx = g.find_date_ndx(start_date) # set to first date
end_idx = start_idx + data_temp.shape[0] # first date plus WINDOW_LENGTH years (since year is 365.25, leap years are counted)
cnt = 0
while end_idx < g.data.shape[0]: # while still in the correct range
    cnt += 1
    data_temp = g.data[start_idx : end_idx] # subselect data
    wave, _, _, _ = wavelet_analysis.continous_wavelet(data_temp, 1, False, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
    phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes
    for i in range(cond_means.shape[0]): # get conditional means for current phase range
        ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))[0]
        cond_means[i] = np.mean(data_temp[ndx])
    difference.append(cond_means.max() - cond_means.min()) # append difference to list
    print start_idx, end_idx, difference[-1]
    start_idx = g.find_date_ndx(date(start_date.year + WINDOW_SHIFT * cnt, start_date.month, start_date.day)) # shift start index by WINDOW_SHIFT years
    end_idx = start_idx + data_temp.shape[0] # shift end index
print("[%s] Wavelet analysis done. Now plotting.." % (str(datetime.now())))
print cnt
    
    
## plotting ##
if PLOT:
    fig = plt.figure(figsize=(10,7))
    plt.plot(difference, color = '#403A37', linewidth = 2, figure = fig)
    plt.axis([0, cnt-1, 0, 3])
    plt.xlabel('start year of %d-year wide window' % WINDOW_LENGTH)
    plt.xticks(np.linspace(0, cnt-1, 7), [i for i in range(start_date.year, end_date.year, 6)], rotation = 30)
    plt.ylabel('difference in cond mean temperature [$^{\circ}$C]')
    if not ANOMALISE:
        plt.title('Evolution of difference in cond mean temp, SAT, %d-year window, %d-year shift' % (WINDOW_LENGTH, WINDOW_SHIFT))
    else:
        plt.title('Evolution of difference in cond mean temp, SATA, %d-year window, %d-year shift' % (WINDOW_LENGTH, WINDOW_SHIFT))
    plt.show()
  

    
    
    
    
    
    
    
    
    
    
