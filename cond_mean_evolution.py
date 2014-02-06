"""
created on Feb 6, 2014

@author: Nikola Jajcay
"""

import data_class
import numpy as np
import wavelet_analysis
from datetime import datetime, date
import matplotlib.pyplot as plt


ANOMALISE = False
PERIOD = 8 # years, period of wavelet
WINDOW_LENGTH = 8 # years, should be at least PERIOD of wavelet
WINDOW_SHIFT = 1 # years, delta in the sliding window analysis


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
wave, p, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, False, dj = 0, s0 = s0, j1 = 0, k0 = k0)

# oscillatory modes
phase = np.arctan2(np.imag(wave), np.real(wave))
phase_bins = np.linspace(-np.pi, np.pi, 9)

def days_in_years(s, e, shift):
    pass

phase_temp = np.zeros((WINDOW_LENGTH * y))
start_ndx = g.find_date_ndx(start_date)
idx = start_ndx
for i in range(phase.shape[1] - WINDOW_LENGTH + 1):
    phase_temp = phase[idx:idx+(WINDOW_LENGTH*y)]
    idx += days_in_years(start, end, WINDOW_SHIFT)

for i in range(cond_means.shape[0]):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))[0]
    cond_means[i] = np.mean(g.data[ndx])
    print phase_bins[i], phase_bins[i+1], cond_means[i]
    
difference = np.abs(cond_means.max()) + np.abs(cond_means.max())
