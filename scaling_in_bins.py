"""
created on July 3, 2014

@author: Nikola Jajcay
"""

import numpy as np
from src.data_class import load_station_data
from datetime import date
from src import wavelet_analysis as wvlt


def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))
    


PERIOD = 8
WINDOW_LENGTH = 13462 # 13462, 16384
MIDDLE_YEAR = 1961 # year around which the window will be deployed


# load whole data
g = load_station_data('TG_STAID000027.txt', date(1775, 1, 1), date(2014, 1, 1), True)
# starting month and day
sm = 7
sd = 28
# starting year of final window
y = 365.25
sy = MIDDLE_YEAR - (WINDOW_LENGTH/y)/2

# get wvlt window
start = g.find_date_ndx(date(sy - 4, sm, sd))
end = start + 16384 if WINDOW_LENGTH < 16000 else start + 32768
g.data = g.data[start : end]
g.time = g.time[start : end]

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

# binning
phase_bins = get_equidistant_bins()
for i in range(phase_bins.shape[0] - 1):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    ## do stuff!!


