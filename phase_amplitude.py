"""
created on Feb 4, 2014

@author: Nikola Jajcay
"""

from src import wavelet_analysis
from src.data_class import load_station_data
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
from surrogates.surrogates import SurrogateField

ANOMALISE = True
PERIOD = 8 # years, period of wavelet
#WINDOW_LENGTH = 32 # years, should be at least PERIOD of wavelet
WINDOW_LENGTH = 16384 / 365.25
WINDOW_SHIFT = 1 # years, delta in the sliding window analysis
PLOT = True
PAD = False # whether padding is used in wavelet analysis (see src/wavelet_analysis)
MEANS = True # if True, compute conditional means, if False, compute conditional variance
USE_SURR = True


start_dates = [date(1834,7,28), date(1958,1,1)]
end_dates = [date(2014,1,1), date(2002,11,10)]

## loading data ##
start_date = start_dates[0]
end_date = end_dates[0] # exclusive
g = load_station_data('TG_STAID000027.txt', start_date, end_date, ANOMALISE)
print g.data[:20]
           
if USE_SURR:
    print("** replacing original data with surrogate data...")
    sg = SurrogateField()
    sg.copy_field(g)
    #sg.construct_multifractal_surrogates()
    sg.construct_fourier_surrogates_spatial()
    d = sg.get_surr()
    g.data = d.copy()

## analysis ##
# wavelet
print g.data[:20]
print("[%s] Wavelet analysis in progress with %d year window shifted by %d year(s)..." % (str(datetime.now()), WINDOW_LENGTH, WINDOW_SHIFT))
k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale 
wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, PAD, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes


cond_means = np.zeros((8,))

def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))

phase_bins = get_equidistant_bins() # equidistant bins
# conditional means
for i in range(cond_means.shape[0]):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))[0]
    cond_means[i] = np.mean(g.data[ndx])
    print phase_bins[i], phase_bins[i+1], cond_means[i]
print("difference between highest and lowest is %s degrees" % str(cond_means.max() - cond_means.min()))
print("[%s] Analysis complete. Plotting..." % (str(datetime.now())))
 
# plot as bar
if PLOT:
    diff = (phase_bins[1]-phase_bins[0])
    fig = plt.figure(figsize=(6,9))
    plt.bar(phase_bins[:-1]+diff*0.05, cond_means, width = diff*0.9, bottom = None, fc = '#403A37', figure = fig)
    plt.xlabel('phase [rad]')
    plt.ylabel('cond mean temperature [$^{\circ}$C]')
    plt.axis([-np.pi, np.pi, -1, 1])
    plt.title('SATA cond means - FT surrogates - %d data points \n %s - %s' % (g.data.shape[0], str(start_date), str(end_date)))
    plt.savefig('debug/MFsurr/SATA_FT_surr_64k_3.png')
    

