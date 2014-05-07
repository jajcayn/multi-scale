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
num_surr = 100
bins = 8


start_dates = [date(1834,7,28), date(1958,1,1)]
end_dates = [date(2014,1,1), date(2002,11,10)]

## loading data ##
start_date = start_dates[0]
end_date = end_dates[0] # exclusive
g = load_station_data('TG_STAID000027.txt', date(1834, 7, 28), date(1879,6,6), ANOMALISE)

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


cond_means = np.zeros((bins,))

def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, bins+1))

phase_bins = get_equidistant_bins() # equidistant bins
# conditional means
for i in range(cond_means.shape[0]):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))[0]
    print i, ndx[ndx == True].shape
    cond_means[i] = np.mean(g.data[ndx])
print cond_means
print("difference between highest and lowest is %s degrees" % str(cond_means.max() - cond_means.min()))
print("[%s] Analysis complete. Plotting..." % (str(datetime.now())))

cond_means_surr = np.zeros((num_surr, bins))
mean, var, trend = g.get_seasonality(True)

sg = SurrogateField()
sg.copy_field(g)

for su in range(num_surr):
    sg.construct_multifractal_surrogates()
    #sg.construct_fourier_surrogates_spatial()
    sg.add_seasonality(mean, var, trend)
    wave, _, _, _ = wavelet_analysis.continous_wavelet(sg.surr_data, 1, PAD, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
    phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes
    
    for i in range(cond_means.shape[0]):
        ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))[0]
        cond_means_surr[su, i] = np.mean(sg.surr_data[ndx])
        
    print su, '. surrogate done....'



# plot as bar
if PLOT:
    print 'done'
    diff = (phase_bins[1]-phase_bins[0])
    fig = plt.figure(figsize=(6,10))
    b1 = plt.bar(phase_bins[:-1], cond_means, width = diff*0.45, bottom = None, fc = '#403A37', figure = fig)
    b2 = plt.bar(phase_bins[:-1] + diff*0.5, np.mean(cond_means_surr, axis = 0), width = diff*0.45, bottom = None, fc = '#A09793', figure = fig)
    plt.xlabel('phase [rad]')
    diff_of_means = np.mean(cond_means_surr, axis = 0).max() - np.mean(cond_means_surr, axis = 0).min()
    mean_of_diffs = np.mean([cond_means_surr[i,:].max() - cond_means_surr[i,:].min() for i in range(cond_means_surr.shape[0])])
    std_of_diffs = np.std([cond_means_surr[i,:].max() - cond_means_surr[i,:].min() for i in range(cond_means_surr.shape[0])], ddof = 1)
    plt.legend( (b1[0], b2[0]), ('data', 'mean of 100 surr') )
    plt.ylabel('cond variance temperature [$^{\circ}$C$^{2}$]')
    plt.axis([-np.pi, np.pi, -1, 1])
    plt.title('%s cond means \n difference data: %.2f$^{\circ}$C \n diff of means (mean of diffs): %.2f$^{\circ}$C (%.2f$^{\circ}$C) \n std of diffs: %.2f$^{\circ}$C$^{2}$' % (g.location, 
              (cond_means.max() - cond_means.min()), diff_of_means, mean_of_diffs, std_of_diffs))
    plt.show()
    #plt.savefig('debug/PRG_variance_bins_%dyear_%dbins.png' % (PERIOD, bins))
    

