"""
created on May 22, 2014

@author: Nikola Jajcay
"""

from src import wavelet_analysis
from src.data_class import load_station_data, DataField
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
from surrogates.surrogates import SurrogateField


ANOMALISE = True
PERIOD = 8 # years, period of wavelet
WINDOW_LENGTH = 16384 / 365.25
WINDOW_SHIFT = 1 # years, delta in the sliding window analysis
MEANS = True # if True, compute conditional means, if False, compute conditional variance
NUM_SURR = 100
season = [9,10,11]
s_name = 'SON'
s_num = 3

# load data - at least 32k of data because of surrogates
g = load_station_data('TG_STAID000027.txt', date(1924, 1, 1), date(2013,9,18), ANOMALISE)
g_data = DataField()


print("[%s] Wavelet analysis in progress with %d year window shifted by %d year(s)..." % (str(datetime.now()), WINDOW_LENGTH, WINDOW_SHIFT))
k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale 

cond_means = np.zeros((8,))

def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))
    
# wavelet - data    
wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes

start_cut = date(1958,1,1)
g_data.data, g_data.time, idx = g.get_data_of_precise_length('16k', start_cut, None, False)
phase = phase[0, idx[0] : idx[1]]
# subselect season
#ndx_season = g_data.select_months(season)
#phase = phase[ndx_season]

phase_bins = get_equidistant_bins() # equidistant bins
for i in range(cond_means.shape[0]):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    cond_means[i] = np.var(g_data.data[ndx], ddof = 1)
    
    
cond_means_surr = np.zeros((NUM_SURR, 8))

mean, var, trend = g.get_seasonality(True)


for su in range(NUM_SURR):
    sg = SurrogateField()
    sg.copy_field(g)
    sg.construct_multifractal_surrogates()
    sg.add_seasonality(mean, var, trend)
    wave, _, _, _ = wavelet_analysis.continous_wavelet(sg.surr_data, 1, True, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
    phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes

    _, _, idx = g.get_data_of_precise_length('16k', start_cut, None, False)
    sg.surr_data = sg.surr_data[idx[0] : idx[1]]
    phase = phase[0, idx[0] : idx[1]]
    
    # subselect season
#    sg.surr_data = sg.surr_data[ndx_season]
#    phase = phase[ndx_season]
    for i in range(cond_means.shape[0]):
        ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
        cond_means_surr[su, i] = np.var(sg.surr_data[ndx], ddof = 1)
        
    if su+1 % 10 == 0:
        print su+1, '. surrogate done...'
    
print("[%s] Wavelet done." % (str(datetime.now())))


diff = (phase_bins[1]-phase_bins[0])
fig = plt.figure(figsize=(6,10))
b1 = plt.bar(phase_bins[:-1], cond_means, width = diff*0.45, bottom = None, fc = '#403A37', figure = fig)
b2 = plt.bar(phase_bins[:-1] + diff*0.5, np.mean(cond_means_surr, axis = 0), width = diff*0.45, bottom = None, fc = '#A09793', figure = fig)
plt.xlabel('phase [rad]')
mean_of_diffs = np.mean([cond_means_surr[i,:].max() - cond_means_surr[i,:].min() for i in range(cond_means_surr.shape[0])])
std_of_diffs = np.std([cond_means_surr[i,:].max() - cond_means_surr[i,:].min() for i in range(cond_means_surr.shape[0])], ddof = 1)
plt.legend( (b1[0], b2[0]), ('data', 'mean of %d surr' % NUM_SURR) )
plt.ylabel('cond variance temperature [$^{\circ}$C$^{2}$]')
plt.axis([-np.pi, np.pi, 7, 30])
plt.title('%s cond variance \n difference data: %.2f$^{\circ}$C \n mean of diffs: %.2f$^{\circ}$C \n std of diffs: %.2f$^{\circ}$C$^{2}$' % (g.location, 
           (cond_means.max() - cond_means.min()), mean_of_diffs, std_of_diffs))

plt.savefig('case_study_58-02/cond_variance.png')
        
