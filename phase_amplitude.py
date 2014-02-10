"""
created on Feb 4, 2014

@author: Nikola Jajcay
"""

from src import data_class, wavelet_analysis
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt

PLOT = True

# loading data
print("[%s] Loading station data..." % (str(datetime.now())))
g = data_class.DataField()
g.load_station_data('Klemday07.raw')
g.select_date(date(1958,1,1), date(2002, 11, 10))
#g.anomalise()
day, month, year = g.extract_day_month_year()
print("[%s] Data from %s loaded with shape %s. Date range is %d.%d.%d - %d.%d.%d inclusive." 
        % (str(datetime.now()), g.location, str(g.data.shape), day[0], month[0], 
           year[0], day[-1], month[-1], year[-1]))
                     
# wavelet
k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = 8 * y # frequency of interest - 8 years
s0 = period / fourier_factor # get scale 
print("[%s] Wavelet analysis in progress..." % (str(datetime.now())))
wave, p, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, False, dj = 0, s0 = s0, j1 = 0, k0 = k0)

# oscillatory modes
phase = np.arctan2(np.imag(wave), np.real(wave))
amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))

phase_bins = np.linspace(-np.pi, np.pi, 9)
cond_means = np.zeros((len(phase_bins) - 1,))

# conditional means
for i in range(cond_means.shape[0]):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))[0]
    cond_means[i] = np.mean(g.data[ndx])
    print phase_bins[i], phase_bins[i+1], cond_means[i]
print("[%s] Analysis complete. Plotting..." % (str(datetime.now())))
 
# plot as bar
if PLOT:
    diff = (phase_bins[1]-phase_bins[0])
    fig = plt.figure(figsize=(6,9))
    plt.bar(phase_bins[:-1]+diff*0.05, cond_means, width = diff*0.9, bottom = None, fc = '#403A37', figure = fig)
    plt.xlabel('phase [rad]')
    plt.ylabel('cond mean temperature [$^{\circ}$C]')
    plt.axis([-np.pi, np.pi, -2, 2])
    plt.title('SATA, padding, mean, %d years' % (p/y))
    plt.show()
    

