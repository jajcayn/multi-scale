from src import wavelet_analysis
from src.data_class import load_station_data
import numpy as np
from datetime import date
import matplotlib.pyplot as plt



AMP_PERIODS = [1, 1]

g_amp = load_station_data('TG_STAID000027.txt', date(1924,1,14), date(2013,10,1), False)
g_amp_sata = load_station_data('TG_STAID000027.txt', date(1924,1,14), date(2013,10,1), True)

k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))

result = dict()

start_cut = date(1958,1,1)
_, _, idx = g_amp.get_data_of_precise_length('16k', start_cut, None, False)
fields = [g_amp, g_amp_sata]

for i in range(len(AMP_PERIODS)):
    period = AMP_PERIODS[i] * 365.25 # frequency of interest
    s0_amp = period / fourier_factor # get scale
    wave, _, _, _ = wavelet_analysis.continous_wavelet(fields[i].data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0_amp, j1 = 0, k0 = k0) # perform wavelet
    amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))
    amplitude = amplitude[0, :]
    phase_amp = np.arctan2(np.imag(wave), np.real(wave))
    phase_amp = phase_amp[0, :]

    # fitting oscillatory phase / amplitude to actual SAT
    reconstruction = amplitude * np.cos(phase_amp)
    fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
    m, c = np.linalg.lstsq(fit_x, fields[i].data)[0]
    # amplitude = m * amplitude + c
    amp = m * reconstruction + c
    result[i] = amp[idx[0] : idx[1]]
    result[i+2] = phase_amp[idx[0] : idx[1]]
    print("Oscillatory series fitted to SAT data with coeff. %.3f and intercept %.3f" % (m, c))
    fields[i].anomalise()

g_amp.time = g_amp.time[idx[0] : idx[1]]
plt.figure(figsize = (12,8))
p1, = plt.plot(result[0], color = "#110D29", linewidth = 1.5)
p2, = plt.plot(result[1], color = "#B87BA5")
# p3, = plt.plot(result[2], color = "#392423")
# p4, = plt.plot(result[3], color = "#050505")
plt.ylabel("Regressed $A \cos{\phi}$ [$^{\circ}$C]")
plt.xlabel("time")
# plt.legend([p1,p2,p3,p4], ['SAT amp', 'SATA amp', 'SAT phase', 'SATA phase'])
# plt.axis([0, result[1].shape[0], -5, 25])
plt.title("%s SAT \n %s -- %s" % (g_amp.location, str(g_amp.get_date_from_ndx(0)), str(g_amp.get_date_from_ndx(-1))))
plt.savefig('debug/cycles1year.png')


