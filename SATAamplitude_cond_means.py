from src import wavelet_analysis
from src.data_class import load_station_data, DataField
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from surrogates.surrogates import SurrogateField


PERIOD = 8
AMP_PERIOD = 8
BINS = 8
ANOMALISE = True # amplitude from SAT / SATA
SURR = True


g = load_station_data('TG_STAID000027.txt', date(1834,4,27), date(2013,10,1), True)
g_amp = load_station_data('TG_STAID000027.txt', date(1834,4,27), date(2013,10,1), ANOMALISE)
g_data = DataField()



k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale 
# wavelet - data    
wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes

period = AMP_PERIOD * 365.25 # frequency of interest
s0_amp = period / fourier_factor # get scale
wave, _, _, _ = wavelet_analysis.continous_wavelet(g_amp.data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0_amp, j1 = 0, k0 = k0) # perform wavelet
amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))
amplitude = amplitude[0, :]
phase_amp = np.arctan2(np.imag(wave), np.real(wave))
phase_amp = phase_amp[0, :]

# fitting oscillatory phase / amplitude to actual SAT
reconstruction = amplitude * np.cos(phase_amp)
fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
m, c = np.linalg.lstsq(fit_x, g_amp.data)[0]
# amplitude = m * amplitude + c
amplitude = m * reconstruction + c
print("Oscillatory series fitted to SAT data with coeff. %.3f and intercept %.3f" % (m, c))


if SURR:
    mean, var, trend = g_amp.get_seasonality(True)
    sg = SurrogateField()
    sg.copy_field(g_amp)
    # sg.construct_multifractal_surrogates()
    sg.construct_fourier_surrogates_spatial()
    sg.add_seasonality(mean, var, trend)

    period = AMP_PERIOD * 365.25 # frequency of interest
    s0_amp = period / fourier_factor # get scale
    wave, _, _, _ = wavelet_analysis.continous_wavelet(sg.surr_data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0_amp, j1 = 0, k0 = k0) # perform wavelet
    amplitude2 = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))
    amplitude2 = amplitude2[0, :]
    phase_amp = np.arctan2(np.imag(wave), np.real(wave))
    phase_amp = phase_amp[0, :]

    reconstruction = amplitude2 * np.cos(phase_amp)
    fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
    m, c = np.linalg.lstsq(fit_x, sg.surr_data)[0]
    # amplitude2 = m * amplitude2 + c
    amplitude2 = m * reconstruction + c

# plt.figure(figsize=(20,10))
# plt.plot(amplitude, color = '#867628', linewidth = 2)
# plt.plot(g_amp.data, color = '#004739', linewidth = 1)
# plt.show()

cond_means = np.zeros((BINS,2))

def get_equidistant_bins(num):
    return np.array(np.linspace(-np.pi, np.pi, num+1))

start_cut = date(1900,1,1)
g_data.data, g_data.time, idx = g.get_data_of_precise_length('32k', start_cut, None, False)
phase = phase[0, idx[0] : idx[1]]
amplitude = amplitude[idx[0] : idx[1]]
amplitude2 = amplitude2[idx[0] : idx[1]]


phase_bins = get_equidistant_bins(BINS)

for i in range(cond_means.shape[0]):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    cond_means[i, 0] = np.mean(amplitude[ndx])
    cond_means[i, 1] = np.mean(amplitude2[ndx])


diff = (phase_bins[1]-phase_bins[0])
fig = plt.figure(figsize=(6,10))
b1 = plt.bar(phase_bins[:-1] + diff*0.05, cond_means[:, 0], width = diff*0.4, bottom = None, fc = '#867628', ec = '#867628', figure = fig)
b2 = plt.bar(phase_bins[:-1] + diff*0.55, cond_means[:, 1], width = diff*0.4, bottom = None, fc = '#004739', ec = '#004739', figure = fig)
plt.xlabel('phase [rad]')
plt.legend([b1[0], b2[0]], ['%s amp' % ('SATA' if ANOMALISE else 'SAT'), '%s amp%s' % ('SATA' if ANOMALISE else 'SAT',  ' - FT surr' if SURR else '')])
plt.ylabel('cond mean %s' % ('SATA' if ANOMALISE else 'SAT'))
plt.xlim([-np.pi, np.pi])
plt.title('PRG %s %d-year $A \cos{\phi}$ \n %s -- %s' % ('SATA' if ANOMALISE else 'SAT', AMP_PERIOD, str(g_data.get_date_from_ndx(0)), str(g_data.get_date_from_ndx(-1))))
plt.savefig('debug/PRG_%samplitude%s5.png' % ('SATA' if ANOMALISE else 'SAT', 'FTsurr' if SURR else ''))