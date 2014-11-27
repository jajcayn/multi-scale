from src import wavelet_analysis
from src.data_class import load_station_data, DataField
import numpy as np
from surrogates.surrogates import SurrogateField
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


PERIOD = 8
BINS = 8
NUM_SURR = 1000

def get_equidistant_bins(num):
    return np.array(np.linspace(-np.pi, np.pi, num+1))


g = load_station_data('TG_STAID000027.txt', date(1924,1,14), date(2013,10,1), True)
g_amp = load_station_data('TG_STAID000027.txt', date(1924,1,14), date(2013, 10, 1), False)
g_data = DataField()

k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale 

wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes

period = 1 * 365.25 # frequency of interest
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
amplitude = m * amplitude + c
print("Oscillatory series fitted to SAT data with coeff. %.3f and intercept %.3f" % (m, c))

start_cut = date(1958,1,1)
g_data.data, g_data.time, idx = g.get_data_of_precise_length('16k', start_cut, None, False)
phase = phase[0, idx[0] : idx[1]]
amplitude = amplitude[idx[0] : idx[1]]

phase_bins = get_equidistant_bins(BINS)
cond_means = np.zeros((BINS, 2)) # :,0 - SATA, :,1 - SATamp

for i in range(cond_means.shape[0]):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    cond_means[i, 1] = np.mean(amplitude[ndx])
    cond_means[i, 0] = np.mean(g_data.data[ndx])

cond_means_surr = np.zeros((NUM_SURR, BINS, 2))

mean, var, trend = g.get_seasonality(True)
mean2, var2, trend2 = g_amp.get_seasonality(True)

for su in range(NUM_SURR):
    sg_amp = SurrogateField()
    sg_amp.copy_field(g_amp)
    sg = SurrogateField()
    sg.copy_field(g)

    # MF
    sg_amp.construct_multifractal_surrogates()
    sg_amp.add_seasonality(mean2, var2, trend2)
    sg.construct_multifractal_surrogates()
    sg.add_seasonality(mean, var, trend)

    # AR
    # sg_amp.prepare_AR_surrogates()
    # sg_amp.construct_surrogates_with_residuals()
    # sg_amp.add_seasonality(mean2[:-1], var2[:-1], trend2[:-1])
    # sg.prepare_AR_surrogates()
    # sg.construct_surrogates_with_residuals()
    # sg.add_seasonality(mean[:-1], var[:-1], trend[:-1])

    wave, _, _, _ = wavelet_analysis.continous_wavelet(sg.surr_data, 1, True, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
    phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes

    wave, _, _, _ = wavelet_analysis.continous_wavelet(sg_amp.surr_data, 1, True, wavelet_analysis.morlet, dj = 0, s0 = s0_amp, j1 = 0, k0 = k0) # perform wavelet
    amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))
    amplitude = amplitude[0, :]
    phase_amp = np.arctan2(np.imag(wave), np.real(wave))
    phase_amp = phase_amp[0, :]

    # fitting oscillatory phase / amplitude to actual SAT
    reconstruction = amplitude * np.cos(phase_amp)
    fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
    m, c = np.linalg.lstsq(fit_x, sg_amp.surr_data)[0]
    amplitude = m * amplitude + c

    _, _, idx = g.get_data_of_precise_length('16k', start_cut, None, False)
    phase = phase[0, idx[0] : idx[1]]
    amplitude = amplitude[idx[0] : idx[1]]
    sg.surr_data = sg.surr_data[idx[0] : idx[1]]

    for i in range(cond_means_surr.shape[1]):
        ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
        cond_means_surr[su, i, 1] = np.mean(amplitude[ndx])
        cond_means_surr[su, i, 0] = np.mean(sg.surr_data[ndx])

    if (su+1) % 10 == 0:
        print("%d. surrogate done..." % (su + 1))


fig = plt.figure(figsize = (10,8), frameon = False)
gs = gridspec.GridSpec(1, 2)
gs.update(left = 0.1, right = 0.9, top = 0.87, bottom = 0.1, wspace = 0.4, hspace = 0.4)

ax = plt.Subplot(fig, gs[0, 0])
fig.add_subplot(ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(color = '#6A4A3C')
diff = (phase_bins[1]-phase_bins[0])
ax.bar(phase_bins[:-1] + diff*0.05, cond_means[:, 0], width = diff*0.4, bottom = None, fc = '#BF3919', ec = '#BF3919',  figure = fig)
ax.bar(phase_bins[:-1] + diff*0.55, np.mean(cond_means_surr[:, :, 0], axis = 0), width = diff*0.4, bottom = None, fc = '#76C06E', ec = '#76C06E',  figure = fig)
ax.axis([-np.pi, np.pi, -1.5, 1.5])
ax.set_xlabel("phase [rad]", size = 16)
ax.set_ylabel("cond. mean SATA [$^{\circ}$C]", size = 16)
ax.tick_params(axis = 'both', which = 'major', labelsize = 15)

ax = plt.Subplot(fig, gs[0, 1])
fig.add_subplot(ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(color = '#6A4A3C')
diff = (phase_bins[1]-phase_bins[0])
ax.bar(phase_bins[:-1] + diff*0.05, cond_means[:, 1], width = diff*0.4, bottom = None, fc = '#BF3919', ec = '#BF3919', figure = fig)
ax.bar(phase_bins[:-1] + diff*0.55, np.mean(cond_means_surr[:, :, 1], axis = 0), width = diff*0.4, bottom = None, fc = '#76C06E', ec = '#76C06E',  figure = fig)
ax.axis([-np.pi, np.pi, 17, 23])
ax.set_xlabel("phase [rad]", size = 16)
ax.set_ylabel("cond. mean SAT amplitude [$^{\circ}$C]", size = 16)
ax.tick_params(axis = 'both', which = 'major', labelsize = 15)

tit = "%s -- %s - %s" % (g.location, g.get_date_from_ndx(0), g.get_date_from_ndx(-1))
plt.suptitle(tit, size = 22)

fname = "debug/PRGhistSATAvsSATamp1000MF"
plt.savefig(fname + ".png")