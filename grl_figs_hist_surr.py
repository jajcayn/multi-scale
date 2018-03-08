from pyclits.geofield import DataField
from pyclits.data_loaders import load_station_data
import numpy as np
from pyclits.surrogates import SurrogateField
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


PERIOD = 8
BINS = 8
NUM_SURR = 1000

def get_equidistant_bins(num):
    return np.array(np.linspace(-np.pi, np.pi, num+1))


g = load_station_data('../data/ECAstation-TG/TG_STAID000027.txt', date(1924,1,14), date(2013,10,1), anom=True)
g_amp = load_station_data('../data/ECAstation-TG/TG_STAID000027.txt', date(1924,1,14), date(2013, 10, 1), anom=False)
g_data = DataField()

mean, var, trend = g.get_seasonality(True)
mean2, var2, trend2 = g_amp.get_seasonality(True)

sg_amp = SurrogateField()
sg_amp.copy_field(g_amp)
sg = SurrogateField()
sg.copy_field(g)

g.return_seasonality(mean, var, trend)
g_amp.return_seasonality(mean2, var2, trend2)

g.wavelet(PERIOD, period_unit='y', cut=None)
# wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = g.phase.copy()

g_amp.wavelet(1, period_unit='y', cut=None)
phase_amp = g_amp.phase.copy()
amplitude = g_amp.amplitude.copy()

# fitting oscillatory phase / amplitude to actual SAT
reconstruction = amplitude * np.cos(phase_amp)
fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
m, c = np.linalg.lstsq(fit_x, g_amp.data)[0]
amplitude = m * amplitude + c
print("Oscillatory series fitted to SAT data with coeff. %.3f and intercept %.3f" % (m, c))

start_cut = date(1958,1,1)
g_data.data, g_data.time, idx = g.get_data_of_precise_length('16k', start_cut, None, False)
phase = phase[idx[0] : idx[1]]
amplitude = amplitude[idx[0] : idx[1]]

phase_bins = get_equidistant_bins(BINS)
cond_means = np.zeros((BINS, 2)) # :,0 - SATA, :,1 - SATamp

for i in range(cond_means.shape[0]):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    cond_means[i, 1] = np.mean(amplitude[ndx])
    cond_means[i, 0] = np.mean(g_data.data[ndx])

cond_means_surr = np.zeros((NUM_SURR, BINS, 2))

for su in range(NUM_SURR):
    # MF
    sg_amp.construct_fourier_surrogates(algorithm='FT')
    sg_amp.add_seasonality(mean2, var2, trend2)
    sg.construct_fourier_surrogates(algorithm='FT')
    sg.add_seasonality(mean, var, trend)

    # AR
    # sg_amp.prepare_AR_surrogates()
    # sg_amp.construct_surrogates_with_residuals()
    # sg_amp.add_seasonality(mean2[:-1], var2[:-1], trend2[:-1])
    # sg.prepare_AR_surrogates()
    # sg.construct_surrogates_with_residuals()
    # sg.add_seasonality(mean[:-1], var[:-1], trend[:-1])
    # sg_time = sg.time.copy()
    sg.wavelet(PERIOD, period_unit='y', cut=None)
    phase = sg.phase.copy()

    # sg_amp_time = sg_amp.time.copy()
    sg_amp.wavelet(1, period_unit='y', cut=None)
    amplitude = sg_amp.amplitude.copy()
    phase_amp = sg_amp.phase.copy()

    # fitting oscillatory phase / amplitude to actual SAT
    reconstruction = amplitude * np.cos(phase_amp)
    fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
    m, c = np.linalg.lstsq(fit_x, sg_amp.data)[0]
    amplitude = m * amplitude + c

    _, _, idx = g.get_data_of_precise_length('16k', start_cut, None, False)
    phase = phase[idx[0] : idx[1]]
    amplitude = amplitude[idx[0] : idx[1]]
    sg.data = sg.data[idx[0] : idx[1]]

    for i in range(cond_means_surr.shape[1]):
        ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
        cond_means_surr[su, i, 1] = np.mean(amplitude[ndx])
        cond_means_surr[su, i, 0] = np.mean(sg.data[ndx])

    if (su+1) % 10 == 0:
        print("%d. surrogate done..." % (su + 1))


fig = plt.figure(figsize = (10,8), frameon = False)
gs = gridspec.GridSpec(1, 2)
gs.update(left = 0.12, right = 0.95, top = 0.95, bottom = 0.1, wspace = 0.4, hspace = 0.4)

ax = plt.Subplot(fig, gs[0, 0])
fig.add_subplot(ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(color = '#6A4A3C')
diff = (phase_bins[1]-phase_bins[0])
ax.bar(phase_bins[:-1] + diff*0.05, cond_means[:, 0], width = diff*0.4, bottom = None, fc = '#BF3919', ec = '#BF3919',  figure = fig)
ax.bar(phase_bins[:-1] + diff*0.55, np.mean(cond_means_surr[:, :, 0], axis = 0), width = diff*0.4, bottom = None, fc = '#76C06E', ec = '#76C06E',  figure = fig)
ax.axis([-np.pi, np.pi, -1, 1])
ax.set_xlabel("PHASE [RAD]", size = 20)
ax.set_ylabel("COND. MEAN SATA [$^{\circ}$C]", size = 20)
ax.tick_params(axis = 'both', which = 'major', labelsize = 18)

ax = plt.Subplot(fig, gs[0, 1])
fig.add_subplot(ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(color = '#6A4A3C')
diff = (phase_bins[1]-phase_bins[0])
ax.bar(phase_bins[:-1] + diff*0.05, cond_means[:, 1], width = diff*0.4, bottom = None, fc = '#BF3919', ec = '#BF3919', figure = fig)
ax.bar(phase_bins[:-1] + diff*0.55, np.mean(cond_means_surr[:, :, 1], axis = 0), width = diff*0.4, bottom = None, fc = '#76C06E', ec = '#76C06E',  figure = fig)
ax.axis([-np.pi, np.pi, 19, 22])
ax.set_xlabel("PHASE [RAD]", size = 20)
ax.set_ylabel("COND. MEAN SAT AMP [$^{\circ}$C]", size = 20)
ax.tick_params(axis = 'both', which = 'major', labelsize = 18)

tit = "%s -- %s - %s" % (g.location, g.get_date_from_ndx(0), g.get_date_from_ndx(-1))
# plt.suptitle(tit, size = 22)

fname = "PRGhistSATAvsSATamp1000MF"
plt.savefig(fname + ".eps")