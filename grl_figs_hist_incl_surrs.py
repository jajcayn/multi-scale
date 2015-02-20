from src import wavelet_analysis
from src.data_class import load_station_data, DataField
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from surrogates.surrogates import SurrogateField
from multiprocessing import Process, Queue


AMPLITUDE = True
PERIOD = 8
BINS = 8
SURR = True
NUM_SURR = 1000
WORKERS = 2
SURR_TYPE = 'FT'


def _reconstruction_surrs(sg, a, jobq, resq, idx):
    mean, var, trend = a

    while jobq.get() is not None:
        if SURR_TYPE == 'MF':
            sg.construct_multifractal_surrogates()
        elif SURR_TYPE == 'FT':
            sg.construct_fourier_surrogates_spatial()
        sg.add_seasonality(mean, var, trend)

        # sg.amplitude_adjust_surrogates(mean, var, trend)

        period = PERIOD * 365.25 # frequency of interest
        s0_amp = period / fourier_factor # get scale
        wave, _, _, _ = wavelet_analysis.continous_wavelet(sg.surr_data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0_amp, j1 = 0, k0 = k0) # perform wavelet
        amplitude2 = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))
        amplitude2 = amplitude2[0, :]
        phase_amp = np.arctan2(np.imag(wave), np.real(wave))
        phase_amp = phase_amp[0, :]

        reconstruction = amplitude2 * np.cos(phase_amp)
        fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
        m, c = np.linalg.lstsq(fit_x, sg.surr_data)[0]
        amplitude2 = m * amplitude2 + c
        # amp_to_plot_surr = amplitude2.copy()
        # amplitude2 = m * reconstruction + c

        amplitude2 = amplitude2[idx[0] : idx[1]]
        # amp_to_plot_surr = amp_to_plot_surr[idx[0] : idx[1]]
        phase_amp = phase_amp[idx[0] : idx[1]]
        sg.surr_data =  sg.surr_data[idx[0] : idx[1]]

        cond_temp = np.zeros((BINS,2))
        for i in range(cond_means.shape[0]):
            ndx = ((phase_amp >= phase_bins[i]) & (phase_amp <= phase_bins[i+1]))
            cond_temp[i,1] = np.mean(sg.surr_data[ndx])
        data_diff = cond_temp[:, 1].max() - cond_temp[:, 1].min()

        resq.put([data_diff, np.mean(amplitude2)])


g = load_station_data('TG_STAID000027.txt', date(1958, 1, 1), date(2013, 11, 10), True)
g_amp = load_station_data('TG_STAID000027.txt', date(1958, 1, 1), date(2013, 11, 10), True)
g_data = DataField()


k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale 
# wavelet - data    


wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes

if AMPLITUDE:
    period = 8 * 365.25 # frequency of interest
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


cond_means = np.zeros((BINS, 2, 1))


def get_equidistant_bins(num):
    return np.array(np.linspace(-np.pi, np.pi, num+1))

start_cut = date(1962,1,1)
l = 17532

g_data.data, g_data.time, idx = g.get_data_of_precise_length(l, start_cut, None, False) # 16k
print g_data.get_date_from_ndx(0), g_data.get_date_from_ndx(-1)
phase = phase[0, idx[0] : idx[1]]
if AMPLITUDE:
    amplitude = amplitude[idx[0] : idx[1]]

phase_bins = get_equidistant_bins(BINS)

for i in range(cond_means.shape[0]):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    cond_means[i, 0, 0] = np.mean(g_data.data[ndx])
    cond_means[i, 1, 0] = np.std(g_data.data[ndx], ddof = 1)

data_diff = cond_means[:, 0, 0].max() - cond_means[:, 0, 0].min()


if SURR:
    surr_diff = np.zeros((NUM_SURR,))
    amp_surr = np.zeros((NUM_SURR,))
    surr_completed = 0
    jobQ = Queue()
    resQ = Queue()
    for i in range(NUM_SURR):
        jobQ.put(1)
    for i in range(WORKERS):
        jobQ.put(None)
    a = g_amp.get_seasonality(True)
    sg = SurrogateField()
    sg.copy_field(g_amp)
    phase_bins = get_equidistant_bins(BINS)
    workers = [Process(target = _reconstruction_surrs, args = (sg, a, jobQ, resQ, idx)) for iota in range(WORKERS)]
    for w in workers:
        w.start()

    while surr_completed < NUM_SURR:
        surr_means = resQ.get()
        surr_diff[surr_completed] = surr_means[0]
        amp_surr[surr_completed] = surr_means[1]

        surr_completed += 1

        if (surr_completed % 100) == 0:
            print("%d. surrogate done..." % surr_completed)

    for w in workers:
        w.join()


fig = plt.figure(figsize = (10,16), frameon = False)
gs = gridspec.GridSpec(2, 2)
gs.update(left = 0.12, right = 0.95, top = 0.95, bottom = 0.1, wspace = 0.4, hspace = 0.4)

ax1 = plt.Subplot(fig, gs[0, 0])
fig.add_subplot(ax1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.tick_params(color = '#6A4A3C')
diff = (phase_bins[1]-phase_bins[0])
ax1.bar(phase_bins[:-1] + diff*0.1, cond_means[:, 0, 0], width = diff*0.8, bottom = None, fc = '#BF3919', ec = '#BF3919',  figure = fig)
ax1.axis([-np.pi, np.pi, -1, 1])
ax1.set_xlabel("PHASE [RAD]", size = 20)
ax1.set_ylabel("COND. MEAN SATA [$^{\circ}$C]", size = 20)
ax1.tick_params(axis = 'both', which = 'major', labelsize = 18)

ax2 = plt.Subplot(fig, gs[1, 0])
fig.add_subplot(ax2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(color = '#6A4A3C')
diff = (phase_bins[1]-phase_bins[0])
ax2.bar(phase_bins[:-1] + diff*0.1, cond_means[:, 1, 0], width = diff*0.8, bottom = None, fc = '#76C06E', ec = '#76C06E', figure = fig)
ax2.axis([-np.pi, np.pi, 2, 4])
ax2.set_xlabel("PHASE [RAD]", size = 20)
ax2.set_ylabel("COND. STANDARD DEVIATION SATA [$^{\circ}$C]", size = 20)
ax2.tick_params(axis = 'both', which = 'major', labelsize = 18)
# fig.text(0.5, 0.04, 'phase [rad]', va = 'center', ha = 'center', size = 16)

ax3 = plt.Subplot(fig, gs[0, 1])
fig.add_subplot(ax3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.tick_params(color = '#6A4A3C') 
n, bins, patch = ax3.hist(surr_diff, 50, histtype = 'stepfilled')
print surr_diff.min(), surr_diff.max(), data_diff
plt.setp(patch, 'facecolor', '#BF3919', 'edgecolor', '#BF3919', 'alpha', 0.9)
ax3.vlines(data_diff, 0, n.max(), color = "#76C06E", linewidth = 3.5)
ax3.axis([0.3, 1.8, 0, 80])
ax3.set_xlabel("DIFFERENCE [$^{\circ}$C]", size = 20)
ax3.tick_params(axis = 'both', which = 'major', labelsize = 18)
p_val_diff = 1. - float(np.sum(np.greater(data_diff, surr_diff))) / NUM_SURR


ax4 = plt.Subplot(fig, gs[1, 1])
fig.add_subplot(ax4)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.tick_params(color = '#6A4A3C') 
n, bins, patch = ax4.hist(amp_surr, 50, histtype = 'stepfilled')
print amp_surr.min(), amp_surr.max(), np.mean(amplitude)
plt.setp(patch, 'facecolor', '#BF3919', 'edgecolor', '#BF3919', 'alpha', 0.9)
ax4.vlines(np.mean(amplitude), 0, n.max(), color = "#76C06E", linewidth = 3.5)
ax4.axis([0.3, 1.8, 0, 60])
ax4.set_xlabel("MEAN AMPLITUDE [$^{\circ}$C]", size = 20)
# ax4.set_xticks(np.arange(0, 1, 0.2))
ax4.tick_params(axis = 'both', which = 'major', labelsize = 18)
p_val_amp = 1. - float(np.sum(np.greater(np.mean(amplitude), amp_surr))) / NUM_SURR

plt.savefig("debug/PRGhist_diff_%.2f_amp_%.2f.eps" % (p_val_diff, p_val_amp))