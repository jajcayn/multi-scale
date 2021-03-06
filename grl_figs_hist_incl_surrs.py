from pyclits.geofield import DataField
from pyclits.data_loaders import load_station_data
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyclits.surrogates import SurrogateField
from multiprocessing import Process, Queue

font = {'fontname' : 'Courier New'}
AMPLITUDE = True
PERIOD = 8
BINS = 8
SURR = True
NUM_SURR = 5000
WORKERS = 2
SURR_TYPE = 'FT'


def _reconstruction_surrs(sg, a, jobq, resq, idx):
    mean, var, trend = a

    while jobq.get() is not None:
        if SURR_TYPE == 'MF':
            sg.construct_multifractal_surrogates()
        elif SURR_TYPE == 'FT':
            sg.construct_fourier_surrogates(algorithm='FT')
        sg.add_seasonality(mean, var, trend)

        # sg.amplitude_adjust_surrogates(mean, var, trend)

        sg.wavelet(PERIOD, period_unit='y', cut=None)
        amplitude2 = sg.amplitude.copy()
        phase_amp = sg.phase.copy()

        reconstruction = amplitude2 * np.cos(phase_amp)
        fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
        m, c = np.linalg.lstsq(fit_x, sg.data)[0]
        amplitude2 = m * amplitude2 + c
        # amp_to_plot_surr = amplitude2.copy()
        # amplitude2 = m * reconstruction + c

        amplitude2 = amplitude2[idx[0] : idx[1]]
        # amp_to_plot_surr = amp_to_plot_surr[idx[0] : idx[1]]
        phase_amp = phase_amp[idx[0] : idx[1]]
        sg.data =  sg.data[idx[0] : idx[1]]

        cond_temp = np.zeros((BINS,2))
        for i in range(cond_means.shape[0]):
            ndx = ((phase_amp >= phase_bins[i]) & (phase_amp <= phase_bins[i+1]))
            cond_temp[i,1] = np.mean(sg.data[ndx])
        data_diff = cond_temp[:, 1].max() - cond_temp[:, 1].min()

        resq.put([data_diff, np.mean(amplitude2)])


g = load_station_data('../data/ECAstation-TG/TG_STAID000027.txt', date(1958, 1, 1), date(2013, 11, 10), True)
g_amp = load_station_data('../data/ECAstation-TG/TG_STAID000027.txt', date(1958, 1, 1), date(2013, 11, 10), True)
g_data = DataField()

g.wavelet(PERIOD, period_unit='y', cut=None)
phase = g.phase.copy()

if AMPLITUDE:
    g_amp.wavelet(8, period_unit='y', cut=None)
    amplitude = g_amp.amplitude.copy()
    phase_amp = g_amp.phase.copy()

    # fitting oscillatory phase / amplitude to actual SAT
    reconstruction = amplitude * np.cos(phase_amp)
    fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
    m, c = np.linalg.lstsq(fit_x, g_amp.data)[0]
    amplitude = m * amplitude + c
    print("Oscillatory series fitted to SAT data with coeff. %.3f and intercept %.3f" % (m, c))


cond_means = np.zeros((BINS, 2, 1))


def get_equidistant_bins(num):
    return np.array(np.linspace(-np.pi, np.pi, num+1))

# start_cut = date(1962,1,1)
start_cut = date(1958,1,1)
# l = 17532
l = '16k'

g_data.data, g_data.time, idx = g.get_data_of_precise_length(l, start_cut, None, False) # 16k
print g_data.get_date_from_ndx(0), g_data.get_date_from_ndx(-1)
phase = phase[idx[0] : idx[1]]
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


# fig = plt.figure(figsize = (10,16), frameon = False)
# gs = gridspec.GridSpec(2, 2)
# gs.update(left = 0.12, right = 0.95, top = 0.95, bottom = 0.1, wspace = 0.4, hspace = 0.4)

# ax1 = plt.Subplot(fig, gs[0, 0])
# fig.add_subplot(ax1)
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['left'].set_visible(False)
# ax1.tick_params(color = '#6A4A3C')
# diff = (phase_bins[1]-phase_bins[0])
# ax1.bar(phase_bins[:-1] + diff*0.1, cond_means[:, 0, 0], width = diff*0.8, bottom = None, fc = '#BF3919', ec = '#BF3919',  figure = fig)
# ax1.axis([-np.pi, np.pi, -1, 1])
# ax1.set_xlabel("PHASE [RAD]", size = 20)
# ax1.set_ylabel("COND. MEAN SATA [$^{\circ}$C]", size = 20)
# ax1.tick_params(axis = 'both', which = 'major', labelsize = 18)

# ax2 = plt.Subplot(fig, gs[1, 0])
# fig.add_subplot(ax2)
# ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['left'].set_visible(False)
# ax2.tick_params(color = '#6A4A3C')
# diff = (phase_bins[1]-phase_bins[0])
# ax2.bar(phase_bins[:-1] + diff*0.1, cond_means[:, 1, 0], width = diff*0.8, bottom = None, fc = '#76C06E', ec = '#76C06E', figure = fig)
# ax2.axis([-np.pi, np.pi, 2, 4])
# ax2.set_xlabel("PHASE [RAD]", size = 20)
# ax2.set_ylabel("COND. STANDARD DEVIATION SATA [$^{\circ}$C]", size = 20)
# ax2.tick_params(axis = 'both', which = 'major', labelsize = 18)
# fig.text(0.5, 0.04, 'phase [rad]', va = 'center', ha = 'center', size = 16)

# ax3 = plt.Subplot(fig, gs[0, 1])
# fig.add_subplot(ax3)
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().tick_params(color = '#6A4A3C') 
weights = np.ones_like(surr_diff)/float(len(surr_diff))
n, bins, patch = plt.gca().hist(surr_diff, 50, histtype = 'bar', weights=weights, rwidth=0.9)
print surr_diff.min(), surr_diff.max(), data_diff
plt.setp(patch, 'facecolor', '#777777', 'edgecolor', '#777777', 'alpha', 0.9)
plt.gca().vlines(data_diff, 0, n.max(), color = "k", linewidth = 3.5)
# ax3.axis([0.3, 1.8, 0, 80])
plt.gca().set_xlim(0.3, 1.9)
plt.gca().set_xlabel("DIFF SATA COND MEAN [$^{\circ}$C]", size = 22, **font)
plt.gca().tick_params(axis = 'both', which = 'major', labelsize = 13)
p_val_diff = 1. - float(np.sum(np.greater(data_diff, surr_diff))) / NUM_SURR
plt.text(data_diff*1.3, n.max()*(3/4.), "p=%.3f" % (p_val_diff), size=25, **font)
plt.savefig("PRG_hist_diff.eps", bbox_inches='tight')

# ax4 = plt.Subplot(fig, gs[1, 1])
# fig.add_subplot(ax4)
plt.figure()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().tick_params(color = '#6A4A3C') 
weights = np.ones_like(amp_surr)/float(len(amp_surr))
n, bins, patch = plt.gca().hist(amp_surr, 50, histtype = 'bar', weights=weights, rwidth=0.9)
print amp_surr.min(), amp_surr.max(), np.mean(amplitude)
plt.setp(patch, 'facecolor', '#777777', 'edgecolor', '#777777', 'alpha', 0.9)
plt.gca().vlines(np.mean(amplitude), 0, n.max(), color = "k", linewidth = 3.5)
# plt.gca().axis([0.3, 1.8, 0, 60])
plt.gca().set_xlim(0.3, 1.9)
plt.gca().set_xlabel("AMP 8-YEAR CYCLE [$^{\circ}$C]", size = 22, **font)
# plt.gca().set_xticks(np.arange(0, 1, 0.2))
plt.gca().tick_params(axis = 'both', which = 'major', labelsize = 13)
p_val_amp = 1. - float(np.sum(np.greater(np.mean(amplitude), amp_surr))) / NUM_SURR
plt.text(np.mean(amplitude)*1.3, n.max()*(3/4.), "p=%.3f" % (p_val_amp), size=25, **font)
plt.savefig("PRG_hist_amp.eps", bbox_inches='tight')

# to_txt = np.zeros((cond_means.shape[0], 3))
# to_txt[:, 0] = np.arange(1, 9, 1)
# to_txt[:, 1] = cond_means[:, 0, 0]
# to_txt[:, 2] = cond_means[:, 1, 0]

# np.savetxt('grl_fig/hist_SATA.txt', to_txt, fmt = '%.3f')

# to_txt2 = np.zeros((NUM_SURR + 1, 2))
# to_txt2[:-1, 0] = surr_diff
# to_txt2[-1, 0] = data_diff
# to_txt2[:-1, 1] = amp_surr
# to_txt2[-1, 1] = np.mean(amplitude)

# np.savetxt('grl_fig/hist_surrSATAdiff_%.2f_amp_%.2f.txt' % (p_val_diff, p_val_amp), to_txt2, fmt = '%.3f')