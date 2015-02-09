from src import wavelet_analysis
from src.data_class import load_station_data, DataField
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from surrogates.surrogates import SurrogateField
from multiprocessing import Process, Queue
import matplotlib.gridspec as gridspec


PERIOD = 8
AMP_PERIOD = 8
BINS = 8
ANOMALISE = True # amplitude from SAT / SATA
SURR = True
NUM_SURR = 1000
WORKERS = 3
SURR_TYPE = 'FT'

# 65k
# g = load_station_data('TG_STAID000027.txt', date(1834,4,27), date(2013,10,1), True)
# g_amp = load_station_data('TG_STAID000027.txt', date(1834,4,27), date(2013,10,1), ANOMALISE)

# 32k
g = load_station_data('TG_STAID000027.txt', date(1958, 1, 1), date(2002, 11, 10), True) # date(1924,1,14), date(2013,10,1)
g_amp = load_station_data('TG_STAID000027.txt', date(1958, 1, 1), date(2002, 11, 10), ANOMALISE)
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

def get_equidistant_bins(num):
    return np.array(np.linspace(-np.pi, np.pi, num+1))


def _reconstruction_surrs(sg, a, jobq, resq, idx):
    mean, var, trend = a

    while jobq.get() is not None:
        if SURR_TYPE == 'MF':
            sg.construct_multifractal_surrogates()
        elif SURR_TYPE == 'FT':
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

        amplitude2 = amplitude2[idx[0] : idx[1]]
        phase_amp = phase_amp[idx[0] : idx[1]]
        sg.surr_data =  sg.surr_data[idx[0] : idx[1]]

        cond_temp = np.zeros((BINS,2))
        for i in range(cond_means.shape[0]):
            ndx = ((phase_amp >= phase_bins[i]) & (phase_amp <= phase_bins[i+1]))
            cond_temp[i,0] = np.mean(amplitude2[ndx])
            cond_temp[i,1] = np.mean(sg.surr_data[ndx])
        amp_diff = cond_temp[:, 0].max() - cond_temp[:, 0].min()
        data_diff = cond_temp[:, 1].max() - cond_temp[:, 1].min()

        resq.put([cond_temp, amp_diff, data_diff, np.mean(amplitude2)])



# plt.figure(figsize=(20,10))
# plt.plot(amplitude, color = '#867628', linewidth = 2)
# plt.plot(g_amp.data, color = '#004739', linewidth = 1)
# plt.show()

cond_means = np.zeros((BINS, 2))

start_cut = date(1962,1,1) # 1958, 1, 1
l = int(16384 - 8*y)
g_data.data, g_data.time, idx = g.get_data_of_precise_length(l, start_cut, None, False) # 16k
phase = phase[0, idx[0] : idx[1]]
amplitude = amplitude[idx[0] : idx[1]]


phase_bins = get_equidistant_bins(BINS)

for i in range(cond_means.shape[0]):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    cond_means[i, 0] = np.mean(amplitude[ndx])
    cond_means[i, 1] = np.mean(g_data.data[ndx])
    # cond_means[i] = np.mean(g_data.data[ndx])
    # if SURR:
    #     cond_means[i, 1] = np.mean(amplitude2[ndx])
    # else:
    #     cond_means[i, 1] = np.mean(g_data.data[ndx])

amp_diff = cond_means[:, 0].max() - cond_means[:, 0].min()
data_diff = cond_means[:, 1].max() - cond_means[:, 1].min()


if SURR:
    cond_means_surr = np.zeros((NUM_SURR, BINS, 2))
    amp_diff_surr = np.zeros((NUM_SURR,))
    surr_diff_surr = np.zeros((NUM_SURR,))
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
        cond_means_surr[surr_completed, :, 0] = surr_means[0][:, 0]
        cond_means_surr[surr_completed, :, 1] = surr_means[0][:, 1]
        amp_diff_surr[surr_completed] = surr_means[1]
        surr_diff_surr[surr_completed] = surr_means[2]
        amp_surr[surr_completed] = surr_means[3]


        surr_completed += 1

        if (surr_completed % 100) == 0:
            print("%d. surrogate done..." % surr_completed)

    for w in workers:
        w.join()


num = 5

diff = (phase_bins[1]-phase_bins[0])
fig = plt.figure(figsize=(6,10))
# b1 = plt.bar(phase_bins[:-1] + diff*0.05, cond_means, width = diff*0.4, bottom = None, fc = '#867628', ec = '#867628', figure = fig)
b1 = plt.bar(phase_bins[:-1] + diff*0.05, np.mean(cond_means_surr[:, :, 1], axis = 0), width = diff*0.4, bottom = None, fc = '#867628', ec = '#867628', figure = fig)
plt.bar(phase_bins[:-1] + diff*0.2, cond_means[:, 1], width = 0.1*diff, bottom = None, fc = '#4A81B9', ec = '#4A81B9', figure = fig)
b2 = plt.bar(phase_bins[:-1] + diff*0.55, np.mean(cond_means_surr[:, :, 0], axis = 0), width = diff*0.4, bottom = None, fc = '#004739', ec = '#004739', figure = fig)
plt.bar(phase_bins[:-1] + diff*0.7, cond_means[:, 0], width = 0.1*diff, bottom = None, fc = '#7A0C15', ec = '#7A0C15', figure = fig)
plt.xlabel('phase [rad]')
if SURR:
    pass
    # plt.legend([b1[0], b2[0]], ['%s $A \cos{\phi}$' % ('SATA' if ANOMALISE else 'SAT'), '%s $A \cos{\phi}$%s' % ('SATA' if ANOMALISE else 'SAT',  ' - MF surr' if SURR else '')])
    # plt.legend([b1[0], b2[0]], ['%s $A \cos{\phi}$' % ('SATA' if ANOMALISE else 'SAT'), '%s $A \cos{\phi}$%s' % ('SATA' if ANOMALISE else 'SAT',  ' - %d%s surr' % (NUM_SURR, SURR_TYPE) if SURR else '')])
    plt.legend([b1[0], b2[0]], ['%s%s' % ('SATA' if ANOMALISE else 'SAT', ' - %d%s surr' % (NUM_SURR, SURR_TYPE) if SURR else ''), '%s $A \cos{\phi}$%s' % ('SATA' if ANOMALISE else 'SAT',  ' - %d%s surr' % (NUM_SURR, SURR_TYPE) if SURR else '')])
else:
    plt.legend([b1[0], b2[0]], ['%s $A \cos{\phi}$' % ('SATA' if ANOMALISE else 'SAT'), '%s' % ('SATA' if ANOMALISE else 'SAT')])
plt.ylabel('cond mean %s' % ('SATA' if ANOMALISE else 'SAT'))
plt.axis([-np.pi, np.pi, -0.75, 1])
plt.yticks(np.arange(-0.75,1.25,0.25), np.arange(-1,1.75,0.25))
p_val_overall = 1. - float(np.sum(np.greater(data_diff, surr_diff_surr))) / NUM_SURR
print 'overall', np.sum(np.greater(data_diff, surr_diff_surr))
p_val_amp = 1. - float(np.sum(np.greater(amp_diff, amp_diff_surr))) / NUM_SURR
print np.sum(np.greater(amp_diff, amp_diff_surr))
plt.title('PRG %s %d-year $A \cos{\phi}$ \n %s -- %s \n SATA: %.2f  $A \cos{\phi}$: %.2f' % ('SATA' if ANOMALISE else 'SAT', AMP_PERIOD, str(g_data.get_date_from_ndx(0)), str(g_data.get_date_from_ndx(-1)), p_val_overall, p_val_amp))
plt.savefig('debug/PRG_%s16to14reconstruction%s%d.png' % ('SATA' if ANOMALISE else 'SAT', '%d%ssurr' % (NUM_SURR, SURR_TYPE) if SURR else '', num))

fig = plt.figure(figsize=(10,12))
gs = gridspec.GridSpec(2, 1)
gs.update(left = 0.05, right = 0.95, top = 0.95, bottom = 0.05, hspace = 0.4)
hist_plot = [surr_diff_surr, amp_diff_surr]
vl = [data_diff, amp_diff]
colors = ["#867628", "#004739"]
colors_data = ["#4A81B9", "#7A0C15"]
titl = ['%s%s differences - %.2f' % ('SATA' if ANOMALISE else 'SAT', ' - %d%s surr' % (NUM_SURR, SURR_TYPE) if SURR else '', p_val_overall), '%s $A \cos{\phi}$%s differences - %.2f' % ('SATA' if ANOMALISE else 'SAT',  ' - %d%s surr' % (NUM_SURR, SURR_TYPE) if SURR else '', p_val_amp)]
for i in range(2):
    ax = plt.subplot(gs[i, 0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color = '#6A4A3C')
    n, bins, patch = ax.hist(hist_plot[i], 50, histtype = 'stepfilled')
    ax.vlines(vl[i], 0, n.max(), color = colors_data[i], linewidth = 5)
    plt.setp(patch, 'facecolor', colors[i], 'edgecolor', colors[i], 'alpha', 0.9)
    plt.title(titl[i])
plt.savefig('debug/PRG_%s16to14reconstruction%s_hist%d.png' % ('SATA' if ANOMALISE else 'SAT', '%d%ssurr' % (NUM_SURR, SURR_TYPE) if SURR else '', num))

fig = plt.figure()
n, bins, patch = plt.hist(amp_surr, 50, histtype = 'stepfilled')
plt.setp(patch, 'facecolor', '#867628', 'edgecolor', '#867628', 'alpha', 0.9)
plt.vlines(np.mean(amplitude), 0, n.max(), color = "#4A81B9", linewidth = 5)
plt.savefig("debug/PRG_amp_means_FT5.png")


# draw A*cos fi 1-year vs. 8-year