from src import wavelet_analysis
from src.data_class import load_station_data, DataField
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


AMPLITUDE = False
PERIOD = 8
BINS = 8
SEASON = [[12,1,2], [6,7,8]]



g = load_station_data('TG_STAID000027.txt', date(1924,1,15), date(2013,10,1), True)
if AMPLITUDE:
    g_amp = load_station_data('TG_STAID000027.txt', date(1924,1,15), date(2013, 10, 1), False)
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
    period = 1 * 365.25 # frequency of interest
    s0_amp = period / fourier_factor # get scale
    wave, _, _, _ = wavelet_analysis.continous_wavelet(g_amp.data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0_amp, j1 = 0, k0 = k0) # perform wavelet
    amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))
    amplitude = amplitude[0, :]
    print 'data', amplitude.max(), amplitude.min()
    phase_amp = np.arctan2(np.imag(wave), np.real(wave))
    phase_amp = phase_amp[0, :]

    # fitting oscillatory phase / amplitude to actual SAT
    reconstruction = amplitude * np.cos(phase_amp)
    fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
    m, c = np.linalg.lstsq(fit_x, g_amp.data)[0]
    amplitude = m * amplitude + c
    print("Oscillatory series fitted to SAT data with coeff. %.3f and intercept %.3f" % (m, c))

cond_means = np.zeros((BINS,2,2))

def get_equidistant_bins(num):
    return np.array(np.linspace(-np.pi, np.pi, num+1))

start_cut = date(1958,1,1)
g_data.data, g_data.time, idx = g.get_data_of_precise_length('16k', start_cut, None, False)
phase = phase[0, idx[0] : idx[1]]
if AMPLITUDE:
    amplitude = amplitude[idx[0] : idx[1]]

mons = {0: 'J', 1: 'F', 2: 'M', 3: 'A', 4: 'M', 5: 'J', 6: 'J', 7: 'A', 8: 'S', 9: 'O', 10: 'N', 11: 'D'}
# if SEASON != None:
idx = 0
for se in SEASON:
    print("[%s] Only %s season will be evaluated.." % (str(datetime.now()), ''.join([mons[m-1] for m in se])))
    g_seasons = DataField(data = g_data.copy_data(), time = g_data.time.copy())
    phase_seasons = phase.copy()
    ndx_season = g_seasons.select_months(se)
    phase_seasons = phase_seasons[ndx_season]
    if AMPLITUDE:
        amplitude = amplitude[ndx_season]

    phase_bins = get_equidistant_bins(BINS)
    for i in range(cond_means.shape[0]):
        print se, i
        ndx = ((phase_seasons >= phase_bins[i]) & (phase_seasons <= phase_bins[i+1]))
        if AMPLITUDE:
            cond_means[i, 0, idx] = np.mean(amplitude[ndx])
            cond_means[i, 1, idx] = np.std(amplitude[ndx], ddof = 1)
        else:
            cond_means[i, 0, idx] = np.mean(g_seasons.data[ndx])
            cond_means[i, 1, idx] = np.std(g_seasons.data[ndx], ddof = 1)
    idx += 1

phase_bins = get_equidistant_bins(BINS)
fig = plt.figure(figsize = (10,16), frameon = False)
gs = gridspec.GridSpec(2, 2)
gs.update(left = 0.15, right = 0.9, top = 0.9, bottom = 0.1, wspace = 0.4, hspace = 0.4)
for i in range(2):
    ax = plt.Subplot(fig, gs[i, 0])
    fig.add_subplot(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color = '#6A4A3C')
    diff = (phase_bins[1]-phase_bins[0])
    ax.bar(phase_bins[:-1] + diff*0.1, cond_means[:, 0, i], width = diff*0.8, bottom = None, fc = '#BF3919', ec = '#BF3919',  figure = fig)
    if AMPLITUDE:
        pass
    else:
        ax.axis([-np.pi, np.pi, -1.5, 1.5])
        ax.set_xlabel("phase [rad]", size = 16)
        ax.set_ylabel("cond. mean SATA [$^{\circ}$C]", size = 16)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)

    ax = plt.Subplot(fig, gs[i, 1])
    fig.add_subplot(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color = '#6A4A3C')
    diff = (phase_bins[1]-phase_bins[0])
    ax.bar(phase_bins[:-1] + diff*0.1, cond_means[:, 1, i], width = diff*0.8, bottom = None, fc = '#76C06E', ec = '#76C06E', figure = fig)
    if AMPLITUDE:
        pass
    else:
        ax.axis([-np.pi, np.pi, 0, 6])
        ax.set_xlabel("phase [rad]", size = 16)
        ax.set_ylabel("cond. standard deviation SATA [$^{\circ}$C]", size = 16)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
# fig.text(0.5, 0.04, 'phase [rad]', va = 'center', ha = 'center', size = 16)
fig.text(0.5, 0.92, 'DJF', va = 'center', ha = 'center', size = 21)
fig.text(0.5, 0.46, 'JJA', va = 'center', ha = 'center', size = 21)

plt.suptitle("%s -- %s - %s" % (g.location, g.get_date_from_ndx(0), g.get_date_from_ndx(-1)), size = 22)

plt.savefig("grl_fig/PRGhistSeasons.pdf")