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
import matplotlib.gridspec as gridspec
from scipy.stats import itemfreq
import matplotlib.mlab as mlab
import scipy.stats as sts


ANOMALISE = True
PERIOD = 8 # years, period of wavelet
WINDOW_LENGTH = 16384 / 365.25
WINDOW_SHIFT = 1 # years, delta in the sliding window analysis
MEANS = True # if True, compute conditional means, if False, compute conditional variance
NUM_SURR = 100
EVAL_SEASON = False
season = [12,1,2]
s_name = 'DJF'
s_num = 100
AA = False
SURR_TYPE = 'MF'
SURR_DETAIL = False
SURR_ANALYSIS = False
MOMENT = 'mean'

if MOMENT == 'mean':
    func = np.mean
    axplot = [-1.5, 1.5]
elif MOMENT == 'std':
    func = np.std
    axplot = [2.5, 5.5]
elif MOMENT == 'skewness':
    func = sts.skew
    axplot = [-0.5, 1]
elif MOMENT == 'kurtosis':
    func = sts.kurtosis
    axplot = [-1.5, 0.5]

# load data - at least 32k of data because of surrogates
g = load_station_data('TG_STAID000027.txt', date(1924, 1, 1), date(2013,9,18), ANOMALISE)
g_amp = load_station_data('TG_STAID000027.txt', date(1924, 1, 1), date(2013, 9, 18), False)
g_data = DataField()


print("[%s] Wavelet analysis in progress with %d year window shifted by %d year(s)..." % (str(datetime.now()), WINDOW_LENGTH, WINDOW_SHIFT))
k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale 
# wavelet - data    
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
print reconstruction.shape
fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
m, c = np.linalg.lstsq(fit_x, g_amp.data)[0]
amplitude = m * amplitude + c
print("Oscillatory series fitted to SAT data with coeff. %.3f and intercept %.3f" % (m, c))

cond_means = np.zeros((8,))

def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))
    

start_cut = date(1958,1,1)
g_data.data, g_data.time, idx = g.get_data_of_precise_length('16k', start_cut, None, False)
phase = phase[0, idx[0] : idx[1]]
amplitude = amplitude[idx[0] : idx[1]]
# subselect season
if EVAL_SEASON:
    ndx_season = g_data.select_months(season)
    phase = phase[ndx_season]
    amplitude = amplitude[ndx_season]

phase_bins = get_equidistant_bins() # equidistant bins
# data_mom = func(g_data.data)
data_mom = func(amplitude)
for i in range(cond_means.shape[0]):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    # cond_means[i] = func(g_data.data[ndx])
    cond_means[i] = func(amplitude[ndx])
    
    
def plot_surr_analysis(bins_surrs, fname = None):
    fig = plt.figure(figsize = (16,8), frameon = False)
    gs = gridspec.GridSpec(2, 3)
    gs.update(left = 0.05, right = 0.95, top = 0.9, bottom = 0.1, wspace = 0.25, hspace = 0.4)
    plt.suptitle("%s - run of %d %s surrogates -- %s - %s%s" % (g.location, NUM_SURR, SURR_TYPE, str(g_data.get_date_from_ndx(0)), 
                  str(g_data.get_date_from_ndx(-1)), (' -- %s only' % s_name) if EVAL_SEASON else ''), size = 16)
    
    # binning
    ax = plt.Subplot(fig, gs[:, 0])
    fig.add_subplot(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(top = 'off', right = 'off', color = '#6A4A3C')
    diff = (phase_bins[1]-phase_bins[0])
    ax.bar(phase_bins[:-1], cond_means, width = 0.45*diff, bottom = None, fc = '#071E21', ec = '#071E21', label = 'data')
    ax.bar(phase_bins[:-1]+0.5*diff, np.mean(bins_surrs, axis = 0), width = 0.45*diff, bottom = None, fc = '#8F8D0A', ec = '#8F8D0A', label = 'mean of %d surrs' % NUM_SURR)
    ax.axis([-np.pi, np.pi, axplot[0], axplot[1]])
    ax.set_xlabel('phase [rad]')
    ax.set_ylabel('cond. %s temperature [$^{\circ}$C]' % (MOMENT))
    ax.set_title('Cond. %s in bins - surrogate mean' % (MOMENT))
    ax.legend()
    
    meandiff = np.mean([bins_surrs[i,:].max() - bins_surrs[i,:].min() for i in range(cond_means_surr.shape[0])])
    stddiff = np.std([bins_surrs[i,:].max() - bins_surrs[i,:].min() for i in range(cond_means_surr.shape[0])], ddof = 1)
    fig.text(0.5, 0.81, 'DATA: \n difference - %.2f' % (cond_means.max() - cond_means.min()), size = 14, ha = 'center', va = 'center')
    fig.text(0.5, 0.74, 'SURROGATES: \n mean of differences - %.2f' % (meandiff), size = 14, ha = 'center', va = 'center')
    fig.text(0.5, 0.7, 'std of differences - %.2f' % (stddiff), size = 14, ha = 'center', va = 'center')
    fig.text(0.5, 0.675, 'difference of means - %.2f' % (np.mean(bins_surrs, axis = 0).max() - np.mean(bins_surrs, axis = 0).min()), size = 14, ha = 'center', va = 'center')
    fig.text(0.5, 0.65, 'total surrogates needed - %d' % tot, ha = 'center', va = 'center', size = 14)
    
    
    # max
    maxs = itemfreq(np.argmax(bins_surrs, axis = 1))
    max_p = np.zeros((8,))
    for i in range(max_p.shape[0]):
        if i in maxs[:, 0]:
            max_p[i] = maxs[np.where(maxs[:, 0] == i)[0][0], 1]
    ax = plt.Subplot(fig, gs[1, 1])
    fig.add_subplot(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(top = 'off', right = 'off', color = '#6A4A3C')
    diff = (phase_bins[1]-phase_bins[0])
    rects = ax.bar(phase_bins[:-1]+0.1*diff, max_p, width = 0.8*diff, bottom = None, fc = '#D3562C', ec = '#D3562C')
    maximum = max_p.argmax()
    ax.text(rects[maximum].get_x() + rects[maximum].get_width()/2., 0, 
             '%d'%int(rects[maximum].get_height()), ha = 'center', va = 'bottom', color = '#6A4A3C')
    ax.set_xbound(lower = -np.pi, upper = np.pi)
    ax.set_xlabel('phase [rad]')
    ax.set_title('Maximum in bins')
    
    # min
    mins = itemfreq(np.argmin(bins_surrs, axis = 1))
    min_p = np.zeros((8,))
    for i in range(min_p.shape[0]):
        if i in mins[:, 0]:
            min_p[i] = mins[np.where(mins[:, 0] == i)[0][0], 1]
    ax = plt.Subplot(fig, gs[1, 2])
    fig.add_subplot(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(top = 'off', right = 'off', color = '#6A4A3C')
    diff = (phase_bins[1]-phase_bins[0])
    rects = ax.bar(phase_bins[:-1]+0.1*diff, min_p, width = 0.8*diff, bottom = None, fc = '#73B1D1', ec = '#73B1D1')
    maximum = min_p.argmax()
    ax.text(rects[maximum].get_x() + rects[maximum].get_width()/2., 0, 
             '%d'%int(rects[maximum].get_height()), ha = 'center', va = 'bottom', color = '#6A4A3C')
    ax.set_xbound(lower = -np.pi, upper = np.pi)
    ax.set_xlabel('phase [rad]')
    ax.set_title('Minimum in bins')
    
    # hist of diffs
    diffs = []
    for i in range(NUM_SURR):
        diffs.append(bins_surrs[i, :].max() - bins_surrs[i, :].min())
    diffs = np.array(diffs)
    ax = plt.Subplot(fig, gs[0, 2])
    fig.add_subplot(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(top = 'off', right = 'off', color = '#6A4A3C')
    ax.hist(diffs, bins = np.arange(0.5,2,0.1), fc = '#CDCA0D', alpha = 0.75, normed = True)
    ax.plot(np.arange(0.4,2,0.02), mlab.normpdf(np.arange(0.4,2,0.02), meandiff, stddiff), color = '#85A0A4', linewidth = 1.75)
    ax.vlines(meandiff, ymin = 0, ymax = mlab.normpdf(meandiff, meandiff, stddiff), color = '#85A0A4', linestyles = 'dashed', linewidth = 1.2)
    ax.vlines([meandiff + stddiff, meandiff - stddiff], ymin = 0, ymax = mlab.normpdf(meandiff + stddiff, meandiff, stddiff), color = '#85A0A4', linestyles = 'dashed', linewidth = 1.2)
    ax.set_title('Histogram of differences')
    ax.set_xticks(np.arange(0.4,2,0.2))
    ax.set_ybound(lower = 0.)
    
    
    
    if fname != None:
        plt.savefig(fname)
    else:
        plt.show()
    
    
    
cond_means_surr = np.zeros((NUM_SURR, 8))
surr_mom = []

mean, var, trend = g.get_seasonality(True)
mean2, var2, trend2 = g_amp.get_seasonality(True)
su = 0
tot = 0
while su < NUM_SURR:
    # sg = SurrogateField()
    sg_amp = SurrogateField()
    # sg.copy_field(g)
    sg_amp.copy_field(g_amp)
    if SURR_TYPE == 'MF':
        # sg.construct_multifractal_surrogates()
        # sg.add_seasonality(mean, var, trend)
        sg_amp.construct_multifractal_surrogates()
        sg_amp.add_seasonality(mean2, var2, trend2)
    elif SURR_TYPE == 'FT':
        # sg.construct_fourier_surrogates_spatial()
        # sg.add_seasonality(mean, var, trend)
        sg_amp.construct_fourier_surrogates_spatial()
        sg_amp.add_seasonality(mean2, var2, trend2)
    elif SURR_TYPE == 'AR':
        # sg.prepare_AR_surrogates()
        # sg.construct_surrogates_with_residuals()
        # sg.add_seasonality(mean[:-1], var[:-1], trend[:-1])
        sg_amp.prepare_AR_surrogates()
        sg_amp.construct_surrogates_with_residuals()
        sg_amp.add_seasonality(mean2, var2, trend2)
    # wave, _, _, _ = wavelet_analysis.continous_wavelet(sg.surr_data, 1, True, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
    phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes

    wave, _, _, _ = wavelet_analysis.continous_wavelet(sg_amp.surr_data, 1, True, wavelet_analysis.morlet, dj = 0, s0 = s0_amp, j1 = 0, k0 = k0) # perform wavelet
    amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))
    amplitude = amplitude[0, :]
    phase_amp = np.arctan2(np.imag(wave), np.real(wave))
    phase_amp = phase_amp[0, :]

    # fitting oscillatory phase / amplitude to actual SAT
    reconstruction = amplitude * np.cos(phase_amp)
    fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
    m, c = np.linalg.lstsq(fit_x, g_amp.data)[0]
    amplitude = m * amplitude + c
    if AA:
        # sg.amplitude_adjust_surrogates(mean, var, trend)
        sg_amp.amplitude_adjust_surrogates(mean2, var2, trend2)
    _, _, idx = g.get_data_of_precise_length('16k', start_cut, None, False)
    # sg.surr_data = sg.surr_data[idx[0] : idx[1]]
    phase = phase[0, idx[0] : idx[1]]
    amplitude = amplitude[idx[0] : idx[1]]
    
    # subselect season
    if EVAL_SEASON:
        # sg.surr_data = sg.surr_data[ndx_season]
        phase = phase[ndx_season]
        amplitude = amplitude[ndx_season]
    temp_means = np.zeros((8,))
    surr_mom.append(func(amplitude))
    for i in range(cond_means.shape[0]):
        ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
        # temp_means[i] = func(sg.surr_data[ndx])
        temp_means[i] = func(amplitude[ndx])
    ma = temp_means.argmax()
    mi = temp_means.argmin()
    print 'max - ', ma, '  min - ', mi
    if (np.abs(ma - mi) > 2) and (np.abs(ma - mi) < 6):
        cond_means_surr[su, :] = temp_means.copy()
        su += 1
        surr_mom.append(temp_means[ma] - temp_means[mi])
    else:
        pass
    
    tot += 1
        
    if (tot+1) % 10 == 0:
        print tot+1, '. surrogate done...'
    
print("[%s] Wavelet done." % (str(datetime.now())))

if SURR_ANALYSIS:
    plot_surr_analysis(cond_means_surr, fname = 'debug/surr_analysis_%d%ssurrs_condition.png' % (NUM_SURR, SURR_TYPE))

if SURR_DETAIL:
    meandiff = np.mean([cond_means_surr[i,:].max() - cond_means_surr[i,:].min() for i in range(cond_means_surr.shape[0])])
    stddiff = np.std([cond_means_surr[i,:].max() - cond_means_surr[i,:].min() for i in range(cond_means_surr.shape[0])], ddof = 1)
    cnt = 0
    for i in range(cond_means_surr.shape[0]):
        d = cond_means_surr[i, :].max() - cond_means_surr[i, :].min()
        if d > meandiff + stddiff:
           cnt += 1
           plt.figure()
           diff = (phase_bins[1]-phase_bins[0])
           plt.bar(phase_bins[:-1], cond_means, width = 0.45*diff, bottom = None, fc = '#071E21', ec = '#071E21', label = 'data')
           plt.bar(phase_bins[:-1]+0.5*diff, cond_means_surr[i, :], width = 0.45*diff, bottom = None, fc = '#8F8D0A', ec = '#8F8D0A', label = 'realization of %s surr' % SURR_TYPE)
           plt.axis([-np.pi, np.pi, -1., 1.5])
           plt.legend()
           plt.savefig('debug/large_surr_diff/bar%d' % cnt)


diff = (phase_bins[1]-phase_bins[0])
fig = plt.figure(figsize=(6,10))
b1 = plt.bar(phase_bins[:-1], cond_means, width = diff*0.45, bottom = None, fc = '#403A37', figure = fig)
b2 = plt.bar(phase_bins[:-1] + diff*0.5, np.mean(cond_means_surr, axis = 0), width = diff*0.45, bottom = None, fc = '#A09793', figure = fig)
plt.xlabel('phase [rad]')
mean_of_surr_mom = np.mean(np.array(surr_mom))
std_of_surr_mom = np.std(np.array(surr_mom), ddof = 1)
plt.legend( (b1[0], b2[0]), ('data', 'mean of %d surr' % NUM_SURR) )
# plt.ylabel('cond %s temperature [$^{\circ}$C]' % (MOMENT))
plt.ylabel('cond %s SAT amplitude' % (MOMENT))
# plt.axis([-np.pi, np.pi, axplot[0], axplot[1]])
plt.xlim(-np.pi, np.pi)
plt.title('%s cond %s \n difference data: %.2f\n mean of surr diffs: %.2f \n std of surr diffs: %.2f' % (g.location, 
           MOMENT, cond_means.max() - cond_means.min(), mean_of_surr_mom, std_of_surr_mom))
s_name = s_name if EVAL_SEASON else ''
MOMENT_SHRTCT = MOMENT[:4] if len(MOMENT) > 4 else MOMENT
plt.savefig('debug/cond_%s_%s%s%s.png' % (MOMENT_SHRTCT, SURR_TYPE, '_' + s_name if EVAL_SEASON else '', '_amplitude_adjusted_before_phase' if AA else ''))
        
