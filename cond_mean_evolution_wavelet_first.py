
from src import wavelet_analysis
from src.data_class import load_station_data
from surrogates.surrogates import SurrogateField
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue


def render(diffs, meanvars, stds = None, subtit = '', percentil = None, phase = None, fname = None):
    fig, ax1 = plt.subplots(figsize=(11,8))
    if len(diffs) > 3:
        ax1.plot(diffs, color = '#403A37', linewidth = 2, figure = fig)
    else:
        p2, = ax1.plot(diffs[1], color = '#899591', linewidth = 1.5, figure = fig)
        if stds is not None:
            ax1.plot(diffs[1] + stds[0], color = '#899591', linewidth = 0.7, figure = fig)
            ax1.plot(diffs[1] - stds[0], color = '#899591', linewidth = 0.7, figure = fig)
            ax1.fill_between(np.arange(0,diffs[1].shape[0],1), diffs[1] + stds[0], diffs[1] - stds[0], 
                             facecolor = "#899591", alpha = 0.5)
        p1, = ax1.plot(diffs[0], color = '#403A37', linewidth = 2, figure = fig)
        if percentil != None:
            for pos in np.where(percentil[:, 0] == True)[0]:
                ax1.plot(pos, diffs[0][pos], 'o', markersize = 8, color = '#403A37')
    #ax1.plot(total_diffs[0], np.arange(0,len(total_diffs[0])), total_diffs[1], np.arange(0, cnt))
    ax1.axis([0, cnt-1, diff_ax[0], diff_ax[1]])
    ax1.set_xlabel('middle year of %.2f-year wide window' % (WINDOW_LENGTH / 365.25), size = 14)
    if MEANS:
        ax1.set_ylabel('difference in cond mean in temperature [$^{\circ}$C]', size = 14)
    elif not MEANS:
        ax1.set_ylabel('difference in cond variance in temperature [$^{\circ}$C$^2$]', size = 14)
#    year_diff = np.round((last_mid_year - first_mid_year) / 10)
#    print last_mid_year, first_mid_year, year_diff
#    xnames = np.arange(first_mid_year, last_mid_year, year_diff)
#    print xnames
#    plt.xticks(np.linspace(0, cnt, len(xnames)), xnames, rotation = 30)
    plt.xticks(np.arange(0, cnt+8, 8), np.arange(first_mid_year, last_mid_year+8, 8), rotation = 30)
    if not PLOT_PHASE:
        ax2 = ax1.twinx()
        if len(meanvars) > 3:
            ax2.plot(meanvars, color = '#CA4F17', linewidth = 2, figure = fig) # color = '#CA4F17'
        else:
            p4, = ax2.plot(meanvars[1], color = '#64C4A0', linewidth = 1.5, figure = fig)
            if stds is not None:
                ax2.plot(meanvars[1] + stds[1], color = '#64C4A0', linewidth = 0.7, figure = fig)
                ax2.plot(meanvars[1] - stds[1], color = '#64C4A0', linewidth = 0.7, figure = fig)
                ax2.fill_between(np.arange(0,diffs[1].shape[0],1), meanvars[1] + stds[1], meanvars[1] - stds[1],
                                 facecolor = "#64C4A0", alpha = 0.5)
            p3, = ax2.plot(meanvars[0], color = '#CA4F17', linewidth = 2, figure = fig)
            if percentil != None:
                for pos in np.where(percentil[:, 1] == True)[0]:
                    ax2.plot(pos, meanvars[0][pos], 'o', markersize = 8, color = '#CA4F17')
        if MEANS:
            ax2.set_ylabel('mean of cond means in temperature [$^{\circ}$C]', size = 14)
        elif not MEANS:
            ax2.set_ylabel('mean of cond variance in temperature [$^{\circ}$C$^2$]', size = 14)
        ax2.axis([0, cnt-1, mean_ax[0], mean_ax[1]])
        for tl in ax2.get_yticklabels():
            tl.set_color('#CA4F17')
        if len(diffs) < 3:
            plt.legend([p1, p2, p3, p4], ["difference DATA", "difference SURROGATE mean", "mean DATA", "mean SURROGATE mean"], loc = 2)

    elif PLOT_PHASE:
        ax2 = ax1.twinx().twiny()
        p3, = ax2.plot(phase, color = '#CA4F17', linewidth = 1.25, figure = fig)
        ax2.set_ylabel('phase of wavelet in window [rad]', size = 14)
        ax2.axis([0, phase.shape[0], -2*np.pi, 2*np.pi])
        for tl in ax2.get_yticklabels():
            tl.set_color('#CA4F17')
        for tl in ax2.get_xticklabels():
            tl.set_color('#CA4F17')
        plt.legend([p1, p2, p3], ["difference DATA", "difference SURROGATE mean", "phase DATA"], loc = 2)
    tit = 'SURR: Evolution of difference in cond'
    if MEANS:
        tit += ' mean in temp, '
    else:
        tit += ' variance in temp, '
    if not ANOMALISE:
        tit += 'SAT, '
    else:
        tit += 'SATA, '
    if np.int(WINDOW_LENGTH) == WINDOW_LENGTH:
        tit += ('%d-year window, %d-year shift' % (WINDOW_LENGTH, WINDOW_SHIFT))
    else:
        tit += ('%.2f-year window, %d-year shift' % (WINDOW_LENGTH, WINDOW_SHIFT))
    #plt.title(tit)
    if MEANS:
        tit = ('Evolution of difference in cond means temp SATA -- %s \n' % g.location)
    else:
        tit = ('Evolution of difference in cond variance in temp SATA -- %s \n' % g.location)
    tit += subtit
    plt.text(0.5, 1.05, tit, horizontalalignment = 'center', size = 16, transform = ax2.transAxes)
    #ax2.set_xticks(np.arange(start_date.year, end_date.year, 20))
    
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
        
    
def render_phase_and_bins(bins, cond_means, cond_means_surr, phase, dates, percentil = False, subtit = '', fname = None):
    diff = (bins[1]-bins[0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,16))
    b1 = ax1.bar(bins[:-1], cond_means, width = diff*0.45, bottom = None, fc = '#403A37', figure = fig)
    b2 = ax1.bar(bins[:-1] + diff*0.5, np.mean(cond_means_surr, axis = 0), width = diff*0.45, bottom = None, fc = '#A09793', figure = fig)
    ax1.set_xlabel('phase [rad]', size = 14)
    mean_of_diffs = np.mean([cond_means_surr[i,:].max() - cond_means_surr[i,:].min() for i in range(cond_means_surr.shape[0])])
    std_of_diffs = np.std([cond_means_surr[i,:].max() - cond_means_surr[i,:].min() for i in range(cond_means_surr.shape[0])], ddof = 1)
    ax1.legend( (b1[0], b2[0]), ('data', 'mean of %d surr' % NUM_SURR) )
    if MEANS:
        ax1.set_ylabel('cond means temperature [$^{\circ}$C]', size = 14)
        ax1.axis([-np.pi, np.pi, -1.5, 1.5])
    else:
        ax1.set_ylabel('cond variance temperature [$^{\circ}$C$^2$]', size = 14)   
        ax1.axis([-np.pi, np.pi, 5, 25])
    ax1.set_title('%s - cond %s \n surr: %.2f$^{\circ}$C (%.2f$^{\circ}$C$^2$)' % (g.location, 'means' if MEANS else 'var',
              mean_of_diffs, std_of_diffs), size = 16)
    
    ax2.plot(phase, color = '#CA4F17', linewidth = 1.25, figure = fig)
    ax2.set_ylabel('phase [rad]', size = 14)
    ax2.axis([0, phase.shape[0], -np.pi, np.pi])
    ax2.set_xlabel('time [days]')
    ax2.set_title('Phase of the wavelet in window', size = 16)
    
    plt.suptitle('%s window: %s -- %s \n difference data: %.2f$^{\circ}$C -- 95percentil: %s' % ('32/16k' if WINDOW_LENGTH > 16000 else '16/14k',
                 str(dates[0]), str(dates[1]), cond_means.max() - cond_means.min(), percentil), size = 18)
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
        
        
        
        
ANOMALISE = True
PERIOD = 8 # years, period of wavelet
WINDOW_LENGTH = 16384
WINDOW_SHIFT = 1 # years, delta in the sliding window analysis
MEANS = True # if True, compute conditional means, if False, compute conditional variance
WORKERS = 16
NUM_SURR = 100 # how many surrs will be used to evaluate
SURR_TYPE = 'AR' # MF, FT, AR
diff_ax = (0, 2) # means -> 0, 2, var -> 1, 8
mean_ax = (18, 22) # means -> -1, 1.5, var -> 9, 18
PLOT_PHASE = False
PHASE_ANALYSIS_YEAR = None # year of detailed analysis - phase and bins, or None
AMPLITUDE = True


## loading data
g = load_station_data('TG_STAID000027.txt', date(1834,7,28), date(2014,1,1), ANOMALISE)
sg = SurrogateField()
if AMPLITUDE:
    g_amp = load_station_data('TG_STAID000027.txt', date(1834,7,28), date(2014, 1, 1), False)
    sg_amp = SurrogateField()



print("[%s] Wavelet analysis in progress with %d year window shifted by %d year(s)..." % (str(datetime.now()), WINDOW_LENGTH, WINDOW_SHIFT))
k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale 

cond_means = np.zeros((8,))

def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))
    
wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes

if AMPLITUDE:
    s0_amp = (1 * y) / fourier_factor
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

mean, var, trend = g.get_seasonality(True)
sg.copy_field(g)
g.return_seasonality(mean, var, trend)
if AMPLITUDE:
    mean2, var2, trend2 = g_amp.get_seasonality(True)
    sg_amp.copy_field(g_amp)
    g_amp.return_seasonality(mean2, var2, trend2)

main_cut_ndx = g.select_date(date(1838,7,28), date(2010,1,1))
y1 = 1838
phase = phase[0, main_cut_ndx]
if AMPLITUDE:
    amplitude = amplitude[main_cut_ndx]

difference_data = []
meanvar_data = []

cnt = 0

start = 0
end = WINDOW_LENGTH
plot_vars = []

first_mid_year = date.fromordinal(g.time[WINDOW_LENGTH/2]).year
last_mid_year = first_mid_year

while end < g.data.shape[0]:
    cnt += 1
    data_temp = g.data[start : end].copy()
    if AMPLITUDE:
        amp_temp = amplitude[start : end].copy()
    #last_mid_year = date.fromordinal(g.time[start + WINDOW_LENGTH/2]).year
    last_mid_year += 1
    phase_temp = phase[start : end].copy()
    phase_bins = get_equidistant_bins()
    for i in range(cond_means.shape[0]): # get conditional means for current phase range
        ndx = ((phase_temp >= phase_bins[i]) & (phase_temp <= phase_bins[i+1]))
        if MEANS:
            if AMPLITUDE:
                cond_means[i] = np.mean(amp_temp[ndx])
            else:
                cond_means[i] = np.mean(data_temp[ndx])
        else:
            if AMPLITUDE:
                cond_means[i] = np.var(amp_temp[ndx], ddof = 1)
            else:
                cond_means[i] = np.var(data_temp[ndx], ddof = 1)
#    print last_mid_year, cond_means
    difference_data.append(cond_means.max() - cond_means.min()) # append difference to list    
    meanvar_data.append(np.mean(cond_means))
    if last_mid_year == PHASE_ANALYSIS_YEAR:
        plot_vars.append(phase_bins)
        plot_vars.append(cond_means.copy())
        plot_vars.append(phase_temp.copy())
        plot_vars.append([g.get_date_from_ndx(start), g.get_date_from_ndx(end)])
        plot_vars.append(cnt)
    start = g.find_date_ndx(date(y1 + cnt*WINDOW_SHIFT, 7, 28))
    end = start + WINDOW_LENGTH

difference_data = np.array(difference_data)
meanvar_data = np.array(meanvar_data)    
print("[%s] Wavelet analysis on data done. Starting analysis on surrogates..." % (str(datetime.now())))

#if PHASE_ANALYSIS_YEAR == last_mid_year:
#        # (bins, cond_means, cond_means_surr, phase, dates, subtit = '', fname = None):
#        fn = ('debug/detail/%s_phase_bins_%d_time_point.png' % ('32to16' if WINDOW_LENGTH > 16000 else '16to14', last_mid_year))
#        render_phase_and_bins(phase_bins, cond_means, cond_means_surrs, phase, 
#                              [g_working.get_date_from_ndx(0), g_working.get_date_from_ndx(-1)], fname = fn)

if SURR_TYPE == 'AR':
    sg.prepare_AR_surrogates()
    if AMPLITUDE:
        sg_amp.prepare_AR_surrogates()


def _cond_difference_surrogates(sg, sg_amp, a, a2, jobq, resq):
    mean, var, trend = a
    mean2, var2, trend2 = a2
    last_mid_year = first_mid_year
    cond_means_out = np.zeros((8,))
    while jobq.get() is not None:
        if SURR_TYPE == 'MF':
            sg.construct_multifractal_surrogates()
            sg.add_seasonality(mean, var, trend)
            if AMPLITUDE:
                sg_amp.construct_multifractal_surrogates()
                sg_amp.add_seasonality(mean2, var2, trend2)
        elif SURR_TYPE == 'FT':
            sg.construct_fourier_surrogates_spatial()
            sg.add_seasonality(mean, var, trend)
            if AMPLITUDE:
                sg_amp.construct_fourier_surrogates_spatial()
                sg_amp.add_seasonality(mean2, var2, trend2)
        elif SURR_TYPE == 'AR':
            sg.construct_surrogates_with_residuals()
            sg.add_seasonality(mean[:-1, ...], var[:-1, ...], trend[:-1, ...])
            if AMPLITUDE:
                sg_amp.construct_surrogates_with_residuals()
                sg_amp.add_seasonality(mean2[:-1, ...], var2[:-1, ...], trend2[:-1, ...])
        wave, _, _, _ = wavelet_analysis.continous_wavelet(sg.surr_data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
        phase = np.arctan2(np.imag(wave), np.real(wave))
        if AMPLITUDE:
            wave, _, _, _ = wavelet_analysis.continous_wavelet(sg_amp.surr_data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0_amp, j1 = 0, k0 = k0) # perform wavelet
            amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))
            amplitude = amplitude[0, :]
            phase_amp = np.arctan2(np.imag(wave), np.real(wave))
            phase_amp = phase_amp[0, :]

            # fitting oscillatory phase / amplitude to actual SAT
            reconstruction = amplitude * np.cos(phase_amp)
            fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
            m, c = np.linalg.lstsq(fit_x, sg_amp.surr_data)[0]
            amplitude = m * amplitude + c

        if SURR_TYPE == 'AR':
            sg.surr_data = sg.surr_data[main_cut_ndx[:-1]]
            if AMPLITUDE:
                amplitude = amplitude[main_cut_ndx[:-1]]
        else:
            sg.surr_data = sg.surr_data[main_cut_ndx]
            if AMPLITUDE:
                amplitude = amplitude[main_cut_ndx]
        phase = phase[0, main_cut_ndx]
        phase_bins = get_equidistant_bins() # equidistant bins
        cnt = 0

        difference_surr = []
        meanvar_surr = []

        start = 0
        end = WINDOW_LENGTH
        
        while end < sg.surr_data.shape[0]:
            cnt += 1
            surr_temp = sg.surr_data[start : end].copy()
            phase_temp = phase[start : end].copy()
            if AMPLITUDE:
                amp_temp = amplitude[start : end].copy()
            last_mid_year += 1
            for i in range(cond_means.shape[0]): # get conditional means for current phase range
                #phase_bins = get_equiquantal_bins(phase_temp) # equiquantal bins
                ndx = ((phase_temp >= phase_bins[i]) & (phase_temp <= phase_bins[i+1]))
                if MEANS:
                    if AMPLITUDE:
                        cond_means[i] = np.mean(amp_temp[ndx])
                    else:
                        cond_means[i] = np.mean(surr_temp[ndx])
                else:
                    if AMPLITUDE:
                        cond_means[i] = np.var(amp_temp[ndx], ddof = 1)
                    else:
                        cond_means[i] = np.var(surr_temp[ndx], ddof = 1)
            if PHASE_ANALYSIS_YEAR == last_mid_year:
                cond_means_out = cond_means.copy()
            difference_surr.append(cond_means.max() - cond_means.min()) # append difference to list    
            meanvar_surr.append(np.mean(cond_means))
            
            start = g.find_date_ndx(date(y1 + cnt*WINDOW_SHIFT, 7, 28))
            end = start + WINDOW_LENGTH
        
        resq.put((np.array(difference_surr), np.array(meanvar_surr), cond_means_out))

# surrs
diffs_surr = np.zeros((NUM_SURR,cnt))
meanvars_surr = np.zeros_like(diffs_surr)
cond_means_surrs = np.zeros((NUM_SURR, 8))
surr_completed = 0
jobQ = Queue()
resQ = Queue()
for i in range(NUM_SURR):
    jobQ.put(1)
for i in range(WORKERS):
    jobQ.put(None)
    
a = (mean, var, trend)
if AMPLITUDE:
    a2 = (mean2, var2, trend2)
workers = [Process(target = _cond_difference_surrogates, args = (sg, sg_amp, a, a2, jobQ, resQ)) for iota in range(WORKERS)]

for w in workers:
    w.start()
while surr_completed < NUM_SURR:
    # get result
    diff, meanVar, cmsurr = resQ.get()
    diffs_surr[surr_completed, :] = diff
    meanvars_surr[surr_completed, :] = meanVar
    cond_means_surrs[surr_completed, :] = cmsurr
    surr_completed += 1
    print surr_completed, '. done...'
for w in workers:
    w.join()
    
    

difference_surr = []
difference_surr_std = []
meanvar_surr = []
meanvar_surr_std = []

difference_95perc = []
mean_95perc = []    

for i in range(cnt):
    difference_surr.append(np.mean(diffs_surr[:, i], axis = 0))
    difference_surr_std.append(np.std(diffs_surr[:, i], axis = 0, ddof = 1))
    
    meanvar_surr.append(np.mean(meanvars_surr[:, i], axis = 0))
    meanvar_surr_std.append(np.std(meanvars_surr[:, i], axis = 0, ddof = 1))
    
    percentil = difference_data[i] > diffs_surr[:, i]
    no_true = percentil[percentil == True].shape[0]
    difference_95perc.append(True if (no_true > NUM_SURR * 0.95) else False)
    
    percentil = meanvar_data[i] > meanvars_surr[:, i]
    no_true = percentil[percentil == True].shape[0]
    mean_95perc.append(True if (no_true > NUM_SURR * 0.95) else False)
 
if PHASE_ANALYSIS_YEAR != None:   
    fn = ('debug/detail/%d_%s_phase_bins_time_point.png' % (PHASE_ANALYSIS_YEAR, 'wavelet_first'))
    render_phase_and_bins(plot_vars[0], plot_vars[1], cond_means_surrs, plot_vars[2], 
                          plot_vars[3], percentil = difference_95perc[plot_vars[4]], fname = fn)
    
difference_95perc = np.array(difference_95perc)
mean_95perc = np.array(mean_95perc)

where_percentil = np.column_stack((difference_95perc, mean_95perc))

fn = ("debug/PRG_%s_%s%d_%ssurr_wavelet_first%s.png" % ('means' if MEANS else 'var', 'SATamplitude_' if AMPLITUDE else '', NUM_SURR, 
                                                        SURR_TYPE, '_phase' if PLOT_PHASE else ''))


render([difference_data, np.array(difference_surr)], [meanvar_data, np.array(meanvar_surr)], [np.array(difference_surr_std), np.array(meanvar_surr_std)],
        subtit = ("95 percentil: difference - %d/%d and mean %d/%d" % (difference_95perc[difference_95perc == True].shape[0], cnt, mean_95perc[mean_95perc == True].shape[0], cnt)), 
        percentil = where_percentil, phase = None, fname = fn)
                
