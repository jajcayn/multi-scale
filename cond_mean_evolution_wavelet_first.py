
from src import wavelet_analysis
from src.data_class import load_station_data
from surrogates.surrogates import SurrogateField
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue


def render(diffs, meanvars, stds = None, subtit = '', percentil = None, fname = None):
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
    year_diff = np.round((last_mid_year - first_mid_year) / 10)
    xnames = np.arange(first_mid_year, last_mid_year, year_diff)
    plt.xticks(np.linspace(0, cnt, len(xnames)), xnames, rotation = 30)
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
        tit = ('Evolution of difference in cond means temp SATA -- %s \n wavelet first, then window -- ' % g.location)
    else:
        tit = ('Evolution of difference in cond variance in temp SATA -- %s \n wavelet first, then window -- ' % g.location)
    tit += subtit
    plt.text(0.5, 1.05, tit, horizontalalignment = 'center', size = 16, transform = ax2.transAxes)
    #ax2.set_xticks(np.arange(start_date.year, end_date.year, 20))
    
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
        
        
        
        
ANOMALISE = True
PERIOD = 8 # years, period of wavelet
WINDOW_LENGTH = 16384
WINDOW_SHIFT = 1 # years, delta in the sliding window analysis
MEANS = True # if True, compute conditional means, if False, compute conditional variance
WORKERS = 3
NUM_SURR = 20 # how many surrs will be used to evaluate
MF_SURR = True
diff_ax = (0, 1.8)
mean_ax = (-1, 1.5)


## loading data
g = load_station_data('TG_STAID000027.txt', date(1834,7,28), date(2014,1,1), ANOMALISE)
sg = SurrogateField()


print("[%s] Wavelet analysis in progress with %d year window shifted by %d year(s)..." % (str(datetime.now()), WINDOW_LENGTH, WINDOW_SHIFT))
k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale 


cond_means = np.zeros((8,))

def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))
    
wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, True, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes

mean, var, trend = g.get_seasonality(True)
sg.copy_field(g)
g.return_seasonality(mean, var, trend)

main_cut_ndx = g.select_date(date(1838,1,1), date(2010,1,1))
y1 = 1838
phase = phase[0, main_cut_ndx]

difference_data = []
meanvar_data = []

cnt = 0

start = 0
end = WINDOW_LENGTH

first_mid_year = date.fromordinal(g.time[WINDOW_LENGTH/2]).year

while end < g.data.shape[0]:
    cnt += 1
    data_temp = g.data[start : end].copy()
    last_mid_year = date.fromordinal(g.time[start + (end-start)/2]).year
    phase_temp = phase[start : end].copy()
    phase_bins = get_equidistant_bins()
    for i in range(cond_means.shape[0]): # get conditional means for current phase range
        ndx = ((phase_temp >= phase_bins[i]) & (phase_temp <= phase_bins[i+1]))
        if MEANS:
            cond_means[i] = np.mean(data_temp[ndx])
        else:
            cond_means[i] = np.var(data_temp[ndx], ddof = 1)
    difference_data.append(cond_means.max() - cond_means.min()) # append difference to list    
    meanvar_data.append(np.mean(cond_means))
    
    start = g.find_date_ndx(date(y1 + cnt*WINDOW_SHIFT, 1, 1))
    end = start + WINDOW_LENGTH

difference_data = np.array(difference_data)
meanvar_data = np.array(meanvar_data)    
print("[%s] Wavelet analysis on data done. Starting analysis on surrogates..." % (str(datetime.now())))


def _cond_difference_surrogates(sg, a, jobq, resq):
    mean, var, trend = a
    while jobq.get() is not None:
        if MF_SURR:
            sg.construct_multifractal_surrogates()
        else:
            sg.construct_fourier_surrogates_spatial()
        sg.add_seasonality(mean, var, trend)
        wave, _, _, _ = wavelet_analysis.continous_wavelet(sg.surr_data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
        phase = np.arctan2(np.imag(wave), np.real(wave))
        
        sg.surr_data = sg.surr_data[main_cut_ndx]
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
            for i in range(cond_means.shape[0]): # get conditional means for current phase range
                #phase_bins = get_equiquantal_bins(phase_temp) # equiquantal bins
                ndx = ((phase_temp >= phase_bins[i]) & (phase_temp <= phase_bins[i+1]))
                if MEANS:
                    cond_means[i] = np.mean(surr_temp[ndx])
                else:
                    cond_means[i] = np.var(surr_temp[ndx], ddof = 1)
            difference_surr.append(cond_means.max() - cond_means.min()) # append difference to list    
            meanvar_surr.append(np.mean(cond_means))
            
            start = g.find_date_ndx(date(y1 + cnt*WINDOW_SHIFT, 1, 1))
            end = start + WINDOW_LENGTH
        
        resq.put((np.array(difference_surr), np.array(meanvar_surr)))

# surrs
diffs_surr = np.zeros((NUM_SURR,cnt))
meanvars_surr = np.zeros_like(diffs_surr)
surr_completed = 0
jobQ = Queue()
resQ = Queue()
for i in range(NUM_SURR):
    jobQ.put(1)
for i in range(WORKERS):
    jobQ.put(None)
    
a = (mean, var, trend)
workers = [Process(target = _cond_difference_surrogates, args = (sg, a, jobQ, resQ)) for iota in range(WORKERS)]

for w in workers:
    w.start()
while surr_completed < NUM_SURR:
    # get result
    diff, meanVar = resQ.get()
    diffs_surr[surr_completed, :] = diff
    meanvars_surr[surr_completed, :] = meanVar
    surr_completed += 1
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
    
difference_95perc = np.array(difference_95perc)
mean_95perc = np.array(mean_95perc)

where_percentil = np.column_stack((difference_95perc, mean_95perc))

fn = ("debug/PRG_%d_surr_" % NUM_SURR)
if not MEANS:
    fn += 'var_'
if MF_SURR:
    fn += 'MF_wavelet_first.png'
else:
    fn += 'FT_wavelet_first.png'

                
render([difference_data, np.array(difference_surr)], [meanvar_data, np.array(meanvar_surr)], [np.array(difference_surr_std), np.array(meanvar_surr_std)],
        subtit = ("95 percentil: difference - %d/%d and mean %d/%d" % (difference_95perc[difference_95perc == True].shape[0], cnt, mean_95perc[mean_95perc == True].shape[0], cnt)), 
        percentil = where_percentil, fname = fn)
                
