"""
created on Mar 8, 2014

@author: Nikola Jajcay
"""
#import matplotlib
#matplotlib.use("Agg")

from src import wavelet_analysis
from src.data_class import load_station_data
from surrogates.surrogates import SurrogateField
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue


def render(diffs, meanvars, stds = None, subtit = '', fname = None):
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
    #ax1.plot(total_diffs[0], np.arange(0,len(total_diffs[0])), total_diffs[1], np.arange(0, cnt))
    if not ANOMALISE and MEANS:
        ax1.axis([0, cnt-1, 0, 3])
    if not ANOMALISE and not MEANS:
        ax1.axis([0, cnt-1, 0, 25])
    if ANOMALISE and MEANS:
        ax1.axis([0, cnt-1, 0, 2])
    if ANOMALISE and not MEANS:
        ax1.axis([0, cnt-1, 0, 6])
    if np.int(WINDOW_LENGTH) == WINDOW_LENGTH:
        ax1.set_xlabel('start year of %d-year wide window' % WINDOW_LENGTH, size = 14)
    else:
        ax1.set_xlabel('start year of %.2f-year wide window' % WINDOW_LENGTH, size = 14)
    if MEANS:
        ax1.set_ylabel('difference in cond mean in temperature [$^{\circ}$C]', size = 14)
    elif not MEANS:
        ax1.set_ylabel('difference in cond variance in temperature [$^{\circ}$C$^2$]', size = 14)
    plt.xticks(np.arange(0,cnt,15), np.arange(start_date.year, end_date.year, 15), rotation = 30)
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
    if MEANS:
        ax2.set_ylabel('mean of cond means in temperature [$^{\circ}$C]', size = 14)
    elif not MEANS:
        ax2.set_ylabel('mean of cond variance in temperature [$^{\circ}$C$^2$]', size = 14)
    ax2.axis([0, cnt-1, -1, 1.5])
    for tl in ax2.get_yticklabels():
        tl.set_color('#CA4F17')
    if len(diffs) < 3:
        plt.legend([p1, p2, p3, p4], ["difference DATA", "difference SURROGATE mean", "mean DATA", "mean SURROGATE mean"])
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
    tit = ('Evolution of difference in cond mean in temp SATA -- %s \n' % g_copy.location)
    tit += subtit
    plt.text(0.5, 1.05, tit, horizontalalignment = 'center', size = 16, transform = ax2.transAxes)
    #ax2.set_xticks(np.arange(start_date.year, end_date.year, 20))
    
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()







ANOMALISE = True
PERIOD = 8 # years, period of wavelet
#WINDOW_LENGTH = 32 # years, should be at least PERIOD of wavelet
WINDOW_LENGTH = 16384 / 365.25
WINDOW_SHIFT = 1 # years, delta in the sliding window analysis
PLOT = True
PAD = False # whether padding is used in wavelet analysis (see src/wavelet_analysis)
MEANS = True # if True, compute conditional means, if False, compute conditional variance
WORKERS = 3
num_surr = 1000 # how many surrs will be used to evaluate
rand = 2



## loading data ##
start_date = date(1834, 7, 28)
end_date = date(2014, 1, 1) # exclusive
g = load_station_data('TG_STAID000010.txt', start_date, end_date, ANOMALISE)
g_copy = load_station_data('TG_STAID000010.txt', start_date, end_date, ANOMALISE)
# length of the time series with date(1954,6,8) with start date(1775,1,1) = 65536 - power of 2
# the same when end date(2014,1,1) than start date(1834,7,28)

## wavelet analysis DATA
print("[%s] Wavelet analysis in progress with %d year window shifted by %d year(s)..." % (str(datetime.now()), WINDOW_LENGTH, WINDOW_SHIFT))
k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale 
#wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, PAD, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
#phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes


cond_means = np.zeros((8,))


def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))
    

d, m, year = g.extract_day_month_year()
#data_temp = np.zeros((WINDOW_LENGTH * y))
#phase_temp = np.zeros((WINDOW_LENGTH * y))
difference = []
mean_var = []
start_idx = g_copy.find_date_ndx(start_date) # set to first date
end_idx = start_idx + int(WINDOW_LENGTH * y) # first date plus WINDOW_LENGTH years (since year is 365.25, leap years are counted)
cnt = 0
while end_idx < g_copy.data.shape[0]: # while still in the correct range
    cnt += 1
    g.data = g_copy.data[start_idx : end_idx].copy() # subselect data
    g.time = g_copy.time[start_idx : end_idx]
    #phase_temp = phase[0, start_idx : end_idx]
    #data_temp = g_copy.data[start_idx : end_idx]
    if np.all(np.isnan(g.data) == False): # check for missing values
        wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, PAD, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
        phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes
        for i in range(cond_means.shape[0]): # get conditional means for current phase range
            #phase_bins = get_equiquantal_bins(phase_temp) # equiquantal bins
            phase_bins = get_equidistant_bins() # equidistant bins
            ndx = ((phase[0, :] >= phase_bins[i]) & (phase[0, :] <= phase_bins[i+1]))
            if MEANS:
                cond_means[i] = np.mean(g.data[ndx])
            else:
                cond_means[i] = np.var(g.data[ndx], ddof = 1)
        difference.append(cond_means.max() - cond_means.min()) # append difference to list    
        mean_var.append(np.mean(cond_means))
    else:
        difference.append(0)
        mean_var.append(0)
    start_idx = g_copy.find_date_ndx(date(start_date.year + WINDOW_SHIFT * cnt, start_date.month, start_date.day)) # shift start index by WINDOW_SHIFT years
    end_idx = start_idx + int(WINDOW_LENGTH * y) # shift end index
print("[%s] Wavelet analysis done." % (str(datetime.now())))

difference = np.array(difference)
mean_var = np.array(mean_var)


## wavelet analysis SURROGATES

print("[%s] Now computing wavelet analysis for %d MF surrogates in parallel." % (str(datetime.now()), num_surr))


#mean, var, trend = g_copy.get_seasonality(DETREND = True) # use when MF surrogates are created before sliding window analysis

# prepare surr field

#sg.copy_field(g_copy)

#==============================================================================
# function for MF surrogates to be created BEFORE sliding window analysis
#==============================================================================
#def _cond_difference_surrogates(sg, jobq, resq):
#    while jobq.get() is not None:
#        sg.construct_multifractal_surrogates(randomise_from_scale = rand)
#        sg.add_seasonality(mean, var, trend)
#        wave, _, _, _ = wavelet_analysis.continous_wavelet(sg.surr_data, 1, PAD, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
#        phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes
#        
#        difference = []
#        mean_var = []
#        data_temp = np.zeros((WINDOW_LENGTH * y))
#        phase_temp = np.zeros((WINDOW_LENGTH * y))
#        start_idx = g.find_date_ndx(start_date) # set to first date
#        end_idx = start_idx + data_temp.shape[0] # first date plus WINDOW_LENGTH years (since year is 365.25, leap years are counted)
#        cnt = 0
#        while end_idx < sg.surr_data.shape[0]: # while still in the correct range
#            cnt += 1
#            data_temp = sg.surr_data[start_idx : end_idx] # subselect data
#            phase_temp = phase[0,start_idx : end_idx]
#            for i in range(cond_means.shape[0]): # get conditional means for current phase range
#                #phase_bins = get_equiquantal_bins(phase_temp) # equiquantal bins
#                phase_bins = get_equidistant_bins() # equidistant bins
#                ndx = ((phase_temp >= phase_bins[i]) & (phase_temp <= phase_bins[i+1]))
#                if MEANS:
#                    cond_means[i] = np.mean(data_temp[ndx])
#                else:
#                    cond_means[i] = np.var(data_temp[ndx], ddof = 1)
#            difference.append(cond_means.max() - cond_means.min()) # append difference to list    
#            mean_var.append(np.mean(cond_means))
#            start_idx = g.find_date_ndx(date(start_date.year + WINDOW_SHIFT * cnt, start_date.month, start_date.day)) # shift start index by WINDOW_SHIFT years
#            end_idx = start_idx + data_temp.shape[0] # shift end index
#        
#        resq.put((np.array(difference), np.array(mean_var)))


#==============================================================================
# function for MF surrogates to be created in EACH WINDOW separately
#==============================================================================
def _cond_difference_surrogates(g, a, jobq, resq):
    mean, var, trend = a
    while jobq.get() is not None:
        sg = SurrogateField()
        sg.copy_field(g)
        sg.construct_multifractal_surrogates()
        #sg.construct_fourier_surrogates_spatial()
        sg.add_seasonality(mean, var, trend)
        wave, _, _, _ = wavelet_analysis.continous_wavelet(sg.surr_data, 1, PAD, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
        phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes
        for i in range(cond_means.shape[0]): # get conditional means for current phase range
            #phase_bins = get_equiquantal_bins(phase_temp) # equiquantal bins
            phase_bins = get_equidistant_bins() # equidistant bins
            ndx = ((phase[0,:] >= phase_bins[i]) & (phase[0,:] <= phase_bins[i+1]))
            if MEANS:
                cond_means[i] = np.mean(sg.surr_data[ndx])
            else:
                cond_means[i] = np.var(sg.surr_data[ndx], ddof = 1)
        diff = (cond_means.max() - cond_means.min()) # append difference to list    
        mean_var = np.mean(cond_means)
        
        resq.put((diff, mean_var))


difference_surr = []
difference_surr_std = []
mean_var_surr = []
mean_var_surr_std = []
difference_95perc = np.zeros_like(difference, np.bool)
mean_95perc = np.zeros_like(difference, np.bool)
start_idx = g_copy.find_date_ndx(start_date) # set to first date
end_idx = start_idx + int(WINDOW_LENGTH * y) # first date plus WINDOW_LENGTH years (since year is 365.25, leap years are counted)
cnt = 0
while end_idx < g_copy.data.shape[0]: # while still in the correct range
    surr_completed = 0
    diffs = np.zeros((num_surr,))
    mean_vars = np.zeros_like(diffs)
    g.data = g_copy.data[start_idx : end_idx].copy()
    g.time = g_copy.time[start_idx : end_idx]
    if np.all(np.isnan(g.data) == False): # check for missing values
        # construct the job queue
        jobQ = Queue()
        resQ = Queue()
        for i in range(num_surr):
            jobQ.put(1)
        for i in range(WORKERS):
            jobQ.put(None)
        a = g.get_seasonality(DETREND = True)
        workers = [Process(target = _cond_difference_surrogates, args = (g, a, jobQ, resQ)) for iota in range(WORKERS)]
        for w in workers:
            w.start()
        while surr_completed < num_surr:
            # get result
            diff, meanVar = resQ.get()
            diffs[surr_completed] = diff
            mean_vars[surr_completed] = meanVar
            surr_completed += 1
            #if surr_completed % 50 == 0:
             #   print("[%s] PROGRESS:%d time window - %d/%d surrogates completed." % (str(datetime.now()), cnt, surr_completed, num_surr))   
        for w in workers:
            w.join()
        
        difference_surr.append(np.mean(diffs))
        difference_surr_std.append(np.std(diffs, ddof = 1))
        mean_var_surr.append(np.mean(mean_vars))
        mean_var_surr_std.append(np.std(mean_vars, ddof = 1))
        
        percentil = difference[cnt] > diffs
        no_true = percentil[percentil == True].shape[0]
        difference_95perc[cnt] = True if (no_true > num_surr * 0.95) else False
        
        percentil = mean_var[cnt] > mean_vars
        no_true = percentil[percentil == True].shape[0]
        mean_95perc[cnt] = True if (no_true > num_surr * 0.95) else False
        print("%d. time point - data: %.2f, surr mean: %.2f, surr std: %.2f" % (cnt, difference[cnt], np.mean(diffs), np.std(diffs, ddof = 1)))    
    else:
        difference_surr.append(0)
        difference_surr_std.append(0)
        mean_var_surr.append(0)
        mean_var_surr_std.append(0)

    cnt += 1
    start_idx = g_copy.find_date_ndx(date(start_date.year + WINDOW_SHIFT * cnt, start_date.month, start_date.day)) # shift start index by WINDOW_SHIFT years
    end_idx = start_idx + int(WINDOW_LENGTH * y) # shift end index

print("[%s] Wavelet analysis of surrogates done." % (str(datetime.now())))


        
render([difference, np.array(difference_surr)], [mean_var, np.array(mean_var_surr)], [np.array(difference_surr_std), np.array(mean_var_surr_std)],
        subtit = ("95 percentil: difference - %d/%d and mean %d/%d" % (difference_95perc[difference_95perc == True].shape[0], cnt, mean_95perc[mean_95perc == True].shape[0], cnt)), 
        fname = "debug/STHLM_%d_surr_MF_in_each_window.png" % (num_surr))
        
    

    
