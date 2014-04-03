"""
created on Mar 8, 2014

@author: Nikola Jajcay
"""

from src import wavelet_analysis
from src.data_class import load_station_data
from surrogates.surrogates import SurrogateField
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue


def render(diffs, meanvars, stds = None, subtit = '',  fname = None):
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
    tit = ('Evolution of difference in cond mean in temp SATA \n')
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
WORKERS = 2
num_surr = 1000 # how many surrs will be used to evaluate
rand = 2



## loading data ##
start_date = date(1834,7,28)
end_date = date(2014, 1, 1) # exclusive
g = load_station_data('TG_STAID000027.txt', start_date, end_date, ANOMALISE)
g_copy = load_station_data('TG_STAID000027.txt', start_date, end_date, ANOMALISE)
# length of the time series with date(1954,6,8) with start date(1775,1,1) = 65536 - power of 2
# the same when end date(2014,1,1) than start date(1834,7,28)

## wavelet analysis DATA
print("[%s] Wavelet analysis in progress with %d year window shifted by %d year(s)..." % (str(datetime.now()), WINDOW_LENGTH, WINDOW_SHIFT))
k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale 
wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, PAD, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes


cond_means = np.zeros((8,))


def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))
    

d, m, year = g.extract_day_month_year()
data_temp = np.zeros((WINDOW_LENGTH * y))
phase_temp = np.zeros((WINDOW_LENGTH * y))
difference = []
mean_var = []
start_idx = g.find_date_ndx(start_date) # set to first date
end_idx = start_idx + data_temp.shape[0] # first date plus WINDOW_LENGTH years (since year is 365.25, leap years are counted)
cnt = 0
while end_idx < g.data.shape[0]: # while still in the correct range
    cnt += 1
    data_temp = g.data[start_idx : end_idx] # subselect data
    phase_temp = phase[0,start_idx : end_idx]
    for i in range(cond_means.shape[0]): # get conditional means for current phase range
        #phase_bins = get_equiquantal_bins(phase_temp) # equiquantal bins
        phase_bins = get_equidistant_bins() # equidistant bins
        ndx = ((phase_temp >= phase_bins[i]) & (phase_temp <= phase_bins[i+1]))
        if MEANS:
            cond_means[i] = np.mean(data_temp[ndx])
        else:
            cond_means[i] = np.var(data_temp[ndx], ddof = 1)
    difference.append(cond_means.max() - cond_means.min()) # append difference to list    
    mean_var.append(np.mean(cond_means))
    start_idx = g.find_date_ndx(date(start_date.year + WINDOW_SHIFT * cnt, start_date.month, start_date.day)) # shift start index by WINDOW_SHIFT years
    end_idx = start_idx + data_temp.shape[0] # shift end index
print("[%s] Wavelet analysis done." % (str(datetime.now())))

difference = np.array(difference)
mean_var = np.array(mean_var)

## wavelet analysis SURROGATES

print("[%s] Now computing wavelet analysis for %d MF surrogates in parallel." % (str(datetime.now()), num_surr))
surrogates_difference = np.zeros((num_surr, difference.shape[0]))
surrogates_mean_var = np.zeros_like(surrogates_difference)
surr_completed = 0

#mean, var, trend = g.get_seasonality(DETREND = True) # use when MF surrogates are created before sliding window analysis

# prepare surr field
sg = SurrogateField()
#sg.copy_field(g)

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
def _cond_difference_surrogates(sg, jobq, resq):
    while jobq.get() is not None:
        difference = []
        mean_var = []
        start_idx = g_copy.find_date_ndx(start_date) # set to first date
        end_idx = start_idx + int(WINDOW_LENGTH * y) # first date plus WINDOW_LENGTH years (since year is 365.25, leap years are counted)
        cnt = 0
        while end_idx < g_copy.data.shape[0]: # while still in the correct range
            g.data = g_copy.data[start_idx : end_idx]
            g.time = g_copy.time[start_idx : end_idx]
            mean, var, trend = g.get_seasonality(DETREND = True)
            sg.copy_field(g)
            sg.construct_multifractal_surrogates(randomise_from_scale = rand)
            sg.add_seasonality(mean, var, trend)
            wave, _, _, _ = wavelet_analysis.continous_wavelet(sg.surr_data, 1, PAD, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
            phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes
            cnt += 1
            for i in range(cond_means.shape[0]): # get conditional means for current phase range
                #phase_bins = get_equiquantal_bins(phase_temp) # equiquantal bins
                phase_bins = get_equidistant_bins() # equidistant bins
                ndx = ((phase[0,:] >= phase_bins[i]) & (phase[0,:] <= phase_bins[i+1]))
                if MEANS:
                    cond_means[i] = np.mean(sg.surr_data[ndx])
                else:
                    cond_means[i] = np.var(sg.surr_data[ndx], ddof = 1)
            difference.append(cond_means.max() - cond_means.min()) # append difference to list    
            mean_var.append(np.mean(cond_means))
            start_idx = g_copy.find_date_ndx(date(start_date.year + WINDOW_SHIFT * cnt, start_date.month, start_date.day)) # shift start index by WINDOW_SHIFT years
            end_idx = start_idx + int(WINDOW_LENGTH * y) # shift end index
        
        resq.put((np.array(difference), np.array(mean_var)))

# construct the job queue
jobQ = Queue()
resQ = Queue()
for i in range(num_surr):
    jobQ.put(1)
for i in range(WORKERS):
    jobQ.put(None)
    
print("[%s] Starting workers..." % (str(datetime.now())))
workers = [Process(target = _cond_difference_surrogates, args = (sg, jobQ, resQ)) for iota in range(WORKERS)]

for w in workers:
    w.start()
    
while surr_completed < num_surr:
    
    # get result
    diff, meanVar = resQ.get()
    surrogates_difference[surr_completed, :] = diff
    surrogates_mean_var[surr_completed, :] = meanVar
    surr_completed += 1
    
    if surr_completed % 10 == 0:
        print("[%s] PROGRESS: %d/%d surrogates completed." % (str(datetime.now()), surr_completed, num_surr))
        
for w in workers:
    w.join()

print("[%s] Wavelet analysis of surrogates done." % (str(datetime.now())))



#statistical testing
print("[%s] Statistical testing in progress..." % (str(datetime.now())))
diff_mean = []
diff_std = []
meanvar_mean = []
meanvar_std = []
diff_test = np.zeros_like(difference, dtype = np.bool)
meanvar_test = np.zeros_like(difference, dtype = np.bool)
for t in range(difference.shape[0]):
#    surr_mean = np.mean(surrogates_difference[:, t], axis = 0)
#    surr_std = np.std(surrogates_difference[:, t], axis = 0, ddof = 1)
#    stats.append([surr_mean, surr_std])

    diff = np.mean(surrogates_difference[:, t], axis = 0)
    std = np.std(surrogates_difference[:, t], axis = 0, ddof = 1)
    meanvar = np.mean(surrogates_mean_var[:, t], axis = 0)
    m_std = np.std(surrogates_mean_var[:, t], axis = 0, ddof = 1)
    diff_mean.append(diff)
    diff_std.append(std)
    meanvar_mean.append(meanvar)
    meanvar_std.append(m_std)
    
    sorted_surr = np.sort(surrogates_difference[:, t], axis = 0)[::-1]
    # make bool array -> false means data is higher than surrogate
    test = difference[t] > sorted_surr
    # count falses
    falses = test[test == False].shape[0]
    if falses <= ((num_surr / 100) * 5):
        diff_test[t] = True
        
    sorted_surr = np.sort(surrogates_mean_var[:, t], axis = 0)[::-1]
    test = mean_var[t] > sorted_surr
    falses = test[test == False].shape[0]
    if falses <= ((num_surr / 100) * 5):
        meanvar_test[t] = True
    
dtest = diff_test[diff_test == True].shape[0]  
mvtest = meanvar_test[meanvar_test == True].shape[0]
    
render([difference, np.array(diff_mean)], [mean_var, np.array(meanvar_mean)], [np.array(diff_std), np.array(meanvar_std)],
        subtit = ("95percentil of difference %d and of mean %d from %d data points" % (dtest, mvtest, difference.shape[0])),
        fname = "debug/%dsurrogates_each_window_rand%d" % (num_surr, rand))
        
    

    
