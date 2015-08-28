"""
created on Apr 10, 2014

@author: Nikola Jajcay
"""

from src.data_class import DataField
import numpy as np
from datetime import date, datetime
from src import wavelet_analysis
from multiprocessing import Process, Queue
from surrogates.surrogates import SurrogateField

#==============================================================================
# ERA data download
# 
# from ecmwfapi import ECMWFDataServer
# 
# server = ECMWFDataServer()
# 
# server.retrieve({
#     "stream" : "oper",
#     "levtype" : "sfc",
#     "param" : "167.128",
#     "dataset" : "interim", ## era40, interim
#     "step" : "0",
#     "grid" : "2.5/2.5",
#     "time" : "00/06/12/18", ## daily
#     "date" : "20010101/to/20131231",
#     "area" : "50/-15/30/5", ## north/west/south/east
#     "type" : "an",
#     "class" : "e4",
#     "format" : "netcdf",
#     "padding" : "0",
#     "target" : "test.nc"
#    })
    
#==============================================================================

# load ERA-40 as g1 and ERA-Interim as g2
print("[%s] Loading data..." % (str(datetime.now())))
g1 = DataField()
g2 = DataField()

g1.load('Spain.ERA.58-01.nc', 't2m', 'ERA-40')
g2.load('Spain.ERA.01-13.nc', 't2m', 'ERA-40')

# concatenate
last = g1.time[-1]
idx = np.where(g2.time == last)[0] + 1
ndays = g1.time.shape[0] + g2.time[idx:].shape[0]

data = np.zeros((ndays, g1.lats.shape[0], g1.lons.shape[0]))
time = np.zeros((ndays, ))

data[:g1.time.shape[0], ...] = g1.data
data[g1.time.shape[0]:, ...] = g2.data[idx:]
time[:g1.time.shape[0]] = g1.time
time[g1.time.shape[0]:] = g2.time[idx:]

# get daily values from 6-hourly values
data_new = np.zeros((ndays // 4, g1.lats.shape[0], g2.lons.shape[0]))
time_new = np.zeros((ndays // 4))
for i in range(data_new.shape[0]):
    data_new[i, ...] = np.mean(data[4*i : 4*i+3, ...], axis = 0)
    time_new[i] = time[4*i]

# enroll into one DataField class
g = DataField(data = data_new, lons = g1.lons, lats = g1.lats, time = time_new)
del g1, g2

# anomalise
g.select_date(date(1969, 2, 22), date(2014, 1, 1))
g.anomalise()
print("[%s] Data loaded. Now performing wavelet analysis..." % str(datetime.now()))


MEANS = True
WORKERS = 3
num_surr = 1000

k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = 8 * y # frequency of interest
s0 = period / fourier_factor # get scale 

cond_means = np.zeros((8,))


def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))
    

d, m, year = g.extract_day_month_year()
difference = np.zeros((g.lats.shape[0], g.lons.shape[0]))
mean_var = np.zeros_like(difference)

for i in range(g.lats.shape[0]):
    for j in range(g.lons.shape[0]):
        
        wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data[:,i,j], 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
        phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes
        for iota in range(cond_means.shape[0]): # get conditional means for current phase range
            #phase_bins = get_equiquantal_bins(phase_temp) # equiquantal bins
            phase_bins = get_equidistant_bins() # equidistant bins
            ndx = ((phase[0,:] >= phase_bins[iota]) & (phase[0,:] <= phase_bins[iota+1]))
            if MEANS:
                cond_means[iota] = np.mean(g.data[ndx])
            else:
                cond_means[iota] = np.var(g.data[ndx], ddof = 1)
        difference[i, j] = cond_means.max() - cond_means.min() # append difference to list    
        mean_var[i, j] = np.mean(cond_means)
        

print("[%s] Wavelet analysis done. Now computing wavelet for MF surrogates in parallel..." % str(datetime.now()))
surrogates_difference = np.zeros([num_surr] + list(difference.shape))
surrogates_mean_var = np.zeros_like(surrogates_difference)
surr_completed = 0

sg = SurrogateField()
sg.copy_field(g)
mean, var, trend = g.get_seasonality(DETREND = True)

def _cond_difference_surrogates(sg, jobq, resq):
    while jobq.get() is not None:
        difference = np.zeros((sg.lats.shape[0], sg.lons.shape[0]))
        mean_var = np.zeros_like(difference)
        sg.construct_multifractal_surrogates()
        sg.add_seasonality(mean, var, trend)
        for i in range(sg.lats.shape[0]):
            for j in range(sg.lons.shape[0]):
                wave, _, _, _ = wavelet_analysis.continous_wavelet(sg.surr_data[:, i, j], 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
                phase = np.arctan2(np.imag(wave), np.real(wave)) # get phases from oscillatory modes
                for iota in range(cond_means.shape[0]): # get conditional means for current phase range
                    #phase_bins = get_equiquantal_bins(phase_temp) # equiquantal bins
                    phase_bins = get_equidistant_bins() # equidistant bins
                    ndx = ((phase[0,:] >= phase_bins[iota]) & (phase[0,:] <= phase_bins[iota+1]))
                    if MEANS:
                        cond_means[iota] = np.mean(g.data[ndx])
                    else:
                        cond_means[iota] = np.var(g.data[ndx], ddof = 1)
                difference[i, j] = cond_means.max() - cond_means.min() # append difference to list    
                mean_var[i, j] = np.mean(cond_means)
        resq.put((difference, mean_var))

jobQ = Queue()
resQ = Queue()
for i in range(num_surr):
    jobQ.put(1)
for i in range(WORKERS):
    jobQ.put(None)
    
workers = [Process(target = _cond_difference_surrogates, args = (sg, jobQ, resQ)) for iota in range(WORKERS)]

print("[%s] Starting %d workers..." % (str(datetime.now()), WORKERS))
for w in workers:
    w.start()

    
while surr_completed < num_surr:
    
    # get result
    diff, meanVar = resQ.get()
    surrogates_difference[surr_completed, ...] = diff
    surrogates_mean_var[surr_completed, ...] = meanVar
    surr_completed += 1
    
    if surr_completed % 10 == 0:
        print("[%s] PROGRESS: %d/%d surrogates completed." % (str(datetime.now()), surr_completed, num_surr))
        
for w in workers:
    w.join()
    
print("[%s] Wavelet analysis of surrogates done." % (str(datetime.now())))

output = np.zeros((2, g.lats.shape[0], g.lons.shape[0]))
for i in range(g.lons.shape[0]):
    for j in range(g.lats.shape[0]):
        surr_mean = np.mean(surrogates_difference[:, i, j], axis = 0)
        surr_std = np.std(surrogates_difference[:, i, j], axis = 0, ddof = 1)
        output[0, i, j] = (difference[i,j] - surr_mean) / surr_std
        
        surr_mean = np.mean(surrogates_mean_var[:, i, j], axis = 0)
        surr_std = np.std(surrogates_mean_var[:, i, j], axis = 0, ddof = 1)
        output[1, i, j] = (mean_var[i,j] - surr_mean) / surr_std

text = np.zeros((g.lats.shape[0] * g.lons.shape[0], 4))
for i in range(g.lats.shape[0]):
    for j in range(g.lons.shape[0]):
        text[i*g.lats.shape[0] + j, 0] = g.lats[i]
        text[i*g.lats.shape[0] + j, 1] = g.lons[j]
        text[i*g.lats.shape[0] + j, 2] = output[0, i, j]
        text[i*g.lats.shape[0] + j, 3] = output[1, i, j]
        
np.savetxt('/home/nikola/Dropbox/multi-scale/Spain.ERA.SD.txt', text, delimiter = ',  ', fmt = '%.2f')
print("[%s] Analysis done. Files saved" % str(datetime.now()))





