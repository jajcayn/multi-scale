"""
created on June 6, 2014

@author: Nikola Jajcay
"""

from src.data_class import load_station_data
from src import wavelet_analysis
from surrogates.surrogates import SurrogateField
import numpy as np
from datetime import date, datetime
from multiprocessing import Process, Queue
import cPickle
import matplotlib.pyplot as plt


ANOMALISE = True
PERIOD = 8
START_DATE = None #date(1970, 1, 1)
END_DATE = date(2013,12,31)
MEANS = True
NUM_SURR = 100
WORKERS = 3
MF_SURR = True
SAVE_FILE = False


## load data
g = load_station_data('TG_STAID000027.txt', date(1950,1,1), date(2014,1,1), ANOMALISE)
sg = SurrogateField()


print("[%s] Wavelet analysis and conditional mean computation in progress..." % (str(datetime.now())))
k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale 


def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))

# get 16k length of the time series because of surrogates
_ = g.get_data_of_precise_length(length = '16k', start_date = START_DATE, end_date = END_DATE, COPY = True)
# remove seasonality from data
mean, var, trend = g.get_seasonality(True)
# copy deseasonalised data to surrogate field
sg.copy_field(g)
# return seasonality to the data
g.return_seasonality(mean, var, trend)


## wavelet DATA
wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
phase = np.arctan2(np.imag(wave), np.real(wave))

END_DATE = g.get_date_from_ndx(-1)
START_DATE = g.get_date_from_ndx(0)


edge_cut_ndx = g.select_date(date(START_DATE.year + 4, START_DATE.month, START_DATE.day), date(END_DATE.year - 4, END_DATE.month, END_DATE.day))
phase = phase[0, edge_cut_ndx]

print g.get_date_from_ndx(0), ' -- ', g.get_date_from_ndx(-1)

## cond means DATA
cond_means = np.zeros((8,))

phase_bins = get_equidistant_bins() # equidistant bins
for i in range(cond_means.shape[0]):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    if MEANS:
        cond_means[i] = np.mean(g.data[ndx])
    else:
        cond_means[i] = np.var(g.data[ndx], ddof = 1)


## wavelet and cond means SURROGATES
def _cond_difference_surrogates(sg, a, jobq, resq):
    while jobq.get() is not None:
        cond_means_temp = np.zeros_like(cond_means)
        if MF_SURR:
            sg.construct_multifractal_surrogates()
        else:
            sg.construct_fourier_surrogates_spatial()
        sg.add_seasonality(a[0], a[1], a[2])

        wave, _, _, _ = wavelet_analysis.continous_wavelet(sg.surr_data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
        phase = np.arctan2(np.imag(wave), np.real(wave))

        sg.surr_data = sg.surr_data[edge_cut_ndx]
        phase = phase[0, edge_cut_ndx]

        phase_bins = get_equidistant_bins()
        for i in range(cond_means_surr.shape[0]):
            ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
            if MEANS:
                cond_means_temp[i] = np.mean(sg.surr_data[ndx])
            else:
                cond_means_temp[i] = np.var(sg.surr_data[ndx], ddof = 1)
        resq.put(cond_means_temp)



print("[%s] Conditional means on data done, now computing %d surrogates in parallel using %d threads..." % (str(datetime.now()), NUM_SURR, WORKERS))
cond_means_surr = np.zeros((8, NUM_SURR))
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
    cond_means_temp = resQ.get()
    cond_means_surr[:, surr_completed] = cond_means_temp
    surr_completed += 1
    if surr_completed % 10 == 0:
        print("...%d. surrogate completed..." % surr_completed)
for w in workers:
    w.join()

print("[%s] Conditional means on surrogates done. Saving data..." % (str(datetime.now())))

if SAVE_FILE:
    fname = ("PRG_50-14_%s.bin" % ('var' if not MEANS else 'means'))
    with open(fname, 'w') as f:
        cPickle.dump({"conditional_means_data" : cond_means, "conditional_means_surrs" : cond_means_surr}, f)
    
    
diff = (phase_bins[1]-phase_bins[0])
fig = plt.figure(figsize=(6,10))
b1 = plt.bar(phase_bins[:-1], cond_means, width = diff*0.45, bottom = None, fc = '#403A37', figure = fig)
b2 = plt.bar(phase_bins[:-1] + diff*0.5, np.mean(cond_means_surr, axis = 1), width = diff*0.45, bottom = None, fc = '#A09793', figure = fig)
plt.xlabel('phase [rad]')
mean_of_diffs = np.mean([cond_means_surr[:,i].max() - cond_means_surr[:,i].min() for i in range(cond_means_surr.shape[1])])
std_of_diffs = np.std([cond_means_surr[:,i].max() - cond_means_surr[:,i].min() for i in range(cond_means_surr.shape[1])], ddof = 1)
plt.legend( (b1[0], b2[0]), ('data', 'mean of %d surr' % NUM_SURR) )
plt.ylabel('cond variance temperature [$^{\circ}$C$^{2}$]')
plt.axis([-np.pi, np.pi, -1.5, 1.5])
plt.title('%s cond variance \n difference data: %.2f$^{\circ}$C \n mean of diffs: %.2f$^{\circ}$C \n std of diffs: %.2f$^{\circ}$C$^{2}$' % (g.location, 
           (cond_means.max() - cond_means.min()), mean_of_diffs, std_of_diffs))

plt.savefig('debug/PRG_eval_from_%s-%s.png' % (str(g.get_date_from_ndx(0).year)[2:], str(g.get_date_from_ndx(-1).year)[2:]))
