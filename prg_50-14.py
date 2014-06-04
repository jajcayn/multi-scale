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


ANOMALISE = True
PERIOD = 8
START_DATE = date(1950, 1, 1)
MEANS = True
NUM_SURR = 20
WORKERS = 2
MF_SURR = True


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
_ = g.get_data_of_precise_length(length = '16k', start_date = START_DATE, end_date = None, COPY = True)
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

edge_cut_ndx = g.select_date(date(START_DATE.year + 4, START_DATE.month, START_DATE.day), date(END_DATE.year - 4, END_DATE.month, END_DATE.day))
phase = phase[0, edge_cut_ndx]

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

fname = ("PRG_50-14_%s.bin" % ('var' if not MEANS else 'means'))
with open(fname, 'w') as f:
    cPickle.dump({"conditional_means_data" : cond_means, "conditional_means_surrs" : cond_means_surr}, f)
