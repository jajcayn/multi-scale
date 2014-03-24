"""
created on Mar24, 2014

@author: Nikola Jajcay
"""

import numpy as np
from src.data_class import load_station_data
from datetime import date
import matplotlib.pyplot as plt


def detrend_with_coefs(ts):
    y = ts
    x = np.arange(0, ts.shape[0], 1)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    print m,c
    output = np.zeros_like(ts)
    for i in range(ts.shape[0]):
        output[i] = ts[i] - (m*x[i] + c)
    
    return output, m


WINDOW = 32 # years


# load AMO smoothed version
amo_sm_raw = np.loadtxt('../data/amon.sm.long.data', comments = "#", skiprows = 1) # no. of years x 13
# load AMO unsmoothed version
amo_un_raw = np.loadtxt('../data/amon.us.long.data', comments = "#", skiprows = 1)
missing_val = -99.99


## loading data ##
start_date = date(1856,1,1)
end_date = date(2014, 1, 1) # exclusive
g = load_station_data('TG_STAID000027.txt', start_date, end_date, True)

# data var
data_var = np.zeros((end_date.year - 1 - start_date.year - WINDOW + 1))
start_idx = g.find_date_ndx(start_date)
year = start_date.year
end_idx = g.find_date_ndx(date(year + WINDOW, 1, 1))
coefs = []

print data_var.shape[0]
for i in range(data_var.shape[0]):
#    detrended_data, coef = detrend_with_coefs(g.data[start_idx:end_idx])
#    coefs.append(coef)
    s_idx = start_idx
    e_idx = g.find_date_ndx(date(year+1,1,1))
    var = []
    for j in range(WINDOW):
        var.append(np.var(g.data[s_idx:e_idx], axis = 0, ddof = 1))
        s_idx = e_idx
        e_idx = g.find_date_ndx(date(year+j+2,1,1))
#    var = np.var(g.data[start_idx:end_idx], axis = 0, ddof = 1)
    data_var[i] = np.mean(np.array(var), axis = 0)
    year += 1
    start_idx = g.find_date_ndx(date(year, 1, 1))
    end_idx = g.find_date_ndx(date(year + WINDOW, 1, 1))

    
    
data_var -= np.mean(data_var, axis = 0)
data_var /= np.std(data_var, axis = 0, ddof = 1)
    
# AMO mean
amo_sm_mean = np.zeros_like(data_var)
amo_un_mean = np.zeros_like(data_var)
iota = 0
for i in range(amo_sm_mean.shape[0]):
    if np.all(amo_sm_raw[iota:iota+WINDOW, 1:] != missing_val):
        mean1 = np.mean(amo_sm_raw[iota:iota+WINDOW, 1:], axis = None)
        amo_sm_mean[i] = mean1
    else:
        amo_sm_mean[i] = (np.nan)
    if np.all(amo_un_raw[iota:iota+WINDOW, 1:] != missing_val):
        mean2 = np.mean(amo_un_raw[iota:iota+WINDOW, 1:], axis = None)
        amo_un_mean[i] = mean2
    else:
        amo_un_mean.append(np.nan)
    iota += 1


# get correlations
lags = [-2, -1, 0, 1, 2]
corrs = []
for lag in lags:
    arr = np.zeros((data_var.shape[0] - np.abs(lag), 2))
    if lag > 0:
        arr[:,0] = data_var[lag:]
        arr[:,1] = amo_un_mean[:-lag]
    elif lag < 0:
        arr[:,0] = data_var[:lag]
        arr[:,1] = amo_un_mean[-lag:]
    elif lag == 0:
        arr[:,0] = data_var[:]
        arr[:,1] = amo_un_mean[:]
    corrs.append(np.corrcoef(arr, rowvar = 0)[0,1])


fig, ax1 = plt.subplots(figsize=(15,12), dpi=300)
p1, = ax1.plot(data_var, linewidth = 2.5, color = "#3E4147") 
ax1.axis([0, len(data_var), -4, 4])
ticks = np.linspace(0, len(data_var), 20)
ticks_n = np.linspace(start_date.year + WINDOW/2, end_date.year - WINDOW/2, 20)
ticks_name = [ int(y) for y in ticks_n ]
plt.xticks(ticks, ticks_name, rotation = 30)
plt.xlabel("years")
ax1.set_ylabel("normalised mean of yearly SATA variance", size = 16)
ax2 = ax1.twinx()
p2, = ax2.plot(amo_sm_mean, linewidth = 1, color = "#FCD036")
p3, = ax2.plot(amo_un_mean, linewidth = 2, color = "#DC5B3E") 
#p4, = ax2.plot(coefs, linewidth = 0.8)
ax2.axis([0, len(data_var), -0.5, 0.5])
ax2.set_ylabel("AMO index", size = 16)
plt.legend([p1, p2, p3], ["Klementinum SATA var", "AMO smoothed mean", "AMO unsmoothed mean"])
tit = ("PRG Klementinum SATA yearly var means over %d years window vs. AMO index mean over the same" % (WINDOW))
plt.text(0.5, 1.02, tit, horizontalalignment = 'center', size = 18, transform = ax2.transAxes)
props = dict(boxstyle = 'round, pad=0.1', facecolor = '#8EBE94', alpha = 0.5)
textstr = 'AMO unsmoothed -> SATA var'
for i in range(len(corrs)):
    textstr += ("\n lag %d years: %.4f" % (lags[i], corrs[i]))
plt.text(0.05, 0.1, textstr, transform = ax2.transAxes, size = 15, verticalalignment = 'bottom', bbox = props)


plt.savefig("results/AMO/AMO_yearly_var_means_%d_window.png" % WINDOW)






