
from src.data_class import load_station_data
from datetime import date
import numpy as np
# from surrogates.surrogates import SurrogateField
from src import wavelet_analysis
# import scipy.io as sio
import matplotlib.pyplot as plt


def running_mean(arr, aver):
    out = np.zeros((arr.shape[0] - aver + 1,))
    for i in range(out.shape[0]):
        out[i] = np.mean(arr[i : i+aver])

    return out

def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))


BINS_COND = True

g = load_station_data('TG_STAID000027.txt', date(1834, 4, 28), date(2013, 10, 1), False)
g_max = load_station_data('TX_STAID000027.txt', date(1834, 4, 28), date(2013, 10, 1), False)
g_min = load_station_data('TN_STAID000027.txt', date(1834, 4, 28), date(2013, 10, 1), False)
if BINS_COND:
    k0 = 6. # wavenumber of Morlet wavelet used in analysis
    y = 365.25 # year in days
    fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
    period = 8 * y # frequency of interest
    s0 = period / fourier_factor # get scale
    wave, _, _, _ = wavelet_analysis.continous_wavelet(g.data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
    phase = np.arctan2(np.imag(wave), np.real(wave))


ndx = g.select_date(date(1838,4,28), date(2009,10,1)) # data as with temporal evolution (of anything studied)
g_max.select_date(date(1838,4,28), date(2009,10,1))
g_min.select_date(date(1838,4,28), date(2009,10,1))
if BINS_COND:
    phase = phase[0, ndx]
# g_max.select_date(date(1838,4,28), date(2009,10,1))
# g_min.select_date(date(1838,4,28), date(2009,10,1))
d, m, y = g.extract_day_month_year()

if BINS_COND:
    cond_means = np.zeros((8,))
    phase_bins = get_equidistant_bins()
    for i in range(cond_means.shape[0]):
        ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
        m_temp = m[ndx].copy()
        # data_temp = g.data[ndx].copy()
        data_temp_min = g_min.data[ndx].copy()
        data_temp_max = g_max.data[ndx].copy()
        djf_ndx = filter(lambda i: m_temp[i] == 12 or m_temp[i] < 3, range(data_temp_max.shape[0]))
        jja_ndx = filter(lambda i: m_temp[i] > 5 and m_temp[i] < 9, range(data_temp_max.shape[0]))
        djf = np.sort(data_temp_min[djf_ndx])
        jja = np.sort(data_temp_max[jja_ndx])[::-1]
        cond_means[i] = np.abs(np.nanmean(djf[:np.floor(0.1*djf.shape[0])], axis = 0) - np.nanmean(jja[:np.floor(0.1*jja.shape[0])], axis = 0))

    diff = (phase_bins[1]-phase_bins[0])
    fig = plt.figure(figsize=(6,10))
    plt.bar(phase_bins[:-1] + diff*0.05, cond_means, width = diff*0.9, bottom = None, fc = '#47C455', figure = fig)
    plt.xlabel('phase [rad]')
    plt.ylabel('cond. means of differences in mean 10perc min DJF vs. max JJA SAT [$^{\circ}$C]')
    plt.axis([-np.pi, np.pi, 40, 55])
    plt.title("min DJF vs. max JJA mean \n of 10 percent SAT differences in bins")
    plt.savefig('debug/cond_DJF_JJA_10perc_mean_diff.png')

else:

    seasonal = []
    season = 1838
    while season < 2009:
        this_jja = filter(lambda i: (m[i] > 5 and y[i] == season) and (m[i] < 9 and y[i] == season), range(g.data.shape[0]))
        this_djf = filter(lambda i: (m[i] == 12 and y[i] == season) or (m[i] < 3 and y[i] == season+1), range(g.data.shape[0]))
        # jja_avg = np.mean(g.data[this_jja], axis = 0)
        # djf_avg = np.mean(g.data[this_djf], axis = 0)
        # seasonal.append([season, jja_avg, djf_avg, np.abs(jja_avg - djf_avg)])
        jja = np.sort(g.data[this_jja])[::-1]
        djf = np.sort(g.data[this_djf])
        seasonal.append([season, np.abs(np.mean(jja[:np.floor(0.05*jja.shape[0])]) - np.mean(djf[:np.floor(0.05*djf.shape[0])]))])
        season += 1

    seasonal = np.array(seasonal)

    for aver in range(1,9,2):
        seasonal_aver = np.zeros((seasonal.shape[0] - aver + 1, seasonal.shape[1]))
        if aver > 1:
            seasonal_aver[:, 0] = seasonal[np.floor(aver/2) : -np.floor(aver/2), 0]
        else:
            seasonal_aver[:, 0] = seasonal[:, 0]
        seasonal_aver[:, 1] = running_mean(seasonal[:, 1], aver)
        # seasonal_aver[:, 2] = running_mean(seasonal[:, 2], aver)
        # seasonal_aver[:, 3] = running_mean(seasonal[:, 3], aver)

        fig, ax = plt.subplots(figsize=(13,8))
        ax.plot(seasonal_aver[:, 0], seasonal_aver[:, 1], linewidth = 2, color = "#A3168E")


    # ax.plot(seasonal_aver[:, 0], seasonal_aver[:, 3], linewidth = 3, color = "#3DBF2F")
        ax.set_xticks(np.arange(seasonal_aver[0,0], seasonal_aver[-1,0]+5, 15))
        ax.set_ylabel("difference max JJA temp vs min DJF temp [$^{\circ}$C]")
        ax.axis([seasonal_aver[0,0], seasonal_aver[-1,0], 25,45])
    # ax.set_xlabel("year")
    # ax.axis([seasonal_aver[0,0], seasonal_aver[-1,0], 14, 26])
    # ax2 = ax.twinx()
    # ax2.plot(seasonal_aver[:, 0], seasonal_aver[:, 1], linewidth = 1.2, color = "#EE311A")
    # ax2.plot(seasonal_aver[:, 0], seasonal_aver[:, 2], linewidth = 1.2, color = "#2FC6C8")
    # ax2.set_ylabel("DJF and JJA means [$^{\circ}$C]")
        plt.suptitle("Mean of 10percent coldest DJF vs. warmest JJA - %dseasons running mean" % aver)


        plt.savefig('debug/5perc_TG_max_min_%daver.png' % aver)

