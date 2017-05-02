import numpy as np
from src.data_class import load_station_data
from datetime import date
import matplotlib.pyplot as plt
import scipy.stats as sts
from src.surrogates import SurrogateField

WINDOW_LENGTH = 36 # years

prg = load_station_data('../data/ECAstation-TG/TG_STAID000027.txt', date(1775, 1, 1), date(2016, 5, 1), 
    anom = False, offset = 1)

ndxs, dates = prg.get_sliding_window_indexes(window_length = WINDOW_LENGTH, window_shift = 1, unit = 'y', return_half_dates = True)
n_windows = len(ndxs)

ndx = prg.select_date(date(1989, 1, 1), date(2016, 4, 30), apply_to_data = False)

# prg_temp = prg.copy(temporal_ndx = ndx)
# print prg_temp.get_date_from_ndx(0), prg_temp.get_date_from_ndx(-1)
# prg_temp.wavelet(1, 'y', cut = 4, cut_time = False, cut_data = False, regress_amp_to_data = True)
# annual_amp = prg_temp.amplitude.copy()
# annual_phase = prg_temp.phase.copy()
# sat_data = prg_temp.data[int(4*365.25):-int(4*365.25)].copy()

# prg_temp.anomalise()
# prg_temp.wavelet(8, 'y', cut = 4, cut_time = False, cut_data = False, regress_amp_to_data = True, continuous_phase = False)
# amplitude = prg_temp.amplitude.copy()
# prg_temp.wavelet(8, 'y', cut = 4, cut_time = True, cut_data = True, regress_amp_to_data = False, continuous_phase = False)
# # amplitudeAACreg = prg_temp.amplitude.copy()

# # m, c, r, p, std_err = sts.linregress(amplitudeAACreg*np.cos(prg_temp.phase), annual_amp*np.cos(annual_phase))
# # amplitudeAACreg = m*amplitudeAACreg + c
# # c = np.mean(sat_data, axis = 0)
# # c = 0

# recon1y = annual_amp*np.cos(annual_phase)
# m, c, r, p, std_err = sts.linregress(recon1y, sat_data)
# recon1y = m*recon1y + c
# annual_amp = m*annual_amp + c

# recon8y = amplitude*np.cos(prg_temp.phase)
# m, c, r, p, std_err = sts.linregress(recon8y, prg_temp.data)
# recon8y = m*recon8y + c


# # plt.figure(figsize = (15, 8))
# # plt.gca().spines['top'].set_visible(False)
# # plt.gca().spines['right'].set_visible(False)
# # plt.gca().spines['bottom'].set_visible(False)
# # plt.gca().spines['left'].set_visible(False)
# # plt.plot(prg_temp.data, linewidth = 0.7, color = "#BCBCBC")
# # # plt.plot(recon1y, linewidth = 1.4, color = "#424242")
# # plt.plot(annual_amp, linestyle = "-", linewidth = 2.2, color = "#3299BB")
# # plt.xticks(np.arange(0, sat_data.shape[0], int(2*365.25)), np.arange(prg_temp.get_date_from_ndx(0).year, 
# #     prg_temp.get_date_from_ndx(-1).year, 2), rotation = 30, size = 20)
# # plt.ylabel("$^{\circ}$C", size = 24)
# # plt.yticks(size = 20)

# def get_equidistant_bins(bins = 8):
#     return np.array(np.linspace(-np.pi, np.pi, bins+1))

# bins = get_equidistant_bins()

# plt.twinx()
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.plot(prg_temp.phase, linestyle = ":", linewidth = 2., color = "#FF9900")
# plt.ylim([-4, 6])
# plt.yticks([-np.pi, 0, np.pi], ["$-\pi$", 0, "$\pi$"],size = 20, color = "#FF9900")
# plt.fill_between(np.arange(0, prg_temp.phase.shape[0]), -4, 6, where = prg_temp.phase < bins[1], 
#     color = "#FFD699", alpha = 0.5)
# plt.show()
# plt.savefig("PRG-SATA-cond-means.eps", bbox_inches = 'tight')

# cond_means_temp = np.zeros((8,2))
# for j in range(cond_means_temp.shape[0]): # get conditional means for current phase range
#     effect_ndx = ((prg_temp.phase >= bins[j]) & (prg_temp.phase <= bins[j+1]))
#     cond_means_temp[j, 0] = np.mean(prg_temp.data[effect_ndx])
#     cond_means_temp[j, 1] = np.mean(annual_amp[effect_ndx])


# plt.figure(figsize = (15, 8))
# plt.subplot(121)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# diff = np.diff(bins)[0]
# plt.bar(bins[:-1] + diff*0.1, cond_means_temp[:, 0], bottom = None, width = diff*0.8, fc = "#BCBCBC", align = 'edge')
# where = bins[np.argmin(cond_means_temp[:, 0])] + diff*0.5
# # where2 = bins[np.argmax(cond_means_temp[:, 0])] + diff*0.5
# plt.vlines(where, np.min(cond_means_temp[:, 0]), np.max(cond_means_temp[:, 0]), linewidth = 0.9, color = "#424242")
# plt.plot((where - diff*0.4, where+diff*0.4), (np.min(cond_means_temp[:, 0]), np.min(cond_means_temp[:, 0])), '-', 
#     color = "#424242", linewidth = 0.9)
# plt.plot((where - diff*0.4, where+diff*0.4), (np.max(cond_means_temp[:, 0]), np.max(cond_means_temp[:, 0])), '-', 
#     color = "#424242", linewidth = 0.9)
# effect = np.around(np.max(cond_means_temp[:, 0]) - np.min(cond_means_temp[:, 0]), decimals = 2)
# plt.text(where - diff*0.3, np.max(cond_means_temp[:, 0]) - 0.4, "%.2f$^{\circ}$C" % effect, horizontalalignment = "right", 
#     verticalalignment = "center", size = 24)
# plt.xlim([-np.pi, np.pi])
# plt.ylim([-0.75, 1.25])
# plt.xticks([-np.pi, 0, np.pi], ["$-\pi$", 0, "$\pi$"], size = 20)
# plt.yticks(size = 20)
# plt.ylabel("$^{\circ}$C", size = 24)

# plt.subplot(122)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# diff = np.diff(bins)[0]
# plt.bar(bins[:-1] + diff*0.1, cond_means_temp[:, 1], bottom = None, width = diff*0.8, fc = "#3299BB", align = 'edge')
# where = bins[np.argmin(cond_means_temp[:, 1])] + diff*0.5
# # where2 = bins[np.argmax(cond_means_temp[:, 1])] + diff*0.5
# plt.vlines(where, np.min(cond_means_temp[:, 1]), np.max(cond_means_temp[:, 1]), linewidth = 0.9, color = "#0C262F")
# plt.plot((where - diff*0.4, where+diff*0.4), (np.min(cond_means_temp[:, 1]), np.min(cond_means_temp[:, 1])), '-', 
#     color = "#0C262F", linewidth = 0.9)
# plt.plot((where - diff*0.4, where+diff*0.4), (np.max(cond_means_temp[:, 1]), np.max(cond_means_temp[:, 1])), '-', 
#     color = "#0C262F", linewidth = 0.9)
# effect = np.around(np.max(cond_means_temp[:, 1]) - np.min(cond_means_temp[:, 1]), decimals = 2)
# plt.text(where + diff*0.3, np.max(cond_means_temp[:, 1]) - 0.1, "%.2f$^{\circ}$C" % effect, horizontalalignment = "left", 
#     verticalalignment = "center", size = 24)
# plt.xlim([-np.pi, np.pi])
# plt.ylim([21, 22])
# plt.xticks([-np.pi, 0, np.pi], ["$-\pi$", 0, "$\pi$"], size = 20)
# plt.yticks(size = 20)
# # plt.ylabel("$^{\circ}$C", size = 24)
# # plt.show()
# plt.savefig("PRG-bins.eps", bbox_inches = 'tight')

import cPickle

for which in ['', 'JJA', 'DJF']:

    with open("PRG-8yr-effect-linear-nonlinear%s-10000-FTsurrs.bin" % (which), "rb") as f:
        raw = cPickle.load(f)

    print raw.keys()

    # nonlinear = raw['amp_windows']
    # linear = raw['mean_ampAAC_windows']
    # nonlinear_surrs = raw['amp_windows_surrs']
    # linear_surrs = raw['mean_ampAAC_windows_surrs']

    # import scipy.stats as sts
    # print sts.pearsonr(nonlinear, linear)

    nonlinear = raw['effect_windows']
    linear = raw['mean_amp_windows']
    nonlinear_surrs = raw['effect_windows_surrs']
    linear_surrs = raw['mean_amp_windows_surrs']

    # print sts.pearsonr(nonlinear, linear)



    plt.figure(figsize = (15,8))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.plot(nonlinear, linewidth = 2., color = "#3299BB")
    plt.plot(np.percentile(nonlinear_surrs, q = 95, axis = 0), linewidth = 0.8, color = "#84C2D6")
    plt.plot(np.percentile(nonlinear_surrs, q = 5, axis = 0), linewidth = 0.8, color = "#84C2D6")
    plt.fill_between(np.arange(0, nonlinear.shape[0]), np.percentile(nonlinear_surrs, q = 5, axis = 0), 
        np.percentile(nonlinear_surrs, q = 95, axis = 0), color = "#84C2D6", alpha = 0.5)
    plt.plot(linear, linewidth = 1.6, color = "#FF9900")
    plt.plot(np.percentile(linear_surrs, q = 95, axis = 0), linewidth = 0.8, color = "#FFC266")
    plt.plot(np.percentile(linear_surrs, q = 5, axis = 0), linewidth = 0.8, color = "#FFC266")
    plt.fill_between(np.arange(0, nonlinear.shape[0]), np.percentile(linear_surrs, q = 5, axis = 0), 
        np.percentile(linear_surrs, q = 95, axis = 0), color = "#FFC266", alpha = 0.5)
    plt.xticks(np.arange(0, len(ndxs), 12), [d.year for d in dates[0::12]], rotation = 30, size = 20)
    plt.yticks(size = 20)
    plt.ylabel("$^{\circ}$C", size = 24)
    if which == '':
        plt.ylim([0, 2.5])
    else:
        plt.ylim([0, 6])
    plt.text(len(ndxs)//2, 5.7, which, horizontalalignment = 'center', verticalalignment = 'center', size = 24)
    # plt.show()
    plt.savefig("plots/egu17/PRG-SATA%seffect-FT.png" % (which), bbox_inches = 'tight')


