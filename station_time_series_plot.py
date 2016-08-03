from src.data_class import load_station_data
from src.mutual_information import mutual_information
import numpy as np
from datetime import date
import matplotlib.pyplot as plt

g = load_station_data('../data/TG_STAID000027.txt', date(1958,1,1), date(2015,11,1), False)
g.get_data_of_precise_length(14976, start_date = date(1958, 1, 1), apply_to_data = True)


mi = [mutual_information(g.data, g.data)]
for d in np.arange(1,401,1):
    mi.append(mutual_information(g.data[d:], g.data[:-d]))

plt.plot(mi)
plt.show()


# # g.wavelet(1, period_unit = 'y')
# # reconstruction = g.amplitude * np.cos(g.phase)
# # fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
# # m, c = np.linalg.lstsq(fit_x, g.data)[0]
# # amplitude = m * g.amplitude + c


# g.wavelet(8, period_unit = 'y')

# g.anomalise()



# def get_equidistant_bins():
#     return np.array(np.linspace(-np.pi, np.pi, 9))

# ndx = g.select_date(date(1950, 1, 1), date(2014, 1, 1), apply_to_data = True)
# g.phase = g.phase[ndx]
# # amplitude = amplitude[ndx]

# phase_bins = get_equidistant_bins()

# cond_means = np.zeros((8,))
# for i in range(cond_means.shape[0]): 
#     ndx = ((g.phase >= phase_bins[i]) & (g.phase <= phase_bins[i+1]))
#     cond_means[i] = np.mean(g.data[ndx])

# print cond_means
# plt.figure(figsize=(8.5,9))
# dif = np.diff(phase_bins)[0]
# plt.bar(0.025*np.diff(phase_bins)[0] + phase_bins[:-1], cond_means, width = 0.95*dif, ec = "#2A2829", fc = "#2A2829")
# plt.xlim([-np.pi, np.pi])
# # plt.ylim([19, 21])
# plt.tick_params(axis='both', which='major', labelsize = 20)
# plt.xlabel("8-YR CYCLE PHASE [RAD]", size = 30)
# plt.ylabel("SATA COND MEAN [$^{\circ}C$]", size = 30)
# plt.savefig("sata-cond-means.eps", bbox_inches = 'tight')
# plt.show()


# annual
# g.wavelet(1, period_unit = 'y')
# reconstruction = g.amplitude * np.cos(g.phase)
# fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
# m, c = np.linalg.lstsq(fit_x, g.data)[0]
# amp_ts = m * reconstruction + c
# amp1y = m * g.amplitude + c

# # 8-year
# g.wavelet(8, period_unit = 'y')
# reconstruction = g.amplitude * np.cos(g.phase)
# fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
# m, c = np.linalg.lstsq(fit_x, g.data)[0]
# slow_ts = m * reconstruction #+ c

# # clim. amp
# ndx = g.select_date(date(1934,1,1), date(1945,1,1), apply_to_data = True)

# y = 1934
# clim_amp = []
# while y < 1945:
#     this_y = g.select_date(date(y, 1, 1), date(y+1, 1, 1), apply_to_data = False)
#     sort_d = np.sort(g.data[this_y])
#     l = int(this_y[this_y == True].shape[0] * (25 / 100.))
#     clim_amp.append([g.find_date_ndx(date(y, 6, 1)), np.mean(sort_d[-l:]) - np.mean(sort_d[:l])])
#     y += 1

# clim_amp = np.array(clim_amp)
# print clim_amp.shape




# # g.data = g.data[ndx]
# amp_ts = amp_ts[ndx]
# slow_ts = slow_ts[ndx]
# amp1y = amp1y[ndx]


# plt.figure(figsize=(17,9))
# plt.plot(g.data, color = "#CEECEF", linewidth = 0.8)
# plt.plot(amp_ts, color = "#030D4F", linewidth = 2)
# plt.plot(amp1y, color = "#FFC52C", linewidth = 2)
# plt.plot(slow_ts, color = "#FB0C06", linewidth = 2)
# plt.plot(clim_amp[:, 0], clim_amp[:, 1], marker = 'o', markersize = 13, color = "#FFC52C", linestyle = 'none')
# plt.xlim([0, g.data.shape[0]])
# plt.xticks(np.arange(0,g.data.shape[0],365), np.arange(1934,1946,1), rotation = 30)
# plt.tick_params(axis='both', which='major', labelsize = 20)
# plt.ylabel("TEMPERATURE [$^{\circ}C$]", size = 30)
# plt.xlabel("YEAR", size = 30)
# # plt.show()
# plt.savefig("sat-cycles.eps", bbox_inches = "tight")

# import scipy.stats as sts

# print sts.pearsonr(slow_ts, amp1y)

