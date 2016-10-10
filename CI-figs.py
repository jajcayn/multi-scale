import numpy as np
from src.data_class import load_station_data
from datetime import date
import matplotlib.pyplot as plt


def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))


g = load_station_data('../data/TG_STAID000027.txt', date(1775, 1, 1), date(2016, 1, 1), True)
# g = load_station_data('../data/TG_STAID000027.txt', date(1958, 1, 1), date(2013, 11, 10), True)
g_amp = load_station_data('../data/TG_STAID000027.txt', date(1775, 1, 1), date(2016, 1, 1), False)

g_amp.wavelet(1, period_unit = 'y')
reconstruction = g_amp.amplitude * np.cos(g_amp.phase)
fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
m, c = np.linalg.lstsq(fit_x, g_amp.data)[0]
amplitude = m * g_amp.amplitude + c

g.wavelet(8, 'y')
rec8y = g.amplitude * np.cos(g.phase)
fit_x = np.vstack([rec8y, np.ones(rec8y.shape[0])]).T
m, c = np.linalg.lstsq(fit_x, g.data)[0]
rec8y = m * rec8y + c
amp8y = m * g.amplitude + c


# 8-year cycle

ndx_date = g.select_date(date(1950, 1, 1), date(2014, 1, 1))

phase = g.phase[ndx_date].copy()
data = g.data.copy()
# data = amplitude[ndx_date].copy()
amp8y = amp8y[ndx_date].copy()
# data = amp8y
rec8y = rec8y[ndx_date]


phase_bins = get_equidistant_bins()
cond_means = np.zeros((8,))

for i in range(phase_bins.shape[0] - 1):
    ndx = ((phase >= phase_bins[i]) & (phase <= phase_bins[i+1]))
    cond_means[i] = np.mean(data[ndx], axis = 0)


_, _, years = g.extract_day_month_year()

plt.figure(figsize=(8.5,9))
dif = np.diff(phase_bins)[0]
plt.bar(0.025*np.diff(phase_bins)[0] + phase_bins[:-1], cond_means, width = 0.95*dif, ec = "#2A2829", fc = "#2A2829")
plt.xlim([-np.pi, np.pi])
plt.ylim([0, 1.5])
# plt.ylim([1.2, 1.4])
plt.xlabel("8-YR CYCLE PHASE [RAD]", size = 30)
plt.ylabel("SATA COND MEAN // AMP OF 8-YR [$^{\circ}C$]", size = 30)
plt.tick_params(axis='both', which='major', labelsize = 20)
ax2 = plt.gca().twiny()
ax2.plot(amp8y, color = "grey", linewidth = 2.5)
ax2.set_xlim([0,amp8y.shape[0]])
ax2.set_xticks(np.arange(0, years.shape[0])[::2290])
ax2.set_xticklabels(years[::2290], rotation = 35)
plt.tick_params(axis='both', which='major', labelsize = 20)
plt.ylim([0, 1.5])
# plt.ylabel("AAC COND MEAN [$^{\circ}C$]", size = 30)
plt.savefig("sata-cond-means.eps", bbox_inches = 'tight')
# plt.show()


