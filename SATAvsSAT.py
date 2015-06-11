from src import wavelet_analysis
from src.data_class import load_station_data
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



AMP_PERIODS = [1, 8]

# g_amp = load_station_data('TG_STAID000027.txt', date(1924,1,14), date(2013,10,1), False)
# g_amp_sata = load_station_data('TG_STAID000027.txt', date(1924,1,14), date(2013,10,1), True)
g_amp = load_station_data('TG_STAID000027.txt', date(1775,1,1), date(2014,1,1), False)
g_amp_sata = load_station_data('TG_STAID000027.txt', date(1775,1,1), date(2014,1,1), True)

k0 = 6. # wavenumber of Morlet wavelet used in analysis
y = 365.25 # year in days
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))

result = dict()

start_cut = date(1779,1,1)
# _, _, idx = g_amp.get_data_of_precise_length('16k', start_cut, None, False)
idx = g_amp.select_date(start_cut, date(2010,1,1), apply_to_data = False)
fields = [g_amp, g_amp_sata]

for i in range(len(AMP_PERIODS)):
    period = AMP_PERIODS[i] * 365.25 # frequency of interest
    s0_amp = period / fourier_factor # get scale
    wave, _, _, _ = wavelet_analysis.continous_wavelet(fields[i].data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0_amp, j1 = 0, k0 = k0) # perform wavelet
    amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))
    amplitude = amplitude[0, :]
    phase_amp = np.arctan2(np.imag(wave), np.real(wave))
    phase_amp = phase_amp[0, :]

    # fitting oscillatory phase / amplitude to actual SAT
    reconstruction = amplitude * np.cos(phase_amp)
    fit_x = np.vstack([reconstruction, np.ones(reconstruction.shape[0])]).T
    m, c = np.linalg.lstsq(fit_x, fields[i].data)[0]
    amplitude = m * amplitude + c
    amp = m * reconstruction + c
    # result[i] = amp[idx[0] : idx[1]]
    result[i] = amp[idx]
    # result[i+2] = amplitude[idx[0] : idx[1]]
    result[i+2] = amplitude[idx]
    # result[i+2] = phase_amp[idx[0] : idx[1]]
    print("Oscillatory series fitted to SAT data with coeff. %.3f and intercept %.3f" % (m, c))
    # fields[i].anomalise()

# g_amp.time = g_amp.time[idx[0] : idx[1]]
g_amp.time = g_amp.time[idx]
g_amp.data = g_amp.data[idx]
clim_amp = []
y = g_amp.get_date_from_ndx(0).year
percentil = 25
while y < g_amp.get_date_from_ndx(-1).year:
    # jja = g_amp.select_date(date(y, 6, 1), date(y, 9, 1), apply_to_data = False)
    # djf = g_amp.select_date(date(y, 12, 1), date(y+1, 3, 1), apply_to_data = False)
    # clim_amp.append([g_amp.find_date_ndx(date(y, 6, 1)), np.mean(g_amp.data[jja]) - np.mean(g_amp.data[djf])])
    # clim_amp.append([g_amp.find_date_ndx(date(y, 6, 1)), g_amp.data[jja].max() - g_amp.data[djf].min()])
    this_y = g_amp.select_date(date(y, 1, 1), date(y+1, 1, 1), apply_to_data = False)
    y += 1
    sort_d = np.sort(g_amp.data[this_y])
    l = int(this_y[this_y == True].shape[0] * (percentil / 100.))
    clim_amp.append([g_amp.find_date_ndx(date(y, 6, 1)), np.mean(sort_d[-l:]) - np.mean(sort_d[:l])])

clim_amp = np.array(clim_amp)
# avg = np.mean(clim_amp, axis = 0)[1]

specific_idx = g_amp.select_date(date(1933, 10, 20), date(1944, 10, 2), apply_to_data = True)
specific_idx_clim = specific_idx[[int(i) for i in clim_amp[:, 0]]]

if g_amp.get_date_from_ndx(0).month > 6:
    first_ndx = g_amp.find_date_ndx(date(g_amp.get_date_from_ndx(0).year + 1, 6, 1))
else:
    first_ndx = g_amp.find_date_ndx(date(g_amp.get_date_from_ndx(0).year, 6, 1))

subtract = clim_amp[specific_idx_clim, 0][0] - first_ndx

# fig = plt.figure(figsize = (15,10), dpi = 1200)
fig, ax = plt.subplots(figsize = (15,10), dpi = 1200)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(color = '#6A4A3C')
p1, = plt.plot(result[2][specific_idx], color = "#4F4975", linewidth = 1.5)
p2, = plt.plot(result[1][specific_idx], color = "#B87BA5", linewidth = 1.5)
p4, = plt.plot(g_amp.data, color = "#C1682A", linewidth = 0.25)
p3, = plt.plot(clim_amp[specific_idx_clim, 0] - subtract, clim_amp[specific_idx_clim, 1], 'o', markersize = 8, color = "#24751F") #  - avg + 20
plt.plot(clim_amp[specific_idx_clim, 0] - subtract, clim_amp[specific_idx_clim, 1], linewidth = 0.5, color = "#24751F")
plt.plot(result[0][specific_idx], color = "#110D29", linewidth = 1.2, alpha = 0.8)
# p4, = plt.plot(result[3], color = "#050505")
plt.ylabel("TEMPERATURE [$^{\circ}$C]", size = 20)
# plt.legend([p1,p2,p3,p4], ['1-year SAT amp.', '8-year SATA recon.', '%.2f warm vs cold' % (percentil/100.), 'SAT data'])
tp = result[1][specific_idx].shape[0]
plt.axis([0, tp, -10, 30])
plt.yticks(np.linspace(-20, 30, 11), np.linspace(-20, 30, 11))
locs, _ = plt.xticks()
locs_new = np.linspace(locs[0], locs[-1], 2*locs.shape[0] - 1)
ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
plt.xticks(locs_new, ['%d / %d' % (g_amp.get_date_from_ndx(i).month, g_amp.get_date_from_ndx(i).year) for i in locs_new[:-1]], rotation = 30)
plt.title("%s 1-year SAT vs. 8-year SATA \n %s -- %s \n Perason 1-year SAT amp vs. 8-year SATA recon.: %.2f" % (g_amp.location, 
    str(g_amp.get_date_from_ndx(0)), str(g_amp.get_date_from_ndx(-1)), np.corrcoef(-result[2][specific_idx], result[1][specific_idx])[0,1]))
# plt.show()
plt.savefig('debug/cycles1ySAT_8ySATA_%s-%s_%.2fcorr.eps' % (str(g_amp.get_date_from_ndx(0)), str(g_amp.get_date_from_ndx(-1)), np.corrcoef(-result[2][specific_idx], result[1][specific_idx])[0,1]))


to_txt = np.zeros((result[2][specific_idx].shape[0], 6))
# continuous year
first_day = (g_amp.get_date_from_ndx(0) - date(g_amp.get_date_from_ndx(0).year, 1, 1)).days
first_ndx = g_amp.get_date_from_ndx(0).year + (first_day / 365.)

last_day = (g_amp.get_date_from_ndx(-1) - date(g_amp.get_date_from_ndx(-1).year, 1, 1)).days
last_ndx = g_amp.get_date_from_ndx(-1).year + (last_day / 365.)

# first row is date
to_txt[:, 0] = np.linspace(first_ndx, last_ndx, to_txt.shape[0])
# second is SAT data
to_txt[:, 1] = g_amp.data
# third is reconstruction of annual cycle
to_txt[:, 2] = result[0][specific_idx]
# fourth is annual amp
to_txt[:, 3] = result[2][specific_idx]
# fifth is 8-year reconstruction
to_txt[:, 4] = result[1][specific_idx]
# sixth is climatological amp.
to_txt[:, 5] = -50. # set all outside visible area
idx = [int(i) for i in clim_amp[specific_idx_clim, 0] - subtract]
to_txt[idx, 5] = clim_amp[specific_idx_clim, 1] # set the rights ones to values

np.savetxt('debug/SATAvsSAT_test.txt', to_txt, fmt = '%.3f')




