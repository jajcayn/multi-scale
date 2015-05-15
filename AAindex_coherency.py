import numpy as np
from src.data_class import load_station_data, load_AAgeomag_data
from datetime import date
from src import wavelet_analysis as wvlt
from src import mutual_information as mi
import matplotlib.pyplot as plt


DAILY = False
SAMPLES = 1024
SCALES_SPAN = [6, 240] # in months
STATION = True


if STATION:
    temp = load_station_data('TG_STAID000027.txt', date(1950, 1, 1), date(2001, 1, 1), True, to_monthly = not DAILY)
else:
    
fname = 'aa_day.raw' if DAILY else 'aa_month1209.raw'
aa = load_AAgeomag_data(fname, date(1950, 1, 1), date(2001, 1, 1), True, daily = DAILY)

temp.data = temp.data[-SAMPLES:]
temp.time = temp.time[-SAMPLES:]
aa.data = aa.data[-SAMPLES:]
aa.time = aa.time[-SAMPLES:]

# from now only monthly -- for daily, wavelet needs polishing !!
scales = np.arange(SCALES_SPAN[0], SCALES_SPAN[-1] + 1, 1)

k0 = 6. # wavenumber of Morlet wavelet used in analysis
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))

coherence = []
wvlt_coherence = []

for sc in scales:
    period = sc # frequency of interest in months
    s0 = period / fourier_factor # get scale
    wave_temp, _, _, _ = wvlt.continous_wavelet(temp.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
    phase_temp = np.arctan2(np.imag(wave_temp), np.real(wave_temp))[0, 12:-12] # get phases from oscillatory modes

    wave_aa, _, _, _ = wvlt.continous_wavelet(aa.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
    phase_aa = np.arctan2(np.imag(wave_aa), np.real(wave_aa))[0, 12:-12] # get phases from oscillatory modes


    # mutual information coherence
    coherence.append(mi.mutual_information(phase_aa, phase_temp, algorithm = 'EQQ2', bins = 8, log2 = False))

    # wavelet coherence
    w1 = np.complex(0, 0)
    w2 = w1; w3 = w1
    for i in range(12,aa.time.shape[0] - 12):
        w1 += wave_aa[0, i] * np.conjugate(wave_temp[0, i])
        w2 += wave_aa[0, i] * np.conjugate(wave_aa[0, i])
        w3 += wave_temp[0, i] * np.conjugate(wave_temp[0, i])
    w1 /= np.sqrt(np.abs(w2) * np.abs(w3))
    wvlt_coherence.append(np.abs(w1))

coherence = np.array(coherence)
wvlt_coherence = np.array(wvlt_coherence)


y1 = temp.get_date_from_ndx(0).year
y2 = temp.get_date_from_ndx(-1).year


ax = plt.subplot(211)
plt.title("COHERENCE AAindex vs. SAT Klementinum -- %d - %d" % (y1, y2), size = 30)
plt.plot(scales, coherence, color = "#006E91", linewidth = 2)
plt.ylabel("MI [nats]", size = 25)
plt.xlim(SCALES_SPAN)
plt.xticks(scales[6::24], scales[6::24]/12)
ax.tick_params(axis='both', which='major', labelsize=15)
ax = plt.subplot(212)
plt.plot(scales, wvlt_coherence, color = "#251F21", linewidth = 2)
plt.ylabel("wavelet coherence", size = 25)
plt.xlim(SCALES_SPAN)
plt.xticks(scales[6::24], scales[6::24]/12)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel("period [years]", size = 25)
plt.show()


