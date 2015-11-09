import numpy as np
from src.data_class import load_station_data, load_AAgeomag_data, load_bin_data, DataField
from datetime import date
from src import wavelet_analysis as wvlt
from src import mutual_information as mi
import matplotlib.pyplot as plt


DAILY = False
SAMPLES = 444
SCALES_SPAN = [6, 240] # in months
STATION = False
GRID_POINTS = [[50, 15], [50, 12.5], [52.5, 12.5], [52.5, 15]]
LEVELS = ['30hPa', '1000hPa']


def load_cosmic_data(fname, start_date, end_date, anom = True, corrected = True):
    # corrected stands for if use corrected data or not
    from dateutil.relativedelta import relativedelta

    raw = open(fname).read()
    lines = raw.split('\n')
    data = []
    time = []
    d = date(int(lines[0][:4]), int(lines[0][5:7]), 1)
    delta = relativedelta(months = +1)
    for line in lines:
        row = line.split(' ')
        if len(row) < 6:
            continue
        time.append(d.toordinal())
        if corrected:
            data.append(float(row[4]))
        else:
            data.append(float(row[5]))
        d += delta

    g = DataField(data = np.array(data), time = np.array(time))
    g.location = 'Oulu cosmic data'

    g.select_date(start_date, end_date)

    if anom:
        g.anomalise()

    return g


for LEVEL in LEVELS:
    for GRID_POINT in GRID_POINTS:

        if STATION:
            temp = load_station_data('../data/TG_STAID000027.txt', date(1964, 4, 1), date(2001, 1, 1), True, to_monthly = not DAILY)
        else:
            fname = ("NCEP%s_time_series_%.1fN_%.1fE.bin" % (LEVEL, GRID_POINT[0], GRID_POINT[1]))
            temp = load_bin_data(fname, date(1964, 4, 1), date(2001, 1, 1), True)
            temp.data.shape
        fname_aa = 'aa_day.raw' if DAILY else 'aa_month1209.raw'
        # aa = load_AAgeomag_data(fname_aa, date(1950, 1, 1), date(2001, 1, 1), True, daily = DAILY)
        aa = load_cosmic_data("../data/oulu_cosmic_ray_data.dat", date(1964, 4, 1), date(2001, 1, 1), True, True)

        temp.data = temp.data[-SAMPLES:]
        temp.time = temp.time[-SAMPLES:]
        aa.data = aa.data[-SAMPLES:]
        aa.time = aa.time[-SAMPLES:]

        print temp.data.shape
        print aa.data.shape

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

        result = np.zeros((scales.shape[0],3))
        result[:, 0] = scales
        result[:, 1] = coherence
        result[:, 2] = wvlt_coherence

        # np.savetxt(fname[:-4] + "_vs_Oulu_cosmic.txt", result, fmt = '%.4f')
        # np.savetxt("station_PRG_vs_Oulu_cosmic.txt", result, fmt = '%.4f')


        plt.figure(figsize=(16,12))
        ax = plt.subplot(211)
        plt.title("COHERENCE cosmic rays vs. %s SAT %.1fN x %.1fE -- %d - %d" % (LEVEL, GRID_POINT[0], GRID_POINT[1], y1, y2), size = 30)
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
        plt.savefig(fname[:-4] + "_vs_Oulu_cosmic-till2000.png")
        # plt.savefig("station_PRG_vs_Oulu_cosmic-till2000.png")



