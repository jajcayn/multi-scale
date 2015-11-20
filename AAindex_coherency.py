import numpy as np
from src.data_class import load_station_data, load_AAgeomag_data, load_bin_data, DataField, load_sunspot_data
from datetime import date
from src.surrogates import SurrogateField
from src import wavelet_analysis as wvlt
from src import mutual_information as mi
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool


DAILY = False
# SAMPLES = 444
SCALES_SPAN = [6, 240] # in months
STATION = True
# GRID_POINTS = [[50, 15], [50, 12.5], [52.5, 12.5], [52.5, 15]]
# LEVELS = ['30hPa', '1000hPa']
LEVELS = ['30hPa']
GRID_POINTS = ['A']
NUM_SURR = 1000


def get_continuous_phase(ph):

    for time in range(ph.shape[0]-1):
        if np.abs(ph[time+1] - ph[time]) > 1:
            ph[time+1: ] += 2 * np.pi

    return ph



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

    if NUM_SURR != 0:
        g_surr = SurrogateField()
        seasonality = g.get_seasonality(True)
        g_surr.copy_field(g)

        g.return_seasonality(seasonality[0], seasonality[1], seasonality[2])
    else:
        g_surr, seasonality = None, None

    return g, g_surr, seasonality


def load_neutron_NESDIS_data(fname, start_date, end_date, anom = True):


    raw = np.loadtxt(fname, skiprows = 2)
    data = []
    time = []
    for year in range(raw.shape[0]):
        for month in range(1,13):
            dat = float(raw[year, month])
            if dat == 9999.:
                dat = (float(raw[year, month-2]) + float(raw[year, month-1]) + float(raw[year, month+1]) + float(raw[year, month+2])) / 4.
            data.append(dat)
            time.append(date(int(raw[year,0]), month, 1).toordinal())

    g = DataField(data = np.array(data), time = np.array(time))
    g.location = ('%s cosmic data' % (fname[32].upper() + fname[33:-4]))

    g.select_date(start_date, end_date)

    if anom:
        g.anomalise()

    if NUM_SURR != 0:
        g_surr = SurrogateField()
        seasonality = g.get_seasonality()
        g_surr.copy_field(g)

        g.return_seasonality(seasonality[0], seasonality[1], None)
    else:
        g_surr, seasonality = None, None

    return g, g_surr, seasonality



def _coherency_surrogates(a):
    aa_surr, aa_seas, scales, temp = a

    aa_surr.construct_fourier_surrogates_spatial()
    # aa_surr.construct_surrogates_with_residuals()
    aa_surr.add_seasonality(aa_seas[0], aa_seas[1], aa_seas[2])

    coherence = []
    wvlt_coherence = []

    for sc in scales:
        period = sc # frequency of interest in months
        s0 = period / fourier_factor # get scale
        wave_temp, _, _, _ = wvlt.continous_wavelet(temp[:-aa_surr.max_ord], 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
        phase_temp = np.arctan2(np.imag(wave_temp), np.real(wave_temp))[0, 12:-12] # get phases from oscillatory modes

        wave_aa, _, _, _ = wvlt.continous_wavelet(aa_surr.surr_data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
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

    return coherence, wvlt_coherence


def _cmi_surrogates(a):
    aa, aa_surr, aa_seas, temp, temp_surr, temp_seas, scales = a
    aa_surr.construct_fourier_surrogates_spatial()
    aa_surr.add_seasonality(aa_seas[0], aa_seas[1], aa_seas[2])


    cmi1 = []
    cmi2 = []

    for sc in scales:
        period = sc # frequency of interest in months
        s0 = period / fourier_factor # get scale
        wave_temp, _, _, _ = wvlt.continous_wavelet(temp.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
        phase_temp = np.arctan2(np.imag(wave_temp), np.real(wave_temp))[0, 12:-12] # get phases from oscillatory modes

        wave_aa, _, _, _ = wvlt.continous_wavelet(aa_surr.surr_data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
        phase_aa_surr = np.arctan2(np.imag(wave_aa), np.real(wave_aa))[0, 12:-12] # get phases from oscillatory modes

        # cmi1
        tmp = []
        for tau in range(1,30):
            x, y, z = mi.get_time_series_condition([phase_temp, phase_aa_surr], tau = tau, dim_of_condition = 1, eta = 1, phase_diff = True)
            tmp.append(mi.cond_mutual_information(x, y, z, algorithm = 'EQQ2', bins = 4, log2 = False))
        cmi1.append(np.mean(np.array(tmp)))

        # cmi2
        tmp = []
        for tau in range(1,30):
            x, y, z = mi.get_time_series_condition([phase_aa_surr, phase_temp], tau = tau, dim_of_condition = 1, eta = 1, phase_diff = True)
            tmp.append(mi.cond_mutual_information(x, y, z, algorithm = 'EQQ2', bins = 4, log2 = False))
        cmi2.append(np.mean(np.array(tmp)))

    cmi1 = np.array(cmi1)
    cmi2 = np.array(cmi2)

    return cmi1, cmi2


## temp -> AA
idx1 = 'sunspots'
idx2 = 'Oulu CR'

for LEVEL in LEVELS:
    for GRID_POINT in GRID_POINTS:

        if STATION:
            # temp = load_station_data('../data/TG_STAID000027.txt', date(1775, 1, 1), date(2014, 1, 1), False, to_monthly = not DAILY)
            pass
        else:
            # fname = ("NCEP%s_time_series_%.1fN_%.1fE.bin" % (LEVEL, GRID_POINT[0], GRID_POINT[1]))
            # temp = load_bin_data(fname, date(1964, 4, 1), date(2014, 1, 1), True)
            pass
        # fname_aa = '../data/aa_day.raw' if DAILY else '../data/aa_month1209.raw'

        # aa = load_AAgeomag_data("../data/aa_month1209.raw", date(1964, 4, 1), date(2009, 1, 1), False, daily = DAILY)
        # aa = load_AAgeomag_data("../data/aa_month1209.raw", date(1964, 4, 1), date(2009, 1, 1), False, daily = DAILY)
        # aa_surr = SurrogateField()
        # aa_seas = aa.get_seasonality()
        # aa_surr.copy_field(aa)
        # aa.return_seasonality(aa_seas[0], aa_seas[1], None)

names = [['sunspot', 'AAindex'], ['sunspot', 'OuluCR'], ['OuluCR', 'AAindex']]

for [idx1, idx2] in names:
        if idx1 == 'OuluCR':
            temp, temp_surr, temp_seas = load_cosmic_data("../data/oulu_cosmic_ray_data.dat", date(1964, 4, 1), date(2009, 1, 1), False, True)
        elif idx1 == 'sunspot':
            temp = load_sunspot_data("../data/sunspot_monthly.txt", date(1964, 4, 1), date(2009, 1, 1), False, daily = DAILY)
            # temp.get_data_of_precise_length(length = 1024, end_date = date(2007, 1, 1), COPY = True)
            temp_surr = SurrogateField()
            temp_seas = temp.get_seasonality(True)
            temp_surr.copy_field(temp)
            temp.return_seasonality(temp_seas[0], temp_seas[1], temp_seas[2])
        elif idx1 == 'AAindex':
            temp = load_AAgeomag_data("../data/aa_month1209.raw", date(1964, 4, 1), date(2009, 1, 1), False, daily = DAILY)
            temp_surr = SurrogateField()
            temp_seas = temp.get_seasonality(True)
            temp_surr.copy_field(temp)
            temp.return_seasonality(temp_seas[0], temp_seas[1], temp_seas[2])

        if idx2 == 'OuluCR':
            aa, aa_surr, aa_seas = load_cosmic_data("../data/oulu_cosmic_ray_data.dat", date(1964, 4, 1), date(2009, 1, 1), False, True)
        elif idx2 == 'sunspot':
            aa = load_sunspot_data("../data/sunspot_monthly.txt", date(1964, 4, 1), date(2009, 1, 1), False, daily = DAILY)
            aa_surr = SurrogateField()
            aa_seas = aa.get_seasonality(True)
            aa_surr.copy_field(aa)
            aa.return_seasonality(aa_seas[0], aa_seas[1], aa_seas[2])
        elif idx2 == 'AAindex':
            aa = load_AAgeomag_data("../data/aa_month1209.raw", date(1964, 4, 1), date(2009, 1, 1), False, daily = DAILY)
            # aa.get_data_of_precise_length(length = 1024, end_date = date(2007,1,1), COPY = True)
            aa_surr = SurrogateField()
            aa_seas = aa.get_seasonality(True)
            aa_surr.copy_field(aa)
            aa.return_seasonality(aa_seas[0], aa_seas[1], aa_seas[2])


        # aa_surr.prepare_AR_surrogates(order_range = [1,10])
        # temp, _, _ = load_cosmic_data("../data/oulu_cosmic_ray_data.dat", date(1964, 4, 1), date(2009, 1, 1), False, True)
        # aa, aa_surr, aa_seas = load_neutron_NESDIS_data('../data/cosmic-ray-flux_monthly_calgary.txt',date(1964, 4, 1), date(2009, 1, 1), True)
        # temp = load_sunspot_data("../data/sunspot_monthly.txt", date(1964, 4, 1), date(2009, 1, 1), False, daily = DAILY)
        # temp_surr = SurrogateField()
        # temp_seas = temp.get_seasonality()
        # temp_surr.copy_field(temp)
        # temp.return_seasonality(temp_seas[0], temp_seas[1], None)
        # temp = load_cosmic_data("../data/oulu_cosmic_ray_data.dat", date(1964, 4, 1), date(2009, 1, 1), False, True)[0]

        # temp.data = temp.data[-SAMPLES:]
        # temp.time = temp.time[-SAMPLES:]
        # aa.data = aa.data[-SAMPLES:]
        # aa.time = aa.time[-SAMPLES:]

        print temp.data.shape
        print aa.data.shape

        # from now only monthly -- for daily, wavelet needs polishing !!
        scales = np.arange(SCALES_SPAN[0], SCALES_SPAN[-1] + 1, 1)

        k0 = 6. # wavenumber of Morlet wavelet used in analysis
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))

        coherence = []
        wvlt_coherence = []
        cmi1 = []
        cmi2 = []

        for sc in scales:
            period = sc # frequency of interest in months
            s0 = period / fourier_factor # get scale
            wave_temp, _, _, _ = wvlt.continous_wavelet(temp.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
            phase_temp = np.arctan2(np.imag(wave_temp), np.real(wave_temp))[0, 12:-12] # get phases from oscillatory modes

            wave_aa, _, _, _ = wvlt.continous_wavelet(aa.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0) # perform wavelet
            phase_aa = np.arctan2(np.imag(wave_aa), np.real(wave_aa))[0, 12:-12] # get phases from oscillatory modes


            # mutual information coherence
            coherence.append(mi.mutual_information(phase_aa, phase_temp, algorithm = 'EQQ2', bins = 8, log2 = False))

            # cmi1
            # plt.figure()
            tmp = []
            for tau in range(1,30):
                x, y, z = mi.get_time_series_condition([phase_temp, phase_aa], tau = tau, dim_of_condition = 1, eta = 0, phase_diff = True)
                tmp.append(mi.cond_mutual_information(x, y, z, algorithm = 'EQQ2', bins = 4, log2 = False))
            cmi1.append(np.mean(np.array(tmp)))
            # plt.plot(tmp, label = "1->2")

            # cmi2
            tmp = []
            for tau in range(1,30):
                x, y, z = mi.get_time_series_condition([phase_aa, phase_temp], tau = tau, dim_of_condition = 1, eta = 0, phase_diff = True)
                tmp.append(mi.cond_mutual_information(x, y, z, algorithm = 'EQQ2', bins = 4, log2 = False))
            cmi2.append(np.mean(np.array(tmp)))
            # plt.plot(tmp, label = "2->1")
            # plt.legend()
            # plt.savefig("CMItesting%dmon.png" % sc)
            # plt.close()

            # wavelet coherence
            w1 = np.complex(0, 0)
            w2 = w1; w3 = w1
            for i in range(12,aa.time.shape[0] - 12):
                w1 += wave_aa[0, i] * np.conjugate(wave_temp[0, i])
                w2 += wave_aa[0, i] * np.conjugate(wave_aa[0, i])
                w3 += wave_temp[0, i] * np.conjugate(wave_temp[0, i])
            w1 /= np.sqrt(np.abs(w2) * np.abs(w3))
            wvlt_coherence.append(np.abs(w1))

            # if sc in np.arange(32,43,1):
            #     cont_phase_temp = get_continuous_phase(phase_temp)
            #     cont_phase_aa = get_continuous_phase(phase_aa)
            #     plt.figure()
            #     plt.plot(cont_phase_temp - cont_phase_aa)
            #     plt.ylabel("PHASE DIFFERENCE")
            #     plt.xlabel("TIME")
            #     plt.savefig("stationPRG-Oulu_phase_diff_at%d.png" % (sc))

        coherence = np.array(coherence)
        wvlt_coherence = np.array(wvlt_coherence)

        cmi1 = np.array(cmi1)
        cmi2 = np.array(cmi2)

        # SURRS - coherence
        pool = Pool(4)
        args = [(aa_surr, aa_seas, scales, temp.data) for i in range(NUM_SURR)]
        results = pool.map(_coherency_surrogates, args)
        pool.close()
        pool.join()

        results = np.array(results)

        coh_sig = np.zeros_like(coherence, dtype = np.bool)
        wvlt_sig = np.zeros_like(coherence, dtype = np.bool)

        for time in range(results.shape[-1]):
            greater = np.greater(coherence[time], results[:, 0, time])
            if np.sum(greater) > 0.95*NUM_SURR:
                coh_sig[time] = True
            else:
                coh_sig[time] = False

            greater = np.greater(wvlt_coherence[time], results[:, 1, time])
            if np.sum(greater) > 0.95*NUM_SURR:
                wvlt_sig[time] = True
            else:
                wvlt_sig[time] = False

        # SURRS - cmi
        pool = Pool(4)
        args = [(aa, aa_surr, aa_seas, temp, temp_surr, temp_seas, scales) for i in range(NUM_SURR)]
        results2 = pool.map(_cmi_surrogates, args)
        pool.close()
        pool.join()

        results2 = np.array(results2)

        cmi1_sig = np.zeros_like(cmi1, dtype = np.bool)
        cmi2_sig = np.zeros_like(cmi2, dtype = np.bool)

        for time in range(results2.shape[-1]):
            greater = np.greater(cmi1[time], results2[:, 0, time])
            if np.sum(greater) > 0.95*NUM_SURR:
                cmi1_sig[time] = True
            else:
                cmi1_sig[time] = False

            greater = np.greater(cmi2[time], results2[:, 1, time])
            if np.sum(greater) > 0.95*NUM_SURR:
                cmi2_sig[time] = True
            else:
                cmi2_sig[time] = False


        y1 = temp.get_date_from_ndx(0).year
        y2 = temp.get_date_from_ndx(-1).year

        # result = np.zeros((scales.shape[0],3))
        # result[:, 0] = scales
        # result[:, 1] = coherence
        # result[:, 2] = wvlt_coherence

        # np.savetxt(fname[:-4] + "_vs_Oulu_cosmic.txt", result, fmt = '%.4f')
        # np.savetxt("station_PRG_vs_Oulu_cosmic.txt", result, fmt = '%.4f')

        import cPickle
        with open("CMI-coh-%s-%s.bin" % (idx1, idx2), "wb") as f:
            cPickle.dump({'cmi1' : cmi1, 'cmi2' : cmi2, 'results' : results, 
                'cmi1_sig' : cmi1_sig, 'cmi2_sig' : cmi2_sig, 
                'coherence' : coherence, 'wvlt_coherence' : wvlt_coherence, 'results2' : results2,
                'coh_sig' : coh_sig, 'wvlt_sig' : wvlt_sig}, f, protocol = cPickle.HIGHEST_PROTOCOL)


        plt.figure(figsize=(16,12))
        ax = plt.subplot(211)
        # plt.title("COHERENCE cosmic rays Oulu vs. %s SAT %.1fN x %.1fE -- %d - %d" % (LEVEL, GRID_POINT[0], GRID_POINT[1], y1, y2), size = 30)
        plt.title("CMI %s vs %s -- %d - %d" % (idx1, idx2, y1, y2), size = 30)
        # plt.title("COHERENCE cosmic rays %s vs. AA index -- %d - %d" % (aa.location[:-12], y1, y2), size = 30)
        plt.plot(scales, cmi1, color = "#006E91", linewidth = 2.2, label = "%s -> %s" % (idx1, idx2))
        # surr mean
        plt.plot(scales, np.mean(results2[:, 0, :], axis = 0), color = "#DDCF0B", linewidth = 1.5)
        results2 = np.sort(results2, axis = 0)
        plt.plot(scales, results2[int(0.95*NUM_SURR), 0, :], color = "#DDCF0B", linewidth = 0.8)
        plt.fill_between(scales, np.mean(results2[:, 0, :], axis = 0), results2[int(0.95*NUM_SURR), 0, :], facecolor = "#DDCF0B", alpha = 0.5)
        for time in range(cmi1.shape[0]):
            if cmi1_sig[time]:
                plt.plot(scales[time], cmi1[time], 'o', markersize = 12, color = "#006E91")
        plt.ylabel("CMI [nats]", size = 25)
        ax.legend()
        plt.xlim(SCALES_SPAN)
        plt.xticks(scales[6::24], scales[6::24]/12)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax = plt.subplot(212)
        plt.plot(scales, cmi2, color = "#251F21", linewidth = 2.2, label = "%s -> %s" % (idx2, idx1))
        plt.plot(scales, np.mean(results2[:, 1, :], axis = 0),color = "#71545E", linewidth = 1.5)
        plt.plot(scales, results2[int(0.95*NUM_SURR), 1, :], color = "#71545E", linewidth = 0.8)
        plt.fill_between(scales, np.mean(results2[:, 1, :], axis = 0), results2[int(0.95*NUM_SURR), 1, :], facecolor = "#71545E", alpha = 0.5)
        for time in range(cmi2.shape[0]):
            if cmi2_sig[time]:
                plt.plot(scales[time], cmi2[time], 'o', markersize = 12, color = "#251F21")
        plt.ylabel("CMI [nats]", size = 25)
        plt.xlim(SCALES_SPAN)
        plt.xticks(scales[6::24], scales[6::24]/12)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.xlabel("period [years]", size = 25)
        ax.legend()
        # plt.savefig(fname[:-4] + "_vs_Oulu_cosmic.png")
        # plt.savefig("AAindex_vs_Oulu_cosmic-surrs_from_cosmic_data.png")
        # plt.savefig("AAindex_vs_%s_cosmic-surrs-from-cosmic-data.png" % aa.location[:-12])
        plt.savefig("CMI%s-%s.png" % (idx1, idx2))
        plt.close()

        plt.figure(figsize=(16,12))
        ax = plt.subplot(211)
        # plt.title("COHERENCE cosmic rays Oulu vs. %s SAT %.1fN x %.1fE -- %d - %d" % (LEVEL, GRID_POINT[0], GRID_POINT[1], y1, y2), size = 30)
        plt.title("coherence %s vs %s -- %d - %d" % (idx1, idx2, y1, y2), size = 30)
        # plt.title("COHERENCE cosmic rays %s vs. AA index -- %d - %d" % (aa.location[:-12], y1, y2), size = 30)
        plt.plot(scales, coherence, color = "#006E91", linewidth = 2.2, label = "%s -> %s" % (idx1, idx2))
        # surr mean
        plt.plot(scales, np.mean(results[:, 0, :], axis = 0), color = "#DDCF0B", linewidth = 1.5)
        results = np.sort(results, axis = 0)
        plt.plot(scales, results[int(0.95*NUM_SURR), 0, :], color = "#DDCF0B", linewidth = 0.8)
        plt.fill_between(scales, np.mean(results[:, 0, :], axis = 0), results[int(0.95*NUM_SURR), 0, :], facecolor = "#DDCF0B", alpha = 0.5)
        for time in range(coherence.shape[0]):
            if coh_sig[time]:
                plt.plot(scales[time], coherence[time], 'o', markersize = 12, color = "#006E91")
        plt.ylabel("MI [nats]", size = 25)
        ax.legend()
        plt.xlim(SCALES_SPAN)
        plt.xticks(scales[6::24], scales[6::24]/12)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax = plt.subplot(212)
        plt.plot(scales, wvlt_coherence, color = "#251F21", linewidth = 2.2, label = "%s -> %s" % (idx2, idx1))
        plt.plot(scales, np.mean(results[:, 1, :], axis = 0),color = "#71545E", linewidth = 1.5)
        plt.plot(scales, results[int(0.95*NUM_SURR), 1, :], color = "#71545E", linewidth = 0.8)
        plt.fill_between(scales, np.mean(results[:, 1, :], axis = 0), results[int(0.95*NUM_SURR), 1, :], facecolor = "#71545E", alpha = 0.5)
        for time in range(wvlt_coherence.shape[0]):
            if wvlt_sig[time]:
                plt.plot(scales[time], wvlt_coherence[time], 'o', markersize = 12, color = "#251F21")
        plt.ylabel("wavelet coherence", size = 25)
        plt.xlim(SCALES_SPAN)
        plt.xticks(scales[6::24], scales[6::24]/12)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.xlabel("period [years]", size = 25)
        ax.legend()
        # plt.savefig(fname[:-4] + "_vs_Oulu_cosmic.png")
        # plt.savefig("AAindex_vs_Oulu_cosmic-surrs_from_cosmic_data.png")
        # plt.savefig("AAindex_vs_%s_cosmic-surrs-from-cosmic-data.png" % aa.location[:-12])
        plt.savefig("coherence%s-%s.png" % (idx1, idx2))



