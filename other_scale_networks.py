# from scale_network import ScaleSpecificNetwork
# from datetime import date
# from multiprocessing import Pool
# import numpy as np

# WORKERS = 10
# def load_nino34_wavelet_phase(start_date, end_date, anom = True):
#     from src.data_class import DataField
#     import src.wavelet_analysis as wvlt
#     from datetime import date
#     raw = np.loadtxt('/home/nikola/Work/phd/data/nino34monthly.txt')
#     data = []
#     time = []
#     for y in range(raw.shape[0]):
#         for m in range(1,13):
#             dat = float(raw[y, m])
#             data.append(dat)
#             time.append(date(int(raw[y, 0]), m, 1).toordinal())

#     g = DataField(data = np.array(data), time = np.array(time))
#     g.location = "NINO3.4"
#     g.select_date(start_date, end_date)

#     if anom:
#         g.anomalise()

#     k0 = 6. # wavenumber of Morlet wavelet used in analysis
#     fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
#     per = PERIOD * 12 # frequency of interest
#     s0 = per / fourier_factor # get scale

#     wave, _, _, _ = wvlt.continous_wavelet(g.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = 6.)
#     phase = np.arctan2(np.imag(wave), np.real(wave))[0, :]

#     return phase




# # nino34_phase = load_nino34_wavelet_phase(date(1948,1,1), date(2014,1,1), True)
# print "computing phase automutual inf. networks, 3x scale"
# to_do = [['L2', 4], ['L2', 6], ['L2', 8], ['L2', 11], ['L2', 15],
#           ['L1', 4], ['L1', 6], ['L1', 8], ['L1', 11], ['L1', 15]]
# for do in to_do:
#     METHOD = do[0]
#     PERIOD = do[1]
#     print("computing for %d period using %s method" % (PERIOD, METHOD))
#     net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
#                        date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = True)
#     pool = Pool(WORKERS)             
#     net.wavelet(PERIOD, get_amplitude = False, pool = pool)
#     print "wavelet on data done"
#     net.get_automutualinf(3*PERIOD*12, pool = pool)
#     print "automutual info function estimate done"
#     pool.close()
#     net.get_adjacency_matrix(net.automutual_info, method = METHOD, pool = None, use_queue = True, num_workers = WORKERS)
#     print "estimating adjacency matrix done"
#     net.save_net('networks/NCEP-SATAsurface-phase-automutual-to-3x-scale-adjmat%s-scale%dyears.bin' % (METHOD, PERIOD), only_matrix = True)


# print "computing phase automutual inf. networks, 2x scale"
# to_do = [['L2', 4], ['L2', 6], ['L2', 8], ['L2', 11], ['L2', 15],
#           ['L1', 4], ['L1', 6], ['L1', 8], ['L1', 11], ['L1', 15]]
# for do in to_do:
#     METHOD = do[0]
#     PERIOD = do[1]
#     print("computing for %d period using %s method" % (PERIOD, METHOD))
#     net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
#                        date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = True)
#     pool = Pool(WORKERS)             
#     net.wavelet(PERIOD, get_amplitude = False, pool = pool)
#     print "wavelet on data done"
#     net.get_automutualinf(2*PERIOD*12, pool = pool)
#     print "automutual info function estimate done"
#     pool.close()
#     net.get_adjacency_matrix(net.automutual_info, method = METHOD, pool = None, use_queue = True, num_workers = WORKERS)
#     print "estimating adjacency matrix done"
#     net.save_net('networks/NCEP-SATAsurface-phase-automutual-to-2x-scale-adjmat%s-scale%dyears.bin' % (METHOD, PERIOD), only_matrix = True)


import numpy as np
from src.data_class import DataField
import src.wavelet_analysis as wvlt
from datetime import date
from scale_network import ScaleSpecificNetwork
from datetime import date
from multiprocessing import Pool

WORKERS = 10

def load_nino34_wavelet_phase(start_date, end_date, anom = True):
    raw = np.loadtxt('/home/nikola/Work/phd/data/nino34monthly.txt')
    data = []
    time = []
    for y in range(raw.shape[0]):
        for m in range(1,13):
            dat = float(raw[y, m])
            data.append(dat)
            time.append(date(int(raw[y, 0]), m, 1).toordinal())

    g = DataField(data = np.array(data), time = np.array(time))
    g.location = "NINO3.4"
    g.select_date(start_date, end_date)

    if anom:
        g.anomalise()

    k0 = 6. # wavenumber of Morlet wavelet used in analysis
    fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
    per = PERIOD * 12 # frequency of interest
    s0 = per / fourier_factor # get scale

    wave, _, _, _ = wvlt.continous_wavelet(g.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = 6.)
    phase = np.arctan2(np.imag(wave), np.real(wave))[0, :]

    return phase


def load_NAOindex_wavelet_phase(start_date, end_date, anom = True):
    raw = np.loadtxt('/home/nikola/Work/phd/data/NAOmonthly.txt')
    data = []
    time = []
    for y in range(raw.shape[0]):
        dat = float(raw[y, 2])
        data.append(dat)
        time.append(date(int(raw[y, 0]), int(raw[y, 1]), 1).toordinal())

    g = DataField(data = np.array(data), time = np.array(time))
    g.location = "NAO"
    g.select_date(start_date, end_date)

    if anom:
        g.anomalise()

    k0 = 6. # wavenumber of Morlet wavelet used in analysis
    fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
    per = PERIOD * 12 # frequency of interest
    s0 = per / fourier_factor # get scale

    wave, _, _, _ = wvlt.continous_wavelet(g.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = 6.)
    phase = np.arctan2(np.imag(wave), np.real(wave))[0, :]

    return phase



# print "computing phase conditioned on NINO3.4"
# to_do = [6, 8, 11, 15]

# for PERIOD in to_do:
#     nino34_phase = load_nino34_wavelet_phase(date(1948,1,1), date(2014,1,1), True)
#     net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
#                            date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = False)
#     pool = Pool(WORKERS)             
#     net.wavelet(PERIOD, get_amplitude = False, pool = pool)
#     print "wavelet on data done"
#     pool.close()
#     net.get_adjacency_matrix_conditioned(nino34_phase, use_queue = True, num_workers = WORKERS)
#     print "estimating adjacency matrix done"
#     net.save_net('networks/NCEP-SAT%dy-phase-adjmatCMIEQQcondNINOphase.bin' % (PERIOD), only_matrix = True)

print "computing phase conditioned on NAO"
to_do = [4, 6, 11, 15]

for PERIOD in to_do:
    nao_phase = load_NAOindex_wavelet_phase(date(1948,1,1), date(2014,1,1), True)
    net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                           date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = False)
    print nao_phase.shape
    print net.data.shape
    # pool = Pool(WORKERS)             
    # net.wavelet(PERIOD, get_amplitude = False, pool = pool)
    # print "wavelet on data done"
    # pool.close()
    # net.get_adjacency_matrix_conditioned(nao_phase, use_queue = True, num_workers = WORKERS)
    # print "estimating adjacency matrix done"
    # net.save_net('networks/NCEP-SAT%dy-phase-adjmatCMIEQQcondNAOphase.bin' % (PERIOD), only_matrix = True)