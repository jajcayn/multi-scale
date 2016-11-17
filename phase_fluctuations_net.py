from scale_network import ScaleSpecificNetwork
from datetime import date
from multiprocessing import Pool
import numpy as np
from src.data_class import DataField
import csv
import matplotlib.pyplot as plt
import src.wavelet_analysis as wvlt


# fname = '/home/nikola/Work/phd/data/air.mon.mean.sig995.nc'
fname = "/Users/nikola/work-ui/data/air.mon.mean.sig995.nc"

## PHASE FLUCTUATIONS NETWORK EQQ
net = ScaleSpecificNetwork(fname, 'air', date(1950,1,1), date(2016,1,1), None, None, None, 'monthly', anom = False)
pool = Pool(20)
net.wavelet(1, 'y', pool = pool, cut = 1)
net.get_continuous_phase(pool = pool)
print "wavelet done"
net.get_phase_fluctuations(rewrite = True, pool = pool)
print "fluctuations done"
pool.close()
pool.join()
net.get_adjacency_matrix(net.phase_fluctuations, method = "MIEQQ", pool = None, use_queue = True, num_workers = 20)
net.save_net('networks/NCEP-SATannual-phase-fluctuations-adjmatMIEQQ-bins=4.bin', only_matrix = True)


## PHASE FLUCTUATIONS NETWORK kNN
net = ScaleSpecificNetwork(fname, 'air', date(1950,1,1), date(2016,1,1), None, None, None, 'monthly', anom = False)
pool = Pool(20)
net.wavelet(1, 'y', pool = pool, cut = 1)
net.get_continuous_phase(pool = pool)
print "wavelet done"
net.get_phase_fluctuations(rewrite = True, pool = pool)
print "fluctuations done"
pool.close()
pool.join()
net.get_adjacency_matrix(net.phase_fluctuations, method = "MIKNN", pool = None, use_queue = True, num_workers = 20)
net.save_net('networks/NCEP-SATannual-phase-fluctuations-adjmatMIKNN-k=32.bin', only_matrix = True)


# ## PHASE FLUCTUATIONS CONDITION NAO
# NAOdata = DataField()
# data = np.loadtxt('/home/nikola/Work/phd/data/NAOmonthly.txt')
# nao = []
# nao_time = []
# for t in range(data.shape[0]):
#     nao_time.append(date(int(data[t, 0]), int(data[t, 1]), 1).toordinal())
#     nao.append(data[t, 2])
# NAOdata.data = np.array(nao)
# NAOdata.time = np.array(nao_time)

# NAOdata.select_date(date(1958,1,1), date(2014,1,1))

# period = 8

# k0 = 6. # wavenumber of Morlet wavelet used in analysis, suppose Morlet mother wavelet
# y = 12
# fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
# per = period * y # frequency of interest
# s0 = per / fourier_factor # get scale
# wave, _, _, _ = wvlt.continous_wavelet(NAOdata.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = 6.)
# NAOphase = np.arctan2(np.imag(wave), np.real(wave))[0, :]

# net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
#                            date(1958,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = False)

# if net.data.shape[0] != NAOphase.shape[0]:
#     print "WRONG SHAPES!"
# else:
#     pool = Pool(5)             
#     net.wavelet(8, get_amplitude = False, pool = pool)
#     print "wavelet done"
#     # net.get_phase_fluctuations(rewrite = True, pool = pool)
#     # print "fluctuations done"
#     pool.close()
#     net.get_adjacency_matrix_conditioned(cond_ts = NAOphase, use_queue = True, num_workers = 5)
#     net.save_net('networks/NCEP-SAT8y-phase-adjmatCMIEQQcondNAOphase.bin', only_matrix = True)


# ## PHASE FLUCTUATIONS CONDITION SOI
# # SOIdata = DataField()
# # soi = []
# # soi_time = []
# # with open("/home/nikola/Work/phd/data/SOImonthly.csv", "r") as f:
# #     reader = csv.reader(f)
# #     for row in reader:
# #         soi_time.append(date(int(row[0][:4]), int(row[0][4:]), 1).toordinal())
# #         soi.append(float(row[1]))

# # SOIdata.data = np.array(soi)
# # SOIdata.time = np.array(soi_time)

# # SOIdata.select_date(date(1958,1,1), date(2014,1,1))


# # net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
# #                            date(1958,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = False)

# # if net.data.shape[0] != SOIdata.data.shape[0]:
# #     print "WRONG SHAPES!"
# # else:
# #     pool = Pool(5)             
# #     net.wavelet(1, get_amplitude = False, pool = pool)
# #     print "wavelet done"
# #     net.get_phase_fluctuations(rewrite = True, pool = pool)
# #     print "fluctuations done"
# #     pool.close()
# #     net.get_adjacency_matrix_conditioned(cond_ts = SOIdata.data, use_queue = True, num_workers = 5)
# #     net.save_net('networks/NCEP-SATannual-phase-fluctuations-adjmatCMIEQQcondSOI.bin', only_matrix = True)