from scale_network import ScaleSpecificNetwork
from datetime import date
from multiprocessing import Pool
import numpy as np
from src.data_class import DataField
import csv
import matplotlib.pyplot as plt


## PHASE FLUCTUATIONS NETWORK
net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                           date(1958,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = False)

pool = Pool(4)             
net.wavelet(1, get_amplitude = False, pool = pool)
print "wavelet done"
net.get_phase_fluctuations(rewrite = True, pool = pool)
print "fluctuations done"
pool.close()
net.get_adjacency_matrix(method = "MIGAU", pool = None, use_queue = True, num_workers = 4)
net.save_net('networks/NCEP-SATannual-phase-fluctuations-adjmatMIGAU.bin', only_matrix = True)


## PHASE FLUCTUATIONS CONDITION NAO
NAOdata = DataField()
data = np.loadtxt('/home/nikola/Work/phd/data/NAOmonthly.txt')
nao = []
nao_time = []
for t in range(data.shape[0]):
    nao_time.append(date(int(data[t, 0]), int(data[t, 1]), 1).toordinal())
    nao.append(data[t, 2])
NAOdata.data = np.array(nao)
NAOdata.time = np.array(nao_time)

NAOdata.select_date(date(1958,1,1), date(2014,1,1))

net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                           date(1958,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = False)

if net.data.shape[0] != NAOdata.data.shape[0]:
    print "WRONG SHAPES!"
else:
    pool = Pool(4)             
    net.wavelet(1, get_amplitude = False, pool = pool)
    print "wavelet done"
    net.get_phase_fluctuations(rewrite = True, pool = pool)
    print "fluctuations done"
    pool.close()
    net.get_adjacency_matrix_conditioned(cond_ts = NAOdata.data, use_queue = True, num_workers = 4)
    net.save_net('networks/NCEP-SATannual-phase-fluctuations-adjmatMIEQQcondNAO.bin', only_matrix = True)


## PHASE FLUCTUATIONS CONDITION SOI
SOIdata = DataField()
soi = []
soi_time = []
with open("/home/nikola/Work/phd/data/SOImonthly.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        soi_time.append(date(int(row[0][:4]), int(row[0][4:]), 1).toordinal())
        soi.append(float(row[1]))

SOIdata.data = np.array(soi)
SOIdata.time = np.array(soi_time)

SOIdata.select_date(date(1958,1,1), date(2014,1,1))


net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                           date(1958,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = False)

if net.data.shape[0] != SOIdata.data.shape[0]:
    print "WRONG SHAPES!"
else:
    pool = Pool(4)             
    net.wavelet(1, get_amplitude = False, pool = pool)
    print "wavelet done"
    net.get_phase_fluctuations(rewrite = True, pool = pool)
    print "fluctuations done"
    pool.close()
    net.get_adjacency_matrix_conditioned(cond_ts = SOIdata.data, use_queue = True, num_workers = 4)
    net.save_net('networks/NCEP-SATannual-phase-fluctuations-adjmatMIEQQcondSOI.bin', only_matrix = True)