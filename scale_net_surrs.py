# import sys
# sys.path.append("/Users/nikola/work-ui/multi-scale/")
from scale_network import ScaleSpecificNetwork
from datetime import date
from pathos.multiprocessing import Pool
# import matplotlib.pyplot as plt
import src.wavelet_analysis as wvlt
import numpy as np
from src.surrogates import SurrogateField


NUM_SURR = 1000

# fname = "/Users/nikola/work-ui/data/NCEP/air.mon.mean.levels.nc"
fname = "/home/nikola/Work/phd/data/air.mon.mean.levels.nc"
net = ScaleSpecificNetwork(fname, 'air', date(1948,1,1), date(2016,1,1), [-60, 0], [40, 100], level = 0, dataset = "NCEP", 
                sampling = 'monthly', anom = False)

surrs = SurrogateField()
a = net.get_seasonality(detrend = True)
surrs.copy_field(net)
net.return_seasonality(a[0], a[1], a[2])

pool = Pool(20)
net.wavelet(8, 'y', cut = 1, pool = pool)
net.get_adjacency_matrix(net.phase, method = "MIEQQ", num_workers = 0, pool = pool, use_queue = False)
pool.close()
pool.join()

data_adj_matrix = net.adjacency_matrix.copy()

surrs_adj_matrices = []

for i in range(NUM_SURR):
    print("surr %d/%d computing..." % (i+1, NUM_SURR))
    pool = Pool(20)
    surrs.construct_fourier_surrogates(pool = pool)
    surrs.add_seasonality(a[0], a[1], a[2])

    net.data = surrs.get_surr()
    net.wavelet(8, 'y', cut = 1, pool = pool)
    net.get_adjacency_matrix(net.phase, method = "MIEQQ", num_workers = 0, pool = pool, use_queue = False)
    pool.close()
    pool.join()
    surrs_adj_matrices.append(net.adjacency_matrix)

import cPickle
with open("8yr-phase-scale-net-surrs-test.bin", "wb") as f:
    cPickle.dump({'data' : data_adj_matrix, 'surrs' : surrs_adj_matrices}, f, protocol = cPickle.HIGHEST_PROTOCOL)

