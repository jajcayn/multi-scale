from scale_network import ScaleSpecificNetwork
from datetime import date
from pathos.multiprocessing import Pool
import numpy as np
from src.data_class import DataField
import csv
import matplotlib.pyplot as plt
import src.wavelet_analysis as wvlt
from src.surrogates import SurrogateField


NUM_SURRS = 1

# fname = '/home/nikola/Work/phd/data/air.mon.mean.sig995.nc'
fname = "/Users/nikola/work-ui/data/air.mon.mean.sig995.nc"

surrs = SurrogateField()

net = ScaleSpecificNetwork(fname, 'air', date(1950,1,1), date(2016,1,1), None, None, None, 'monthly', anom = False)
a = net.get_seasonality(detrend = True)
surrs.copy_field(net)
# surrs.construct_fourier_surrogates()
# surrs.add_seasonality(a[0], a[1], a[2])


for num in range(NUM_SURRS):
    pool = Pool(20)
    surrs.construct_fourier_surrogates(pool = pool)
    surrs.add_seasonality(a[0], a[1], a[2])

    net.data = surrs.get_surr()
    net.wavelet(1, 'y', pool = pool, cut = 1)
    net.get_continuous_phase(pool = pool)
    print "wavelet done"
    net.get_phase_fluctuations(rewrite = True, pool = pool)
    print "fluctuations done"
    pool.close()
    pool.join()
    net.get_adjacency_matrix(net.phase_fluctuations, method = "MIEQQ", pool = None, use_queue = True, num_workers = 20)
    net.save_net('networks/NCEP-SATannual-phase-fluctuations-adjmatMIEQQ-bins=4-FTsurr%d.bin' % (num), only_matrix = True)
