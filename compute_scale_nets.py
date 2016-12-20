from scale_network import ScaleSpecificNetwork
from datetime import date
from pathos.multiprocessing import Pool
import matplotlib.pyplot as plt
import src.wavelet_analysis as wvlt
import numpy as np

fname = '/home/nikola/Work/phd/data/air.mon.mean.sig995.nc'
# fname = "/Users/nikola/work-ui/data/NCEP/air.mon.mean.sig995.nc"
NUM_WORKERS = 20

SCALES = np.arange(24, 186, 6) # 2 - 15yrs, 0.5yr step, in months
METHODS = ['MIEQQ', 'MIKNN']


for method in METHODS:

    for scale in SCALES:

        print("Computing networks for %d month scale using %s method..." % (scale, method))

        # phase
        net = ScaleSpecificNetwork(fname, 'air', date(1948,1,1), date(2016,1,1), None, None, None, dataset = "NCEP", 
                sampling = 'monthly', anom = False)
        pool = Pool(NUM_WORKERS)
        net.wavelet(scale, 'm', pool = pool, cut = 1)
        pool.close()
        pool.join()
        net.get_adjacency_matrix(net.phase, method = method, pool = None, use_queue = True, num_workers = NUM_WORKERS)
        net.save_net('networks/NCEP-SATsurface-scale%dmonths-phase-adjmat%s.bin' % (scale, method), only_matrix = True)

        # amplitude
        net = ScaleSpecificNetwork(fname, 'air', date(1948,1,1), date(2016,1,1), None, None, None, dataset = "NCEP", 
                sampling = 'monthly', anom = False)
        pool = Pool(NUM_WORKERS)
        net.wavelet(scale, 'm', pool = pool, cut = 1)
        pool.close()
        pool.join()
        net.get_adjacency_matrix(net.amplitude, method = method, pool = None, use_queue = True, num_workers = NUM_WORKERS)
        net.save_net('networks/NCEP-SATsurface-scale%dmonths-amplitude-adjmat%s.bin' % (scale, method), only_matrix = True)

        # reconstructed signal A*cos(phi)
        net = ScaleSpecificNetwork(fname, 'air', date(1948,1,1), date(2016,1,1), None, None, None, dataset = "NCEP", 
                sampling = 'monthly', anom = False)
        pool = Pool(NUM_WORKERS)
        net.wavelet(scale, 'm', pool = pool, cut = 1)
        pool.close()
        pool.join()
        net.get_adjacency_matrix(net.amplitude * np.cos(net.phase), method = method, pool = None, use_queue = True, num_workers = NUM_WORKERS)
        net.save_net('networks/NCEP-SATsurface-scale%dmonths-reconstructed-signal-adjmat%s.bin' % (scale, method), only_matrix = True)
