from scale_network import ScaleSpecificNetwork
from datetime import date
from pathos.multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np

# fname = '/home/nikola/Work/phd/data/air.mon.mean.levels.nc'
# fname = "/Users/nikola/work-ui/data/NCEP/air.mon.mean.levels.nc"
fname = '/home/nikola/Work/phd/data/ERAconcat.t2m.mon.means.1958-2014.bin'
NUM_WORKERS = 20

SCALES = np.arange(24, 186, 6) # 2 - 15yrs, 0.5yr step, in months
METHODS = ['MIEQQ', 'CORR', 'MIGAU', 'MPC']

# net = ScaleSpecificNetwork(fname, 'air', date(1948,1,1), date(2016,1,1), None, None, level = 0, dataset = "NCEP", 
#             sampling = 'monthly', anom = False)
# pool = Pool(NUM_WORKERS)
# net.get_hilbert_phase_amp(period = 90, width = 12, pool = pool, cut = 1)
# pool.close()
# pool.join()
# net.get_adjacency_matrix(net.phase, method = 'MPC', pool = None, use_queue = True, num_workers = NUM_WORKERS)
# net.save_net('networks/NCEP-SATsurface-7-8yrs-Hilb-phase-adjmat%s.bin' % ('MPC'), only_matrix = True)


for method in METHODS:

    for scale in SCALES:

        print("Computing networks using %s method..." % (method))

        # phase
        if method in ['MIEQQ', 'MIGAU', 'MPC']:
            # net = ScaleSpecificNetwork(fname, 'air', date(1948,1,1), date(2016,1,1), None, None, level = 0, dataset = "NCEP", 
            #         sampling = 'monthly', anom = False)
            net = ScaleSpecificNetwork(fname, 't2m', date(1958,1,1), date(2014,1,1), None, None, level=None, pickled=True,
                        sampling='monthly', anom=False)
            pool = Pool(NUM_WORKERS)
            # net.get_hilbert_phase_amp(period = 90, width = 12, pool = pool, cut = 1)
            net.wavelet(scale, period_unit='m', cut=2, pool=pool)
            pool.close()
            pool.join()
            net.get_adjacency_matrix(net.phase, method = method, pool = None, use_queue = True, num_workers = NUM_WORKERS)
            net.save_net('networks/ERA-SATsurface-scale%dmonths-phase-adjmat%s.bin' % (scale, method), only_matrix = True)

        # amplitude
        if method in ['MIEQQ', 'MIGAU', 'CORR']:
            # net = ScaleSpecificNetwork(fname, 'air', date(1948,1,1), date(2016,1,1), None, None, level = 0, dataset = "NCEP", 
            #         sampling = 'monthly', anom = False)
            net = ScaleSpecificNetwork(fname, 't2m', date(1958,1,1), date(2014,1,1), None, None, level=None, pickled=True,
                        sampling='monthly', anom=False)
            pool = Pool(NUM_WORKERS)
            # net.get_hilbert_phase_amp(period = 90, width = 12, pool = pool, cut = 1)
            net.wavelet(scale, period_unit='m', cut=2, pool=pool)
            pool.close()
            pool.join()
            net.get_adjacency_matrix(net.amplitude, method = method, pool = None, use_queue = True, num_workers = NUM_WORKERS)
            net.save_net('networks/ERA-SATsurface-scale%dmonths-amplitude-adjmat%s.bin' % (scale, method), only_matrix = True)

            # reconstructed signal A*cos(phi)
            # net = ScaleSpecificNetwork(fname, 'air', date(1948,1,1), date(2016,1,1), None, None, level = 0, dataset = "NCEP", 
            #         sampling = 'monthly', anom = False)
            net = ScaleSpecificNetwork(fname, 't2m', date(1958,1,1), date(2014,1,1), None, None, level=None, pickled=True,
                        sampling='monthly', anom=False)
            pool = Pool(NUM_WORKERS)
            # net.get_hilbert_phase_amp(period = 90, width = 12, pool = pool, cut = 1)
            net.wavelet(scale, period_unit='m', cut=2, pool=pool)
            pool.close()
            pool.join()
            net.get_adjacency_matrix(net.amplitude * np.cos(net.phase), method = method, pool = None, use_queue = True, num_workers = NUM_WORKERS)
            net.save_net('networks/ERA-SATsurface-scale%dmonths-reconstructed-signal-adjmat%s.bin' % (scale, method), only_matrix = True)
