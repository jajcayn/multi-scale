from scale_network import ScaleSpecificNetwork
from datetime import date
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np


WORKERS = 10

# print "computing SATA phase ERA data networks..."
# to_do = [['MIGAU', 8], ['MIGAU', 4], ['MIGAU', 6], ['MIGAU', 11], ['MIGAU', 15],
#             ['MIEQQ', 4], ['MIEQQ', 6], ['MIEQQ', 8], ['MIEQQ', 11], ['MIEQQ', 15]]

# for do in to_do:
    # METHOD = do[0]
    # PERIOD = do[1]
    # print("computing for %d period using %s method" % (PERIOD, METHOD))
net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/ERAconcat.t2m.mon.means.1958-2014.bin', None, 
                   date(1958,1,1), date(2014,1,1), None, None, None, 'monthly', anom = True, pickled = True)
pool = Pool(WORKERS)             
net.wavelet(PERIOD, get_amplitude = False, pool = pool)
net.get_continuous_phase(pool = pool)
print "wavelet on data done"
pool.close()

net2 = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                           date(1958,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = True)
pool = Pool(WORKERS)             
net2.wavelet(PERIOD, get_amplitude = False, pool = pool)
net2.get_continuous_phase(pool = pool)
pool.close()

phase_diffs = net.phase - net2.phase
print phase_diffs.shape

# import cPickle
# with open("networks/NCEP-ERA-phase_diff.bin", "wb") as f:
#     cPickle.dump({'phase_diff' : })
# net.get_adjacency_matrix(net.phase, method = METHOD, pool = None, use_queue = True, num_workers = WORKERS)
# print "estimating adjacency matrix done"
# net.save_net('networks/ERA-SATAsurface-phase-adjmat%s-scale%dyears.bin' % (METHOD, PERIOD), only_matrix = True)