from scale_network import ScaleSpecificNetwork
from datetime import date
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np

WORKERS = 10 


print "computing SAT wavelet coherence..."
to_do = [['WCOH', 4], ['WCOH', 6], ['WCOH', 8], ['WCOH', 11], ['WCOH', 15]]
for do in to_do:
    METHOD = do[0]
    PERIOD = do[1]
    print("computing for %d period using %s method" % (PERIOD, METHOD))
    net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                       date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = False)
    pool = Pool(WORKERS)             
    net.wavelet(PERIOD, get_amplitude = False, save_wavelet = True, pool = pool)
    print "wavelet on data done"
    pool.close()
    net.get_adjacency_matrix(net.wave, method = METHOD, pool = None, use_queue = True, num_workers = WORKERS)
    print "estimating adjacency matrix done"
    net.save_net('networks/NCEP-SATsurface-wave-adjmat%s-scale%dyears.bin' % (METHOD, PERIOD), only_matrix = True)



print "computing SATA wavelet coherence..."
to_do = [['WCOH', 4], ['WCOH', 6], ['WCOH', 8], ['WCOH', 11], ['WCOH', 15]]
for do in to_do:
    METHOD = do[0]
    PERIOD = do[1]
    print("computing for %d period using %s method" % (PERIOD, METHOD))
    net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                       date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = True)
    pool = Pool(WORKERS)             
    net.wavelet(PERIOD, get_amplitude = False, save_wavelet = True, pool = pool)
    print "wavelet on data done"
    pool.close()
    net.get_adjacency_matrix(net.wave, method = METHOD, pool = None, use_queue = True, num_workers = WORKERS)
    print "estimating adjacency matrix done"
    net.save_net('networks/NCEP-SATAsurface-wave-adjmat%s-scale%dyears.bin' % (METHOD, PERIOD), only_matrix = True)
