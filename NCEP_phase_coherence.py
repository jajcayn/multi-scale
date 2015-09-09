from scale_network import ScaleSpecificNetwork
from datetime import date
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np

WORKERS = 10 


print "computing SAT amplitude data networks..."
to_do = [['MIEQQ', 4], ['MIEQQ', 6], ['MIEQQ', 8], ['MIEQQ', 11], ['MIEQQ', 15], 
            ['MIGAU', 4], ['MIGAU', 6], ['MIGAU', 8], ['MIGAU', 11], ['MIGAU', 15]]
for do in to_do:
    METHOD = do[0]
    PERIOD = do[1]
    print("computing for %d period using %s method" % (PERIOD, METHOD))
    net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                       date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = False)
    pool = Pool(WORKERS)             
    net.wavelet(PERIOD, get_amplitude = True, pool = pool)
    print "wavelet on data done"
    pool.close()
    net.get_adjacency_matrix(net.amplitude, method = METHOD, pool = None, use_queue = True, num_workers = WORKERS)
    print "estimating adjacency matrix done"
    net.save_net('networks/NCEP-SATsurface-amplitude-adjmat%s-scale%dyears.bin' % (METHOD, PERIOD), only_matrix = True)


print "computing SATA amplitude data networks..."
to_do = [['MIEQQ', 4], ['MIEQQ', 6], ['MIEQQ', 8], ['MIEQQ', 11], ['MIEQQ', 15], 
            ['MIGAU', 4], ['MIGAU', 6], ['MIGAU', 8], ['MIGAU', 11], ['MIGAU', 15]]
for do in to_do:
    METHOD = do[0]
    PERIOD = do[1]
    print("computing for %d period using %s method" % (PERIOD, METHOD))
    net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                       date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = True)
    pool = Pool(WORKERS)             
    net.wavelet(PERIOD, get_amplitude = True, pool = pool)
    print "wavelet on data done"
    pool.close()
    net.get_adjacency_matrix(net.amplitude, method = METHOD, pool = None, use_queue = True, num_workers = WORKERS)
    print "estimating adjacency matrix done"
    net.save_net('networks/NCEP-SATAsurface-amplitude-adjmat%s-scale%dyears.bin' % (METHOD, PERIOD), only_matrix = True)


print "computing SAT covariance data network..."
to_do = [['COV', 4]]
for do in to_do:
    METHOD = do[0]
    PERIOD = do[1]
    print("computing using %s method" % (METHOD))
    net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                       date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = False)
    # pool = Pool(WORKERS)             
    # net.wavelet(PERIOD, get_amplitude = True, pool = pool)
    # print "wavelet on data done"
    # pool.close()
    net.get_adjacency_matrix(net.data, method = METHOD, pool = None, use_queue = True, num_workers = WORKERS)
    print "estimating adjacency matrix done"
    net.save_net('networks/NCEP-SATsurface-adjmat%s.bin' % (METHOD), only_matrix = True)


print "computing SATA covariance data network..."
to_do = [['COV', 4]]
for do in to_do:
    METHOD = do[0]
    PERIOD = do[1]
    print("computing using %s method" % (METHOD))
    net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                       date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = True)
    # pool = Pool(WORKERS)             
    # net.wavelet(PERIOD, get_amplitude = True, pool = pool)
    # print "wavelet on data done"
    # pool.close()
    net.get_adjacency_matrix(net.data, method = METHOD, pool = None, use_queue = True, num_workers = WORKERS)
    print "estimating adjacency matrix done"
    net.save_net('networks/NCEP-SATAsurface-adjmat%s.bin' % (METHOD), only_matrix = True)


print "computing SAT filtered covariance data networks..."
to_do = [['COV', 4], ['COV', 6], ['COV', 8], ['COV', 11], ['COV', 15]]
for do in to_do:
    METHOD = do[0]
    PERIOD = do[1]
    print("computing for %d period using %s method" % (PERIOD, METHOD))
    net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                       date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = False)
    pool = Pool(WORKERS)             
    net.wavelet(PERIOD, get_amplitude = True, pool = pool)
    print "wavelet on data done"
    net.get_filtered_data(pool = pool)
    print "filtered data acquired"
    pool.close()
    net.get_adjacency_matrix(net.filtered_data, method = METHOD, pool = None, use_queue = True, num_workers = WORKERS)
    print "estimating adjacency matrix done"
    net.save_net('networks/NCEP-SATsurface-filtered-AcosPHI-adjmat%s-scale%dyears.bin' % (METHOD, PERIOD), only_matrix = True)


print "computing SATA filtered covariance data networks..."
to_do = [['COV', 4], ['COV', 6], ['COV', 8], ['COV', 11], ['COV', 15]]
for do in to_do:
    METHOD = do[0]
    PERIOD = do[1]
    print("computing for %d period using %s method" % (PERIOD, METHOD))
    net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                       date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = True)
    pool = Pool(WORKERS)             
    net.wavelet(PERIOD, get_amplitude = True, pool = pool)
    print "wavelet on data done"
    net.get_filtered_data(pool = pool)
    print "filtered data acquired"
    pool.close()
    net.get_adjacency_matrix(net.filtered_data, method = METHOD, pool = None, use_queue = True, num_workers = WORKERS)
    print "estimating adjacency matrix done"
    net.save_net('networks/NCEP-SATAsurface-filtered-AcosPHI-adjmat%s-scale%dyears.bin' % (METHOD, PERIOD), only_matrix = True)

