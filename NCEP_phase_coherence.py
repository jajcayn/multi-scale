from scale_network import ScaleSpecificNetwork
from datetime import date
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np

         
methods = ['MIEQQ', 'MPC']
periods = [15, 11]

for METHOD in methods:
	for PERIOD in periods:
		net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                           date(1948,1,1), date(2013,1,1), None, None, 0, 'monthly', anom = True)
		pool = Pool(3)             
		net.wavelet(PERIOD, get_amplitude = False, pool = pool)
		pool.close()
		print "wavelet on data done"
		net.get_adjacency_matrix(net.phase, method = METHOD, pool = None, use_queue = True, num_workers = 3)
		print "estimating adjacency matrix done"
		net.save_net('networks/NCEP-SATAsurface-adjmat%s-scale%dyears.bin' % (METHOD, PERIOD), only_matrix = True)
