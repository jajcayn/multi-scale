from scale_network import ScaleSpecificNetwork
from datetime import date
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np


net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                           date(1948,1,1), date(2013,1,1), None, None, 0, 'monthly', anom = False)
                           
         

METHOD = 'MIGAU'
PERIOD = 8
         
pool = Pool(3)             
net.wavelet(PERIOD, get_amplitude = False, pool = pool)
pool.close()
print "wavelet on data done"
net.get_adjacency_matrix(method = METHOD, pool = None, use_queue = True, num_workers = 3)
print "estimating adjacency matrix done"

print net.adjacency_matrix


# net.save_net('networks/NCEPair-surface-adjmatSCALE%d-method-%s.bin' % (PERIOD, METHOD), only_matrix = True)