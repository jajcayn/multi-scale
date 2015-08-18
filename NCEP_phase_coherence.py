from scale_network import ScaleSpecificNetwork
from datetime import date
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np


net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                           date(1948,1,1), date(2013,1,1), 'monthly', anom = False)
                           
         

net.g.select_level(np.where(net.g.level == 1000)[0][0])

         
pool = Pool(4)             
net.wavelet(8, pool)
#print 'wavelet ready'
#print net.g.data.shape
net.get_adjacency_matrix()
##
##
#pool.close()

plt.plot(net.phase[:, 2, 6])
plt.plot(net.amplitude[:, 2, 6])
plt.show()