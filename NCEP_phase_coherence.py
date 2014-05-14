from scale_network import ScaleSpecificNetwork
from datetime import date
from multiprocessing import Pool
import matplotlib.pyplot as plt


net = ScaleSpecificNetwork('/home/nikola/Work/climate/data/air.mon.mean.nc', 'air', 
                           date(1948,1,1), date(2013,1,1), 'monthly', anom = False)
                           
         
         
pool = Pool(4)             
net.wavelet(8, pool)
#print 'wavelet ready'
#print net.g.data.shape
net.get_phase_coherence_matrix(pool)
#
#
pool.close()

#plt.plot(net.phase[:, 2, 6])
#plt.plot(net.amplitude[:, 2, 6])
##plt.plot(net.g.data[:, 2, 6])
#plt.show()