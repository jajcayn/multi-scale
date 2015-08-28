from scale_network import ScaleSpecificNetwork
from datetime import date
from multiprocessing import Pool
import numpy as np
import cPickle
import sys
sys.path.append('/home/nikola/Work/phd/mutual_information')
from mutual_information import mutual_information


WORKERS = 4
periods = [4, 6, 8, 11]
avg_to = [0.5, 3]


def _get_autocoherence(a):
    i, j, avg, ts = a

    coh = []
    for tau in range(1, avg):
        ts0 = ts[tau:].copy()
        ts1 = ts[:-tau].copy()
        coh.append(mutual_information(ts0, ts1, algorithm = 'EQQ2', bins = 4, log2 = False))

    return i, j, np.mean(np.array(coh))


## autocoherence phase
# print "computing autocoherence for phase..."
# for PERIOD in periods:
#     for AVG in avg_to:
#         print("computing for %d year period and averaging up to %d" % (PERIOD, 12*AVG*PERIOD))
#         net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
#                                    date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = True)
#         pool = Pool(WORKERS)             
#         net.wavelet(PERIOD, get_amplitude = False, pool = pool)
#         print "wavelet on data done"
#         autocoherence = np.zeros(net.get_spatial_dims())
#         job_args = [ (i, j, int(AVG*12*PERIOD), net.phase[:, i, j]) for i in range(net.lats.shape[0]) for j in range(net.lons.shape[0]) ]
#         job_result = pool.map(_get_autocoherence, job_args)
#         del job_args
#         pool.close()
#         for i, j, res in job_result:
#             autocoherence[i, j] = res
#         del job_result

#         with open("networks/NCEP-SATAsurface-autocoherence-phase-scale%dyears-avg-to-%.1f.bin" % (PERIOD, AVG), "wb") as f:
#             cPickle.dump({'autocoherence' : autocoherence, 'lats' : net.lats, 'lons' : net.lons}, f, protocol = cPickle.HIGHEST_PROTOCOL)


## autocoherence filtered data - SATA
print "computing autocoherence for SATA filtered data"
for PERIOD in periods:
    for AVG in avg_to:
        print("computing for %d year period and averaging up to %d" % (PERIOD, 12*AVG*PERIOD))
        net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                                   date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = True)
        pool = Pool(WORKERS)             
        net.wavelet(PERIOD, get_amplitude = True, pool = pool)
        print "wavelet on data done"
        net.get_filtered_data(pool = pool)
        print "filtered data acquired"
        autocoherence = np.zeros(net.get_spatial_dims())
        job_args = [ (i, j, int(AVG*12*PERIOD), net.filtered_data[:, i, j]) for i in range(net.lats.shape[0]) for j in range(net.lons.shape[0]) ]
        job_result = pool.map(_get_autocoherence, job_args)
        del job_args
        pool.close()
        for i, j, res in job_result:
            autocoherence[i, j] = res
        del job_result

        with open("networks/NCEP-SATAsurface-autocoherence-filtered-scale%dyears-avg-to-%.1f.bin" % (PERIOD, AVG), "wb") as f:
            cPickle.dump({'autocoherence' : autocoherence, 'lats' : net.lats, 'lons' : net.lons}, f, protocol = cPickle.HIGHEST_PROTOCOL)


## autocoherence filtered data - SAT
print "computing autocoherence for SAT filtered data"
for PERIOD in periods:
    for AVG in avg_to:
        print("computing for %d year period and averaging up to %d" % (PERIOD, 12*AVG*PERIOD))
        net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                                   date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = False)
        pool = Pool(WORKERS)             
        net.wavelet(PERIOD, get_amplitude = True, pool = pool)
        print "wavelet on data done"
        net.get_filtered_data(pool = pool)
        print "filtered data acquired"
        autocoherence = np.zeros(net.get_spatial_dims())
        job_args = [ (i, j, int(AVG*12*PERIOD), net.filtered_data[:, i, j]) for i in range(net.lats.shape[0]) for j in range(net.lons.shape[0]) ]
        job_result = pool.map(_get_autocoherence, job_args)
        del job_args
        pool.close()
        for i, j, res in job_result:
            autocoherence[i, j] = res
        del job_result

        with open("networks/NCEP-SATsurface-autocoherence-filtered-scale%dyears-avg-to-%.1f.bin" % (PERIOD, AVG), "wb") as f:
            cPickle.dump({'autocoherence' : autocoherence, 'lats' : net.lats, 'lons' : net.lons}, f, protocol = cPickle.HIGHEST_PROTOCOL)
