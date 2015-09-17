from scale_network import ScaleSpecificNetwork
from datetime import date
from multiprocessing import Pool
import numpy as np
import cPickle
import sys
sys.path.append('/home/nikola/Work/phd/mutual_information')
from mutual_information import mutual_information
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits import basemap


WORKERS = 4
PLOT = True
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


def dummy_plot(ldata, lats, lons, tit, fname):
    plt.figure(figsize=(20,10))
    m = Basemap(projection = 'robin', lon_0 = 0)
    data = np.zeros((ldata.shape[0],ldata.shape[1]+1))
    data[:,:-1] = ldata
    data[:,-1] = data[:,0]
    llons = lons.tolist()
    llons.append(360)
    lons = np.array(llons)
    lat_ndx = np.argsort(lats)
    lats = lats[lat_ndx]
    data = data[lat_ndx, :]
    data, lons = basemap.shiftgrid(180.,data,lons,start=False)
    m.fillcontinents(color = "#ECF0F3", lake_color = "#A9E5FF", zorder = 0)
    m.drawmapboundary(fill_color = "#A9E5FF")
    m.drawcoastlines(linewidth = 2, color = "#333333")
    m.drawcountries(linewidth = 1.5, color = "#333333")
    m.drawparallels(np.arange(-90, 120, 30), linewidth = 1.2, labels = [1,0,0,0], color = "#222222", size = 15)
    m.drawmeridians(np.arange(-180, 180, 60), linewidth = 1.2, labels = [0,0,0,1], color = "#222222", size = 15)
    x, y = m(*np.meshgrid(lons, lats))
    cs = m.contourf(x, y, data, 20, cmap = plt.get_cmap('CMRmap_r'))
    plt.colorbar(cs)
    plt.title(tit, size = 25)
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()


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


if not PLOT:
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

else:
    
    def parse_name(f):
        l = f.split('-')
        typ = l[1][:-7]
        typ2 = l[3]
        sc = int(l[4][5:-5])
        avg = int(float(l[-1][:-4]) * 12 * sc)

        return typ, typ2, sc, avg


    import os
    for _, _, files in os.walk('networks'):
        pass

    for f in files:
        if 'autocoherence' in f:
            try:
                with open('networks/' + f, 'rb') as fl:
                    data = cPickle.load(fl)
                typ, typ2, sc, avg = parse_name(f)
                tit = ("NCEP %s -- %s autocoherence / %dyrs scale -- lags 1-%d months" % (typ, typ2, sc, avg))
                dummy_plot(data['autocoherence'], data['lats'], data['lons'], tit, 'networks/' + f[:-4] + '.png')
            except:
                pass


