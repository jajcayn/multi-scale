from scale_network import ScaleSpecificNetwork
from datetime import date
import numpy as np
import cPickle
from multiprocessing import Pool


periods = [4, 6, 8, 11, 15]
varnorms = [True, False]
detrends = [True, False]
cos = [True, False]



for per in periods:
    for var in varnorms:
        for det in detrends:
            for c in cos:
                print("computing for %d period with %s for varnorm, %s for detrend and %s for cosweighting" % (per, 
                    str(var), str(det), str(c)))
                net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                                            date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = True)
                if var:
                    net.get_seasonality(det)
                
                if not var and det:
                    continue

                if c:
                    net.data *= net.latitude_cos_weights()

                pool = Pool(3)             
                net.wavelet(per, get_amplitude = True, pool = pool)
                net.get_filtered_data(pool = pool)
                pool.close()

                fname = ("filt-data/SATA-1000hPa-filtered%dperiod-%svarnorm-%sdetrend-%scosweighting.bin" % (per, 
                    '' if var else 'NO', '' if det else 'NO', '' if c else 'NO'))

                with open(fname, 'wb') as f:
                    cPickle.dump({'filt. data' : net.filtered_data, 'lats' : net.lats, 'lons' : net.lons, 'time' : net.time}, 
                        f, protocol = cPickle.HIGHEST_PROTOCOL)


