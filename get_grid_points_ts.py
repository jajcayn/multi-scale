"""
created on August 12, 2014

@author: Nikola Jajcay
"""

from src.data_class import load_ECA_D_data_daily, load_ERA_data_daily, DataField
import numpy as np
from datetime import date, datetime
import cPickle



ECA = True # if False ERA-40 / ERA-Interim reanalysis will be used
LAT = 50.08
LON = 14.41
# Prague - 50.08N x 14.41E



print("[%s] Loading %s reanalysis..." % (str(datetime.now()), 'ECA & D' if ECA else 'ERA-40 / ERA-Interim'))
if ECA:
    g = load_ECA_D_data_daily('tg_0.25deg_reg_v10.0.nc', 'tg', date(1950,1,1), date(2014,1,1), 
                                [LAT - 5, LAT + 5], [LON - 5, LON + 5], False)
else:
    g = load_ERA_data_daily('ERA40_EU', 't2m', date(1958,1,1), date(2014,1,1), [LAT - 5, LAT + 5], 
                            [LON - 5, LON + 5], False, parts = 3)
                            
lat_arg = np.argmin(np.abs(LAT - g.lats))
lon_arg = np.argmin(np.abs(LON - g.lons))

ts = g.data[:, lat_arg, lon_arg].copy()
time = g.time.copy()
loc = ("GRID | lat: %.1f, lon: %.1f" % (g.lats[lat_arg], g.lons[lon_arg]))
g_grid = DataField(data = ts, time = time)
g_grid.location = loc

with open("%s_time_series_%.1fN_%.1fE.bin" % ('ECA&D' if ECA else 'ERA', g.lats[lat_arg], g.lons[lon_arg]), 'wb') as f:
    cPickle.dump({'g' : g_grid}, f, protocol = cPickle.HIGHEST_PROTOCOL)
    
print("[%s] Dumped time-series from %.1f N and %.1f E." % (str(datetime.now()), g.lats[lat_arg], g.lons[lon_arg]))
