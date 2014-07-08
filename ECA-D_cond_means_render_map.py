"""
created on June 13, 2014

@author: Nikola Jajcay
"""

import cPickle
from datetime import datetime, date
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np


def render_differences_map(diffs, surr_diffs, lats, lons, fname = None):
    fig = plt.figure(figsize=(15,12))
    lat_ndx = np.argsort(lats)
    lats = lats[lat_ndx]
    m = Basemap(projection = 'merc',
                llcrnrlat = lats[0], urcrnrlat = lats[-1],
                llcrnrlon = lons[0], urcrnrlon = lons[-1],
                resolution = 'c')
                
    x, y = m(*np.meshgrid(lons, lats))
    cs = m.contourf(x, y, diffs, levels = np.arange(-1,1,0.1))
    plt.colorbar(cs)
    m.drawcoastlines(linewidth = 1.5)
    m.drawmapboundary()
    plt.title('test map')
    
    if fname != None:
        plt.savefig(fname)
    else:
        plt.show()



SURR_TYPE = 'MF' # MF, FT or AR
START_DATE = date(1960,1,1)
MEANS = True
ANOMALISE = True



# load data 
print("[%s] Loading data..." % (str(datetime.now())))
fname = ('result/ECA-D_%s_cond_%s_data_from_%s_16k.bin' % ('SATA' if ANOMALISE else 'SAT', 
         'means' if MEANS else 'std', str(START_DATE)))
with open(fname, 'rb') as f:
    data = cPickle.load(f)
difference_data = data['difference_data']
mean_data = data['mean_data']
lats = data['lats']
lons = data['lons']
del data

# load surrogates
print("[%s] Data loaded. Now loading surrogates..." % (str(datetime.now())))
fname = ('result/ECA-D_%s_cond_%s_%ssurrogates_from_%s_16k.bin' % ('SATA' if ANOMALISE else 'SAT', 
             'means' if MEANS else 'std', SURR_TYPE, str(START_DATE)))
with open(fname, 'rb') as f:
    data = cPickle.load(f) 
difference_surrogates = data['difference_surrogates']
mean_surrogates = data['mean surrogates']
del data
print("[%s] Surrogates loaded." % (str(datetime.now())))

render_differences_map(mean_data, None, lats, lons, 'test.png')

