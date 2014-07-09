"""
created on June 13, 2014

@author: Nikola Jajcay
"""

import cPickle
import hickle as hkl
from datetime import datetime, date
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np


def render_differences_map(diffs, lats, lons, fname = None):
    fig = plt.figure(figsize=(20,16))
    lat_ndx = np.argsort(lats)
    lats = lats[lat_ndx]
    m = Basemap(projection = 'merc',
                llcrnrlat = lats[0], urcrnrlat = lats[-1],
                llcrnrlon = lons[0], urcrnrlon = lons[-1],
                resolution = 'c')
                
    x, y = m(*np.meshgrid(lons, lats))
    cs = m.contourf(x, y, diffs)
    plt.colorbar(cs)
    m.drawcoastlines(linewidth = 1)
    m.drawcountries(linewidth = 0.75)
    # m.drawmapboundary()
    plt.title('test map')
    
    if fname != None:
        plt.savefig(fname)
    else:
        plt.show()



SURR_TYPE = 'MF' # MF, FT or AR
START_DATE = date(1960,1,1)
MEANS = True
ANOMALISE = True
PICKLE = True # whether to use pickled file or hickled
SIGMAS_ABOVE = 2.4
PERCENTIL = 95 



# load data 
print("[%s] Loading data..." % (str(datetime.now())))
fname = ('result/ECA-D_%s_cond_%s_data_from_%s_16k' % ('SATA' if ANOMALISE else 'SAT', 
         'means' if MEANS else 'std', str(START_DATE)))
if PICKLE:
    with open(fname + '.bin', 'rb') as f:
        data = cPickle.load(f)
else:
    data = hkl.load(fname + '.hkl')
difference_data = data['difference_data']
mean_data = data['mean_data']
lats = data['lats']
lons = data['lons']
del data

# load surrogates
print("[%s] Data loaded. Now loading surrogates..." % (str(datetime.now())))
fname = ('result/ECA-D_%s_cond_%s_%ssurrogates_from_%s_16k' % ('SATA' if ANOMALISE else 'SAT', 
             'means' if MEANS else 'std', SURR_TYPE, str(START_DATE)))
if PICKLE:
    with open(fname + '.bin', 'rb') as f:
        data = cPickle.load(f)
else:
    data = hkl.load(fname + '.hkl')
difference_surrogates = data['difference_surrogates']
mean_surrogates = data['mean surrogates']
del data
print("[%s] Surrogates loaded." % (str(datetime.now())))

# compute significance
result_sigma = np.zeros_like(difference_data)
result_percentil = np.zeros_like(difference_data)
num_surr = difference_surrogates.shape[0]
for lat in range(lats.shape[0]):
    for lon in range(lons.shape[0]):
        if np.isnan(difference_data[lat, lon]): # if on lat x lon is NaN
            result_sigma[lat, lon] = np.nan
            result_percentil[lat, lon] = np.nan
        else:
            # sigma-based significance
            sigma = np.std(difference_surrogates[:, lat, lon], axis = 0, ddof = 1)
            mean = np.mean(difference_surrogates[:, lat, lon], axis = 0)
            if difference_data[lat, lon] >= mean + SIGMAS_ABOVE*sigma:
                result_sigma[lat, lon] = difference_data[lat, lon]
            else:
                result_sigma[lat, lon] = 0 # or np.nan
                
            # percentil-based significance
            greater_mat = np.greater(difference_data[lat, lon], difference_surrogates[:, lat, lon])
            if np.sum(greater_mat) > PERCENTIL/100 * num_surr:
                result_percentil[lat, lon] = difference_data[lat, lon]
            else:
                result_percentil[lat, lon] = 0 # or np.nan
            
                
fname = ('debug/ECA-D_%s_cond_%s_%ssurrogates_from_%s_16k_above_%.1fsigma.png' % ('SATA' if ANOMALISE else 'SAT', 
             'means' if MEANS else 'std', SURR_TYPE, str(START_DATE), SIGMAS_ABOVE))
render_differences_map(result_sigma, lats, lons, fname)

fname = ('debug/ECA-D_%s_cond_%s_%ssurrogates_from_%s_16k_above_%dpercentil.png' % ('SATA' if ANOMALISE else 'SAT', 
             'means' if MEANS else 'std', SURR_TYPE, str(START_DATE), PERCENTIL))
render_differences_map(result_percentil, lats, lons, fname)

