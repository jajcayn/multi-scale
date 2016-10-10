"""
created on August 20, 2014

@author: Nikola Jajcay
"""

from src import wavelet_analysis as wvlt
from src.data_class import load_ECA_D_data_daily
from datetime import date, datetime
import numpy as np
from multiprocessing import Pool
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as sst
import cPickle


def get_equidistant_bins():
    return np.array(np.linspace(-np.pi, np.pi, 9))
    
    
def _get_oscillatory_modes(a):
    """
    Gets oscillatory modes in terms of phase and amplitude from wavelet analysis for given data.
    """
    i, j, s0, data = a
    if not np.all(np.isnan(data)):
        wave, _, _, _ = wvlt.continous_wavelet(data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = 6.)
        phase = np.arctan2(np.imag(wave), np.real(wave))
    else:
        phase = np.nan
    
    return i, j, phase
    
def _get_extremes(a):
    """
    Gets extremes from SAT data.
    """
    i, j, phase, sat, sig, phase_bins = a
    h_ex = np.zeros((8,))
    c_ex = np.zeros((8,))
    if not np.all(np.isnan(phase)):
        for iota in range(h_ex.shape[0]):
            ndx = ((phase >= phase_bins[iota]) & (phase <= phase_bins[iota+1]))
            sat_temp = sat[ndx].copy()
            
            g_e = np.greater_equal(sat_temp, np.mean(sat) + 2 * sig)
            h_ex[iota] = np.sum(g_e)
            
            l_e = np.less_equal(sat_temp, np.mean(sat) - 2 * sig)
            c_ex[iota] = np.sum(l_e)
            
    return i, j, h_ex, c_ex
    
  

PERIOD = 8
START_DATE = date(1958,1,1)  
WORKERS = 16
SAVE = False
    

if SAVE:    
    g = load_ECA_D_data_daily('tg_0.25deg_reg_v10.0.nc', 'tg', date(1950,1,1), date(2014,1,1), 
                              None, None, False)
                              
    tg_sat = g.copy_data()
    g.anomalise()

    idx = g.get_data_of_precise_length('16k', START_DATE, None, True)
    END_DATE = g.get_date_from_ndx(-1)
    tg_sat = tg_sat[idx[0]:idx[1], ...]

    print("[%s] Running wavelet analysis using %d workers..." % (str(datetime.now()), WORKERS))
    k0 = 6. # wavenumber of Morlet wavelet used in analysis
    y = 365.25 # year in days
    fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
    period = PERIOD * y # frequency of interest
    s0 = period / fourier_factor # get scale

    phase_data = np.zeros_like(g.data)

    pool = Pool(WORKERS)
    map_func = pool.map

    job_args = [ (i, j, s0, g.data[:, i, j]) for i in range(g.lats.shape[0]) for j in range(g.lons.shape[0]) ]
    job_result = map_func(_get_oscillatory_modes, job_args)
    del job_args
    # map results
    for i, j, ph in job_result:
        phase_data[:, i, j] = ph
    del job_result
    print("[%s] Wavelet done. Finding extremes for each grid point..." % (str(datetime.now())))

    IDX = g.select_date(date(START_DATE.year + 4, START_DATE.month, START_DATE.day), 
                        date(END_DATE.year - 4, END_DATE.month, END_DATE.day))

    phase_data = phase_data[IDX, ...]
    tg_sat = tg_sat[IDX, ...]

    # sigmas
    sigma = np.nanstd(tg_sat, axis = 0, ddof = 1)
    phase_bins = get_equidistant_bins()

    # result matrix
    extremes = np.zeros([2, 8] + g.get_spatial_dims())

    job_args = [ (i, j, phase_data[:, i, j], tg_sat[:, i, j], sigma[i, j], phase_bins) for i in range(g.lats.shape[0]) for j in range(g.lons.shape[0]) ]
    job_result = map_func(_get_extremes, job_args)
    del job_args, phase_data
    # map results
    for i, j, h_ex, c_ex in job_result:
        extremes[0, :, i, j] = h_ex
        extremes[1, :, i, j] = c_ex
    del job_result

    pool.close()
    pool.join()
    del pool
    print("[%s] Extremes analysis done. The array has shape of %s. Now saving..." % (str(extremes.shape), str(datetime.now())))
    with open('ECA_extremes.bin', 'wb') as f:
        cPickle.dump({'extremes' : extremes, 'lats' : g.lats, 'lons' : g.lons}, f, protocol = cPickle.HIGHEST_PROTOCOL)

else:
    print("[%s] Loading data..." % (str(datetime.now())))
    with open('/Users/nikola/Desktop/extremes/ECA_extremes.bin', 'rb') as f:
        data = cPickle.load(f)

    # plotting
    cmaps = ['bwr', 'PuOr']
    names = ['Pearson ', 'Spearman ']
    ext = ['hot extremes >2$\sigma$', 'cold extremes <-2$\sigma$']
    x = np.linspace(0., np.pi, 8)
    sin = np.sin(x)
    lats = data['lats']
    lons = data['lons']
    extremes = data['extremes']
    print extremes.shape
    del data

    fig = plt.figure(figsize=(15,15))
    gs = gridspec.GridSpec(3, 2, height_ratios = [1, 0.05, 1])
    gs.update(left = 0.05, right = 0.95, top = 0.92, bottom = 0.08, wspace = 0.2, hspace = 0.2)
    for row in range(2):
        for column in range(2):
            rowt = 0 if row == 0 else 2
            ax = plt.Subplot(fig, gs[rowt, column])
            fig.add_subplot(ax)
            lat_ndx = np.argsort(lats)
            lats = lats[lat_ndx]
            m = Basemap(projection = 'merc',
                        llcrnrlat = lats[0], urcrnrlat = lats[-1],
                        llcrnrlon = lons[0], urcrnrlon = lons[-1],
                        resolution = 'c')
                        
            m.fillcontinents(color = "#ECF0F3", lake_color = "#A9E5FF", zorder = 0)
            m.drawmapboundary(fill_color = "#A9E5FF")
            m.drawcoastlines(linewidth = 2, color = "#333333")
            m.drawcountries(linewidth = 1.5, color = "#333333")
            m.drawparallels(np.arange(25, 80, 10), linewidth = 1.2, labels = [1,0,0,0], color = "#222222")
            m.drawmeridians(np.arange(-40, 80, 10), linewidth = 1.2, labels = [0,0,0,1], color = "#222222")
            x, y = m(*np.meshgrid(lons, lats))
            levs = np.linspace(-1,1,21)
            res = np.zeros((lats.shape[0], lons.shape[0]))
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    # rows should go: Pearson, then Spearman, columns should go hot extremes, then cold extremes
                    res[i,j] = np.corrcoef(extremes[column, :, i, j], sin)[0,1] if row == 0 else sst.spearmanr(extremes[column, :, i, j], sin)[0]
            # cold extremes should be correlated with -sin, not sin, so just multiply by -1 when computing cold extremes
            res *= -1 if column == 1 else 1
            if column == 0: 
                cs_h = m.contourf(x, y, res, levels = levs, cmap = plt.get_cmap(cmaps[column]))
            else:
                cs_c = m.contourf(x, y, res, levels = levs, cmap = plt.get_cmap(cmaps[column]))
            ax.set_title(names[row] + ext[column], size = 20)

    # colorbars
    for column in range(2):        
        ax = plt.Subplot(fig, gs[1, column])
        fig.add_subplot(ax)
        if column == 0:
            cbar = plt.colorbar(cs_h, orientation = 'horizontal', cax = ax, ticks = np.linspace(-1,1,11))
        else:
            cbar = plt.colorbar(cs_c, orientation = 'horizontal', cax = ax, ticks = np.linspace(-1,1,11))
        cbar.set_label("correlation", size = 15)

    plt.suptitle("ECA&D reanalysis extremes correlation with sin / -sin from %s" %(str(START_DATE)), size = 22)

    plt.savefig('/Users/nikola/Desktop/extremes/ECA-D_extremes_corr_with_sin.png')

