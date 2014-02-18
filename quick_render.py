from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from src.data_class import DataField
from datetime import date




def render_geo_field(data, lats, lons, u = None, v = None, symm = False, 
                     title = 'Test image', cbar_label = 'SLP Anomalies [Pa]', filename = None):

    plt.figure(figsize=(12,10), dpi=300)
    lat_ndx = np.argsort(lats)
    lats = lats[lat_ndx]
    #data = data[::-1, :]
    m = Basemap(projection = 'merc',
                llcrnrlat = lats[0], urcrnrlat = lats[-1],
                llcrnrlon = lons[0], urcrnrlon = lons[-1],
                resolution = 'c')
    x, y = m(*np.meshgrid(lons, lats))
    mi = -47.#np.min(data)
    ma = 18.#np.max(data)
    if symm:
        if abs(ma) > abs(mi):
            mi = -ma
        else:
            ma = -mi
    step = (ma-mi)/100
    levels = np.arange(mi, ma + step, step)
    cs = m.contourf(x, y, data, levels = levels)
    #if (u != None) & (v != None):
    #    q = m.quiver(x, y, u, v, width = 0.0015)
    #    qk = plt.quiverkey(q, -0.05, 1.1, 2, '$2 ms^{-1}$', coordinates = 'axes', labelpos = 'N')
    plt.clim(mi, ma)
    m.drawcoastlines(linewidth = 1.5)
    m.drawmapboundary()
    #m.drawparallels(np.arange(lats[0], lats[-1]+1, 10), dashes = [1,3], labels = [0,0,0,0], color = (.2, .2, .2), 
                    #fontsize = 8.5)
    #m.drawmeridians(np.arange(lons[0], lons[-1]+1, 10), dashes = [1,3], labels = [0,0,0,0], color = (.2, .2, .2), 
                    #fontsize = 8.5)
    plt.title(title)
    cbar = plt.colorbar(format = r"%2.2f", shrink = 0.75, ticks = np.arange(mi, ma + step, (ma-mi)/8), 
                        aspect = 25, drawedges = False)
    cbar.set_label(cbar_label)
    cbar_obj = plt.getp(cbar.ax.axes, 'yticklabels')
    plt.setp(cbar_obj, fontsize = 10, color = (.1, .1, .1))
    if filename != None:
        plt.savefig(filename)
    else:
        plt.show()
        
        
g = DataField()
g.load('tg_0.25deg_reg_v9.0.nc', 'tg')

for i in range(59):
    render_geo_field(g.data[i, ...], g.lats, g.lons, None, None, False, 'Temperature %s' % str(date.fromordinal(g.time[i])), 		              'temperature [$^{\circ}C$]', 'imgs/temp%s.png' % str(i+1))
