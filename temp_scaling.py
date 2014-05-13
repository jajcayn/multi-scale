"""
created on May 7, 2014

@author: Nikola Jajcay
"""

import numpy as np
from src.data_class import load_station_data
from datetime import date
import matplotlib.pyplot as plt


g = load_station_data('TG_STAID000027.txt', date(1775, 1, 1), date(2014, 1, 1), True)
g.get_monthly_data()

# seasons as MAM, JJA, SON, DJF

seasons_data = []
for i in range(g.data.shape[0]/4 - 1):
    seasons_data.append(np.mean(g.data[2 + i*3 : 2 + i*3 + 3], axis = 0))
    
g.data = np.array(seasons_data)


scaling_mean = []
scaling_max = []

for diff in range(1, g.data.shape[0]):
    difs = []
    for i in range(g.data.shape[0] - diff):
        difs.append(np.abs(g.data[i+diff] - g.data[i]))
    difs = np.array(difs)
    scaling_mean.append(np.mean(difs))
    scaling_max.append(difs.max())
    print diff


plt.loglog(scaling_mean)
plt.xlabel('log $\Delta$ seasons')
plt.ylabel('log difference [$^{\circ}$C]')
plt.title('Scaling of mean differences in temperature')
plt.savefig('scaling_mean_seasonly_SATA.png')


