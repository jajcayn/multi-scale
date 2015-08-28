"""
created on May 6, 2014

@author: Nikola Jajcay
"""

from src.oscillatory_time_series import OscillatoryTimeSeries
from src.data_class import DataField
from datetime import date, timedelta
import matplotlib.pyplot as plt
import numpy as np
from surrogates.surrogates import SurrogateField
import calendar


ts = OscillatoryTimeSeries('TG_STAID000027.txt', date(1834,7,28), date(2014,1,1), False)
sg = SurrogateField()
g = DataField()


daily_var = np.zeros((365,3))
mean, var_data, trend = ts.g.get_seasonality(True)
sg.copy_field(ts.g)

#MF
sg.construct_multifractal_surrogates()
sg.add_seasonality(mean, var_data, trend)

g.data = sg.surr_data.copy()
g.time = sg.time.copy()

_, var_surr_MF, _ = g.get_seasonality(True)

#FT
sg.construct_fourier_surrogates_spatial()
sg.add_seasonality(mean, var_data, trend)

g.data = sg.surr_data.copy()
g.time = sg.time.copy()

_, var_surr_FT, _ = g.get_seasonality(True)

delta = timedelta(days = 1)
d = date(1895,1,1)

for i in range(daily_var.shape[0]):
    ndx = ts.g.find_date_ndx(d)
    daily_var[i,0] = var_data[ndx]
    daily_var[i,1] = var_surr_MF[ndx]
    daily_var[i,2] = var_surr_FT[ndx]
    d += delta
    
    
    
    

p1, = plt.plot(daily_var[:, 0], linewidth = 2, color = '#AEBC76')
p2, = plt.plot(daily_var[:, 1], linewidth = 2, color = '#350E18')
p3, = plt.plot(daily_var[:, 2], linewidth = 2, color = '#8C4E5E')
plt.axis([0, 365, 2, 7])
plt.ylabel('standard deviation in temperature [$^{\circ}$C$^{2}$]')
plt.xlabel('time [months]')
plt.title('STD - data / MF surr / FT surr - %s \n %s -- %s' % (ts.g.location,
            str(ts.g.get_date_from_ndx(0)), str(ts.g.get_date_from_ndx(-1))), size = 22)
plt.legend([p1, p2, p3], ('data', 'MF surr', 'FT surr'))
plt.xticks(np.arange(1,365,365/11), calendar.month_name[1:13], rotation = 30)
plt.show()
    




