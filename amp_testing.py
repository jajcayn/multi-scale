
from src.data_class import load_station_data
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from surrogates.surrogates import SurrogateField
from src import wavelet_analysis
import scipy.io as sio
import matplotlib.pyplot as plt


#g = load_ERA_data_daily('ERA40_EU', 't2m', date(1958,1,1), date(2009,1,1), None, None, True, parts = 3)

#def LCLtemp(t, td):
#    first = 1 / (td - 56.)
#    second = np.log(t / td) / 800.
#    
#    return (1 / (first + second)) + 56 - 273.15
#    
#def LCLheight(tlcl, t, p):
#    bracket = tlcl / (t * np.power((1000. / p), 0.286))
#    
#    return 1000. * np.power(bracket, 3.48)
# 
#   
#a = LCLtemp(13 + 273.15, 6 + 273.15)
#print 'temp [C]: ', a
#b = LCLheight(a + 273.15, 13 + 273.15, 939)
#print 'height [hPa]: ', b



g = load_station_data('TG_STAID000027.txt', date(1834, 4, 28), date(2013, 10, 1), False)
#sio.savemat('data.mat', {'data' : g.data})


ndx = g.select_date(date(1838,4,28), date(2009,10,1)) # data as with temporal evolution (of anything studied)
d, m, y = g.extract_day_month_year()

seasonal = []
season = 1838
while season < 2009:
	this_jja = filter(lambda i: (m[i] > 5 and y[i] == season) and (m[i] < 9 and y[i] == season), range(g.data.shape[0]))
	this_djf = filter(lambda i: (m[i] == 12 and y[i] == season) or (m[i] < 3 and y[i] == season+1), range(g.data.shape[0]))
	jja_avg = np.mean(g.data[this_jja], axis = 0)
	djf_avg = np.mean(g.data[this_djf], axis = 0)
	seasonal.append([season, jja_avg, djf_avg, np.abs(jja_avg - djf_avg)])

	season += 1

seasonal = np.array(seasonal)

plt.figure()
plt.plot(seasonal[:, 0], seasonal[:, 3], linewidth = 3, color = "#FBC014")
plt.xticks(np.arange(seasonal[0,0], seasonal[-1,0]+5, 15))
plt.axis([seasonal[0,0], seasonal[-1,0], 14, 26])
plt.show()

