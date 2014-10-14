
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
g_max = load_station_data('TX_STAID000027.txt', date(1834, 4, 28), date(2013, 10, 1), False)
g_min = load_station_data('TN_STAID000027.txt', date(1834, 4, 28), date(2013, 10, 1), False)
#sio.savemat('data.mat', {'data' : g.data})

def running_mean(arr, aver):
	out = np.zeros((arr.shape[0] - aver + 1,))
	for i in range(out.shape[0]):
		out[i] = np.mean(arr[i : i+aver])

	return out



ndx = g.select_date(date(1838,4,28), date(2009,10,1)) # data as with temporal evolution (of anything studied)
g_max.select_date(date(1838,4,28), date(2009,10,1))
g_min.select_date(date(1838,4,28), date(2009,10,1))
d, m, y = g.extract_day_month_year()

seasonal = []
season = 1838
while season < 2009:
	this_jja = filter(lambda i: (m[i] > 5 and y[i] == season) and (m[i] < 9 and y[i] == season), range(g.data.shape[0]))
	this_djf = filter(lambda i: (m[i] == 12 and y[i] == season) or (m[i] < 3 and y[i] == season+1), range(g.data.shape[0]))
	# jja_avg = np.mean(g.data[this_jja], axis = 0)
	# djf_avg = np.mean(g.data[this_djf], axis = 0)
	# seasonal.append([season, jja_avg, djf_avg, np.abs(jja_avg - djf_avg)])
	jja = np.sort(g_max.data[this_jja])[::-1]
	djf = np.sort(g_min.data[this_djf])
	seasonal.append([season, np.abs(np.mean(jja[:np.floor(0.1*jja.shape[0])]) - np.mean(djf[:np.floor(0.1*djf.shape[0])]))])

	season += 1

seasonal = np.array(seasonal)

for aver in range(1,9,2):
	seasonal_aver = np.zeros((seasonal.shape[0] - aver + 1, seasonal.shape[1]))
	if aver > 1:
		seasonal_aver[:, 0] = seasonal[np.floor(aver/2) : -np.floor(aver/2), 0]
	else:
		seasonal_aver[:, 0] = seasonal[:, 0]
	seasonal_aver[:, 1] = running_mean(seasonal[:, 1], aver)
	# seasonal_aver[:, 2] = running_mean(seasonal[:, 2], aver)
	# seasonal_aver[:, 3] = running_mean(seasonal[:, 3], aver)

	fig, ax = plt.subplots(figsize=(13,8))
	ax.plot(seasonal_aver[:, 0], seasonal_aver[:, 1], linewidth = 2, color = "#A3168E")


# ax.plot(seasonal_aver[:, 0], seasonal_aver[:, 3], linewidth = 3, color = "#3DBF2F")
	ax.set_xticks(np.arange(seasonal_aver[0,0], seasonal_aver[-1,0]+5, 15))
	ax.set_ylabel("difference max JJA temp vs min DJF temp [$^{\circ}$C]")
	ax.axis([seasonal_aver[0,0], seasonal_aver[-1,0], 30,60])
# ax.set_xlabel("year")
# ax.axis([seasonal_aver[0,0], seasonal_aver[-1,0], 14, 26])
# ax2 = ax.twinx()
# ax2.plot(seasonal_aver[:, 0], seasonal_aver[:, 1], linewidth = 1.2, color = "#EE311A")
# ax2.plot(seasonal_aver[:, 0], seasonal_aver[:, 2], linewidth = 1.2, color = "#2FC6C8")
# ax2.set_ylabel("DJF and JJA means [$^{\circ}$C]")
	plt.suptitle("Mean of 10percent coldest DJF vs. warmest JJA - %dseasons running mean" % aver)


	plt.savefig('debug/10perc_max_min_%daver.png' % aver)

