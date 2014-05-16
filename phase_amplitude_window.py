"""
created on May 6, 2014

@author: Nikola Jajcay
"""

from src.oscillatory_time_series import OscillatoryTimeSeries
from datetime import date
import matplotlib.pyplot as plt
import numpy as np

start_dates = [date(1775,1,1), date(1958,1,1)] # 1958
end_dates = [date(2014,1,1), date(2002,11,10)] # 2002

subtit = 'YES padding, TS is full length 1775-2014'

ts = OscillatoryTimeSeries('TG_STAID000027.txt', start_dates[0], end_dates[0], True)
ts.wavelet(8, PAD = True)

start = ts.g.find_date_ndx(date(1800,1,1))
end = ts.g.find_date_ndx(date(1979,6,8))
time = ts.g.time.copy()

#ts.phase = ts.phase[start:end]
#ts.amplitude = ts.amplitude[start:end]
#ts.g.time = ts.g.time[start:end]

phase_short = ts.phase.copy()
amp_short = ts.amplitude.copy()
#
ts = OscillatoryTimeSeries('TG_STAID000027.txt', date(1796,1,1), date(1985,6,8), True)
ts.wavelet(8, PAD = True)

s = ts.g.find_date_ndx(date(1800,1,1))
e = ts.g.find_date_ndx(date(1979,6,8))

ts.phase = ts.phase[s:e]
ts.amplitude = ts.amplitude[s:e]
#ts.g.time = ts.g.time[start:end]

cc = np.empty_like(phase_short)
cc[:] = np.nan
cc[start:end] = ts.phase.copy()

ca = np.empty_like(amp_short)
ca[:] = np.nan
ca[start:end] = ts.amplitude.copy()

phase_long = cc.copy()
amp_long = ca.copy()
#
#
#for i in range(phase_short.shape[0] - 1):
#    if np.abs(phase_short[i+1] - phase_short[i]) > 1:
#        phase_short[i+1:] += 2 * np.pi
#    if np.abs(phase_long[i+1] - phase_long[i]) > 1:
#        phase_long[i+1:] += 2 * np.pi
#             
#diff = phase_long - phase_short
#coh = np.mean(np.cos(diff)) * np.mean(np.cos(diff)) + np.mean(np.sin(diff)) * np.mean(np.sin(diff))
#
ts.g.time = time.copy()
#fig = plt.subplots(figsize=(15,9))
#p1, = plt.plot(phase_short, color = '#343434')
#p2, = plt.plot(phase_long, color = '#F8D361')
#p3, = plt.plot(diff, color = '#C48F65', linewidth = 1.75)
#year_diff = np.round((date.fromordinal(ts.g.time[-1]).year - date.fromordinal(ts.g.time[0]).year) / 10)
#xnames = np.arange(date.fromordinal(ts.g.time[0]).year, date.fromordinal(ts.g.time[-1]).year+1, year_diff)
#plt.xticks(np.linspace(0,phase_short.shape[0],len(xnames)), xnames, rotation = 30)
#plt.axis([0, phase_short.shape[0], -np.pi, 48*np.pi])
#plt.xlabel('time [years]')
#plt.ylabel('continuous phase [rad]')
#plt.legend([p1, p2, p3], ['64k experiment', '64k classic', 'difference of phases'], loc = 2)
#plt.title('Continuous phase - coherence \n mean phase coherence: %.2f \n %s' % (np.sqrt(coh), subtit))
#print 'plotting...'
#plt.show()
        
#plt.plot(phase_short)
#plt.plot(np.log2(ts.coi) / np.log2(1.1))
#plt.plot(ts.phase)
#grad = np.zeros((ts.coi.shape[0]-1))
#for i in range(grad.shape[0]):
#    grad[i] = np.abs(ts.coi[i+1] - ts.coi[i])
#plt.plot(grad)
#plt.show()

#==============================================================================
rec_short = amp_short * np.cos(phase_short)
rec_long = amp_long * np.cos(phase_long)
 
fig = plt.subplots(figsize=(15,9))
p1, = plt.plot(rec_short, color = '#343434')
p2, = plt.plot(rec_long, color = '#F8D361')
year_diff = np.round((date.fromordinal(ts.g.time[-1]).year - date.fromordinal(ts.g.time[0]).year) / 10)
xnames = np.arange(date.fromordinal(ts.g.time[0]).year, date.fromordinal(ts.g.time[-1]).year+1, year_diff)
plt.xticks(np.linspace(0,phase_short.shape[0],len(xnames)), xnames, rotation = 30)
plt.axis([0, phase_short.shape[0], -35, 40])
plt.xlabel('time [year]')
plt.ylabel('reconstruction of temperature [$^{\circ}$C]')
plt.legend([p1, p2], ['full TS length', 'longer, yes pad, then cropped to 64k'])
plt.title('Reconstruction of temperature from wavelet phase/amplitude - $t = A \cdot \cos(\phi)$ \n %s' % subtit)
plt.show()
#==============================================================================
