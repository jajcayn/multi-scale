"""
created on Mar 19, 2014

@author: Nikola Jajcay
"""


from src import wavelet_analysis
from src.data_class import DataField
import numpy as np
from datetime import datetime, date
from multiprocessing import Pool



##--- load daily NCEP data ---##
# whole period 1948-2012
start_year = 1948
end_year = 2012
glist = []
Ndays = 0

print("[%s] Loading daily data..." % str(datetime.now()))
# load each .nc file and store DataField in list
for year in range(start_year, end_year+1):
    g = DataField(data_folder = '../../climate/data/SATdaily/') # relative path to SATdaily data, change for yours
    fname = ("air.sig995.%d.nc" % year)
    g.load(fname, 'air', dataset = 'NCEP', print_prog = False)
    Ndays += len(g.time)
    glist.append(g)
    
# iterate though list and append all values together
data = np.zeros((Ndays, len(glist[0].lats), len(glist[0].lons)))
time = np.zeros((Ndays,))
n = 0
for g in glist:
    Ndays_i = len(g.time)
    data[n:Ndays_i + n, ...] = g.data
    time[n:Ndays_i + n] = g.time
    n += Ndays_i
    
g = DataField(data = data, lons = glist[0].lons, lats = glist[0].lats, time = time)
del glist

## if slice temporal
g.select_date(date(1948,1,1), date(2013,1,1))

## if slice spatial
g.select_lat_lon(None, None)

## if slice level
#g.select_level()

## if anomalise
#g.anomalise() # (removes just mean)

## if normalise
#_, _ = g.get_seasonality() # (removes mean and std from data and also returns it as arrays, you should not need them)
print("[%s] Data loaded and pre-processed. Shape of the data is %s" % (str(datetime.now()), str(g.data.shape)))


PERIOD = 8 # period of wavelet, in years
WORKERS = 4 # either 0 for single thread or num for num threads

##--- wavelet analysis ---##

def _get_oscillatory_modes(a):
    """
    gets oscillatory modes in terms of phase and amplitude from wavelet analysis for given data
    supposes use of map function, either single or multi-thread
    """
    i, j, data = a
    wave, _, _, _ = wavelet_analysis.continous_wavelet(data, 1, False, wavelet_analysis.morlet, dj = 0, s0 = s0, j1 = 0, k0 = k0)
    phase = np.arctan2(np.imag(wave), np.real(wave))
    #amplitude = np.sqrt(np.power(np.real(wave),2) + np.power(np.imag(wave),2))
    
    return i, j, phase#, amplitude
    
    
# needed to compute scale in Fourier domain, not in time domain
k0 = 6. # wavenumber of Morlet wavelet used in analysis, suppose Morlet mother wavelet
y = 365.25 # year in days, for periods at least 4years totally sufficient, effectively omitting leap years
fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
period = PERIOD * y # frequency of interest
s0 = period / fourier_factor # get scale

if WORKERS == 0:
    print("[%s] Performing wavelet analysis with period %d years..." % (str(datetime.now()), PERIOD))
    pool = None
    map_function = map
else:
    print("[%s] Performing wavelet analysis with period %d years in parallel using %d threads..." % (str(datetime.now()), PERIOD, WORKERS))
    pool = Pool(processes = WORKERS)
    map_function = pool.map
        
phase = np.zeros_like(g.data)
#amplitude = np.zeros_like(g.data)

job_args = [ (i, j, g.data[:, i, j]) for i in range(g.lats.shape[0]) for j in range(g.lons.shape[0]) ]
print 'args done'
job_result = map_function(_get_oscillatory_modes, job_args)

del job_args

# map results
for i, j, ph in job_result:
    phase[:, i, j] = ph
    #amplitude[:, i, j] = am

del job_result
        
print("[%s] Wavelet analysis done. Shape of the phases is %s" % (str(datetime.now()), str(phase.shape)))

#==============================================================================
# david - tvoj turn, mas fazy pre kazdy grid point...
#==============================================================================


