"""
created on Mar 19, 2014

@author: Nikola Jajcay
"""


from src import wavelet_analysis
from src.data_class import load_NCEP_data_daily
import numpy as np
from datetime import datetime, date
from multiprocessing import Pool



##--- load daily NCEP data ---##
g = load_NCEP_data_daily('../../climate/data/SATdaily/air.sig995.%d.nc', 'air', date(1948,1,1), date(2013,1,1), 
                         None, None, None, False)


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
# if job run in parallel now close the pool
if WORKERS != 0:
    pool.close()
        
print("[%s] Wavelet analysis done. Shape of the phases is %s" % (str(datetime.now()), str(phase.shape)))

#==============================================================================
# david - tvoj turn, mas fazy pre kazdy grid point...
#==============================================================================


