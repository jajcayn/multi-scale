from scale_network import ScaleSpecificNetwork
from datetime import date
from multiprocessing import Pool
import numpy as np

WORKERS = 10
PERIOD = 4

def load_nino34_wavelet_phase(start_date, end_date, anom = True):
    from src.data_class import DataField
    import src.wavelet_analysis as wvlt
    from datetime import date
    raw = np.loadtxt('/home/nikola/Work/phd/data/nino34monthly.txt')
    data = []
    time = []
    for y in range(raw.shape[0]):
        for m in range(1,13):
            dat = float(raw[y, m])
            data.append(dat)
            time.append(date(int(raw[y, 0]), m, 1).toordinal())

    g = DataField(data = np.array(data), time = np.array(time))
    g.location = "NINO3.4"
    g.select_date(start_date, end_date)

    if anom:
        g.anomalise()

    k0 = 6. # wavenumber of Morlet wavelet used in analysis
    fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + np.power(k0,2)))
    per = PERIOD * 12 # frequency of interest
    s0 = per / fourier_factor # get scale

    wave, _, _, _ = wvlt.continous_wavelet(g.data, 1, False, wvlt.morlet, dj = 0, s0 = s0, j1 = 0, k0 = 6.)
    phase = np.arctan2(np.imag(wave), np.real(wave))[0, :]

    return phase




nino34_phase = load_nino34_wavelet_phase(date(1948,1,1), date(2014,1,1), True)
net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 'air', 
                       date(1948,1,1), date(2014,1,1), None, None, 0, 'monthly', anom = False)
pool = Pool(WORKERS)             
net.wavelet(PERIOD, get_amplitude = False, pool = pool)
print "wavelet on data done"
pool.close()
net.get_adjacency_matrix_conditioned(nino34_phase, use_queue = True, num_workers = WORKERS)
print "estimating adjacency matrix done"
net.save_net('networks/NCEP-SAT4y-phase-adjmatCMIEQQcondNINOphase.bin', only_matrix = True)