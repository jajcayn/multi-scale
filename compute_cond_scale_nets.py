from __future__ import print_function
from scale_network import ScaleSpecificNetwork
from datetime import date
from pathos.multiprocessing import Pool
import numpy as np
import pyclits as clt

def load_nino34_wavelet_phase(start_date, end_date, period, anom = False):
    raw = np.loadtxt('/home/nikola/Work/phd/data/nino34monthly.txt')
    data = []
    time = []
    for y in range(raw.shape[0]):
        for m in range(1,13):
            dat = float(raw[y, m])
            data.append(dat)
            time.append(date(int(raw[y, 0]), m, 1).toordinal())

    g = clt.geofield.DataField(data=np.array(data), time=np.array(time))
    g.location = "NINO3.4"
    g.select_date(start_date, end_date)

    if anom:
        g.anomalise()

    g.wavelet(period, period_unit="y", cut=2)

    return g.phase.copy()

def load_NAOindex_wavelet_phase(start_date, end_date, period, anom = False):
    raw = np.loadtxt('/home/nikola/Work/phd/data/NAOmonthly.txt')
    data = []
    time = []
    for y in range(raw.shape[0]):
        dat = float(raw[y, 2])
        data.append(dat)
        time.append(date(int(raw[y, 0]), int(raw[y, 1]), 1).toordinal())

    g = clt.geofield.DataField(data=np.array(data), time=np.array(time))
    g.location = "NAO"
    g.select_date(start_date, end_date)

    if anom:
        g.anomalise()

    g.wavelet(period, period_unit="y", cut=2)

    return g.phase.copy()

def load_sunspot_number_phase(start_date, end_date, period, anom = False):
    raw = np.loadtxt('/home/nikola/Work/phd/data/sunspot.monthly.1749-2017.txt')
    data = []
    time = []
    for y in range(raw.shape[0]):
        dat = float(raw[y, 3])
        data.append(dat)
        time.append(date(int(raw[y, 0]), int(raw[y, 1]), 1).toordinal())

    g = clt.geofield.DataField(data=np.array(data), time=np.array(time))
    g.location = "suspots"
    g.select_date(start_date, end_date)

    if anom:
        g.anomalise()

    g.wavelet(period, period_unit="y", cut=2)

    return g.phase.copy()

def load_pdo_phase(start_date, end_date, period, anom = False):
    raw = np.loadtxt('/home/nikola/Work/phd/data/PDO.monthly.1900-2015.txt')
    data = raw[:,1:].reshape((-1))

    g = clt.geofield.DataField(data=np.array(data))
    g.create_time_array(date_from=date(1900,1,1), sampling='m')
    g.location = "PDO"
    g.select_date(start_date, end_date)

    if anom:
        g.anomalise()

    g.wavelet(period, period_unit="y", cut=2)

    return g.phase.copy()


WORKERS = 20
to_do_periods = [4, 6, 8, 11, 15]

for period in to_do_periods:
    print("computing phase conditioned on NAO")
    nao_phase = load_NAOindex_wavelet_phase(date(1950,1,1), date(2014,1,1), period, False)
    net = ScaleSpecificNetwork('/home/nikola/Work/phd/data/air.mon.mean.levels.nc', 
                                    'air', date(1950,1,1), date(2014,1,1), None, None, 
                                    level = 0, dataset="NCEP", sampling='monthly', anom=False)
    pool = Pool(WORKERS)             
    net.wavelet(period, period_unit="y", pool=pool, cut=2)
    print("wavelet on data done")
    pool.close()
    pool.join()
    net.get_adjacency_matrix_conditioned(nao_phase, use_queue=True, num_workers=WORKERS)
    print("estimating adjacency matrix done")
    net.save_net('networks/NCEP-SAT%dy-phase-adjmatCMIEQQcondNAOphase.bin' % (period), only_matrix=True)

    print("computing phase conditioned on NINO")
    nino_phase = load_nino34_wavelet_phase(date(1950,1,1), date(2014,1,1), period, False)
    net.get_adjacency_matrix_conditioned(nino_phase, use_queue=True, num_workers=WORKERS)
    print("estimating adjacency matrix done")
    net.save_net('networks/NCEP-SAT%dy-phase-adjmatCMIEQQcondNINOphase.bin' % (period), only_matrix=True)

    print("computing phase conditioned on sunspots")
    sunspot_phase = load_sunspot_number_phase(date(1950,1,1), date(2014,1,1), period, False)
    net.get_adjacency_matrix_conditioned(sunspot_phase, use_queue=True, num_workers=WORKERS)
    print("estimating adjacency matrix done")
    net.save_net('networks/NCEP-SAT%dy-phase-adjmatCMIEQQcondSUNSPOTphase.bin' % (period), only_matrix=True)

    print("computing phase conditioned on PDO")
    pdo_phase = load_pdo_phase(date(1950,1,1), date(2014,1,1), period, False)
    net.get_adjacency_matrix_conditioned(pdo_phase, use_queue=True, num_workers=WORKERS)
    print("estimating adjacency matrix done")
    net.save_net('networks/NCEP-SAT%dy-phase-adjmatCMIEQQcondPDOphase.bin' % (period), only_matrix=True)



