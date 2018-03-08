from __future__ import print_function
from scale_network import ScaleSpecificNetwork
from datetime import date
from pathos.multiprocessing import Pool
import numpy as np
import pyclits as clt
import h5py
import matplotlib.pyplot as plt
plt.style.use('ipython')


def load_nino34_wavelet_phase(start_date, end_date, period, anom = False):
    g = clt.data_loaders.load_enso_index('/Users/nikola/work-ui/data/nino34raw.txt', '3.4', start_date, end_date, anom=anom)
    a = g.get_seasonality()
    sg = clt.surrogates.SurrogateField()
    sg.copy_field(g)
    g.return_seasonality(a[0], a[1], a[2])
    g.wavelet(period, period_unit="y", cut=2)

    return g, g.phase.copy(), sg, a

def load_NAOindex_wavelet_phase(start_date, end_date, period, anom = False):
    raw = np.loadtxt('/Users/nikola/work-ui/data/NAO.station.monthly.1865-2016.txt')
    data = raw[:,1:].reshape((-1))
    g = clt.geofield.DataField(data=np.array(data))
    g.create_time_array(date_from=date(1865,1,1), sampling='m')
    g.location = "NAO"
    g.select_date(start_date, end_date)

    if anom:
        g.anomalise()
    a = g.get_seasonality()
    sg = clt.surrogates.SurrogateField()
    sg.copy_field(g)
    g.return_seasonality(a[0], a[1], a[2])
    g.wavelet(period, period_unit="y", cut=2)

    return g, g.phase.copy(), sg, a

def load_sunspot_number_phase(start_date, end_date, period, anom = False):
    raw = np.loadtxt('/Users/nikola/work-ui/data/sunspot.monthly.1749-2017.txt')
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
    a = g.get_seasonality()
    sg = clt.surrogates.SurrogateField()
    sg.copy_field(g)
    g.return_seasonality(a[0], a[1], a[2])
    g.wavelet(period, period_unit="y", cut=2)

    return g, g.phase.copy(), sg, a

def load_pdo_phase(start_date, end_date, period, anom = False):
    raw = np.loadtxt('/Users/nikola/work-ui/data/PDO.monthly.1900-2015.txt')
    data = raw[:,1:].reshape((-1))

    g = clt.geofield.DataField(data=np.array(data))
    g.create_time_array(date_from=date(1900,1,1), sampling='m')
    g.location = "PDO"
    g.select_date(start_date, end_date)

    if anom:
        g.anomalise()
    a = g.get_seasonality()
    sg = clt.surrogates.SurrogateField()
    sg.copy_field(g)
    g.return_seasonality(a[0], a[1], a[2])
    g.wavelet(period, period_unit="y", cut=2)

    return g, g.phase.copy(), sg, a


def _compute_MI_synch(a):
    ph, i, j, nao, nino, sunspots, pdo = a
    # nao_s = clt.knn_mutual_information(ph, nao, k=128)
    # nino_s = clt.knn_mutual_information(ph, nino, k=128)
    # sunspot_s = clt.knn_mutual_information(ph, sunspots, k=128)
    # pdo_s = clt.knn_mutual_information(ph, pdo, k=128)
    nao_s = clt.mutual_information(ph, nao, algorithm='EQQ2', bins=8)
    nino_s = clt.mutual_information(ph, nino, algorithm='EQQ2', bins=8)
    sunspot_s = clt.mutual_information(ph, sunspots, algorithm='EQQ2', bins=8)
    pdo_s = clt.mutual_information(ph, pdo, algorithm='EQQ2', bins=8)

    return (i, j, nao_s, nino_s, sunspot_s, pdo_s)


WORKERS = 5
NUM_SURRS = 100
to_do_periods = np.arange(2,15.5,0.5)
net = ScaleSpecificNetwork('/Users/nikola/work-ui/data/NCEP/air.mon.mean.levels.nc', 
                                    'air', date(1950,1,1), date(2014,1,1), None, None, 
                                    level = 0, dataset="NCEP", sampling='monthly', anom=False)

synchronization = {}
for period in to_do_periods:
    print("running for %.1f period..." % (period))
    _, nao_ph, sg_nao, a_nao = load_NAOindex_wavelet_phase(date(1950,1,1), date(2014,1,1), period, anom=False)
    _, nino_ph, sg_nino, a_nino = load_nino34_wavelet_phase(date(1950,1,1), date(2014,1,1), period, anom=False)
    _, sunspots_ph, sg_sunspots, a_sunspots = load_sunspot_number_phase(date(1950,1,1), date(2014,1,1), period, anom=False)
    _, pdo_ph, sg_pdo, a_pdo = load_pdo_phase(date(1950,1,1), date(2014,1,1), period, anom=False)
    pool = Pool(WORKERS)
    net.wavelet(period, period_unit='y', cut=2, pool=pool)
    args = [(net.phase[:, i, j], i, j, nao_ph, nino_ph, sunspots_ph, pdo_ph) for i in range(net.lats.shape[0]) for j in range(net.lons.shape[0])]
    result = pool.map(_compute_MI_synch, args)
    synchs = np.zeros((4, net.lats.shape[0], net.lons.shape[0]))
    synchs_surrs = np.zeros((NUM_SURRS, 4, net.lats.shape[0], net.lons.shape[0]))
    for i, j, naos, ninos, suns, pdos in result:
        synchs[0, i, j] = naos
        synchs[1, i, j] = ninos
        synchs[2, i, j] = suns
        synchs[3, i, j] = pdos
    for surr in range(NUM_SURRS):
        sg_nao.construct_fourier_surrogates(algorithm='FT')
        sg_nao.add_seasonality(a_nao[0], a_nao[1], a_nao[2])
        sg_nao.wavelet(period, period_unit="y", cut=2)
        sg_nino.construct_fourier_surrogates(algorithm='FT')
        sg_nino.add_seasonality(a_nino[0], a_nino[1], a_nino[2])
        sg_nino.wavelet(period, period_unit="y", cut=2)
        sg_sunspots.construct_fourier_surrogates(algorithm='FT')
        sg_sunspots.add_seasonality(a_sunspots[0], a_sunspots[1], a_sunspots[2])
        sg_sunspots.wavelet(period, period_unit="y", cut=2)
        sg_pdo.construct_fourier_surrogates(algorithm='FT')
        sg_pdo.add_seasonality(a_pdo[0], a_pdo[1], a_pdo[2])
        sg_pdo.wavelet(period, period_unit="y", cut=2)
        args = [(net.phase[:, i, j], i, j, sg_nao.phase, sg_nino.phase, sg_sunspots.phase, sg_pdo.phase) for i in range(net.lats.shape[0]) for j in range(net.lons.shape[0])]
        result = pool.map(_compute_MI_synch, args)
        for i, j, naos, ninos, suns, pdos in result:
            synchs_surrs[surr, 0, i, j] = naos
            synchs_surrs[surr, 1, i, j] = ninos
            synchs_surrs[surr, 2, i, j] = suns
            synchs_surrs[surr, 3, i, j] = pdos
        if (surr%20 == 0):
            print("...%d/%d surrs done..." % (surr, NUM_SURRS))

    pool.close()
    pool.join()
    synchronization[period] = synchs
    synchronization["%.1f_surrs" % (period)] = synchs_surrs

hf = h5py.File('networks/phase_synch_eqq_bins=8_all_periods_%dFTsurrs.h5' % (NUM_SURRS))
for k in synchronization:
    hf.create_dataset(str(k), data=synchronization[k])
hf.close()


## spectra
# nino, _ = load_nino34_wavelet_phase(date(1950,1,1), date(2014,1,1), 4, anom=False)
# suns, _ = load_sunspot_number_phase(date(1950,1,1), date(2014,1,1), 4, anom=False)
# pdo, _ = load_pdo_phase(date(1950,1,1), date(2014,1,1), 4, anom=False)

# scales = np.arange(2, 15*12+1)
# wvlt_power = np.zeros((4,scales.shape[0]))
# for sc in range(scales.shape[0]):
#     nao.wavelet(scales[sc], period_unit='m', cut=24, save_wave=True)
#     wvlt_power[0,sc] = np.sum(np.power(np.abs(nao.wave), 2)) / float(nao.wave.shape[0])

#     nino.wavelet(scales[sc], period_unit='m', cut=24, save_wave=True)
#     wvlt_power[1,sc] = np.sum(np.power(np.abs(nino.wave), 2)) / float(nino.wave.shape[0])

#     suns.wavelet(scales[sc], period_unit='m', cut=24, save_wave=True)
#     wvlt_power[2,sc] = np.sum(np.power(np.abs(suns.wave), 2)) / float(suns.wave.shape[0])

#     pdo.wavelet(scales[sc], period_unit='m', cut=24, save_wave=True)
#     wvlt_power[3,sc] = np.sum(np.power(np.abs(pdo.wave), 2)) / float(pdo.wave.shape[0])

# tits = ["NAO", "NINO3.4", "sunspot #", "PDO"]
# for j, tit in zip(range(4), tits):
#     plt.subplot(2,2,j+1)
#     plt.plot(scales, wvlt_power[j, :])
#     plt.title(tit)
#     plt.xticks(np.arange(10, scales.shape[0], 12), [i/12 for i in scales[10::12]], rotation = 30)
#     # print(j, tit)
#     if j > 1:
#         plt.xlabel("scale [year]")
# plt.show()