from scale_network import ScaleSpecificNetwork
from datetime import date
from pathos.multiprocessing import Pool
import numpy as np
from src.data_class import DataField
from src.surrogates import get_single_FT_surrogate
from scipy.stats import pearsonr
import cPickle


# path_to_data = "/Users/nikola/work-ui/data/"
path_to_data = "/home/nikola/Work/phd/data/"

NUM_SURR = 1000
NUM_WORKERS = 20




net = ScaleSpecificNetwork('%sair.mon.mean.levels.nc' % path_to_data, 'air', 
                            date(1948,1,1), date(2016,1,1), None, None, 0, dataset = "NCEP", sampling = 'monthly', anom = False)

pool = Pool(NUM_WORKERS)
net.wavelet(1, 'y', pool = pool, cut = 1)
net.get_continuous_phase(pool = pool)
net.get_phase_fluctuations(rewrite = True, pool = pool)
pool.close()
pool.join()

nao = DataField()
raw = np.loadtxt("%sNAO.station.monthly.1865-2016.txt" % (path_to_data))
raw = raw[:, 1:]
nao.data = raw.reshape(-1)
nao.create_time_array(date_from = date(1865, 1, 1), sampling = 'm')
nao.select_date(date(1949, 1, 1), date(2015, 1, 1))
nao.anomalise()
jfm_index = nao.select_months([1,2,3], apply_to_data = False)

jfm_nao = nao.data[jfm_index]
_, _, y = nao.extract_day_month_year()
y = y[jfm_index]
ann_nao = []
for year in np.unique(y):
    ann_nao.append(np.mean(jfm_nao[np.where(year == y)[0]]))
    
ann_nao = np.array(ann_nao)

ann_phase_fluc = np.zeros([ann_nao.shape[0]] + list(net.get_spatial_dims()))
for lat in range(net.lats.shape[0]):
    for lon in range(net.lons.shape[0]):
        jfm_data = net.phase[jfm_index, lat, lon]
        for i, year in zip(range(np.unique(y).shape[0]), np.unique(y)):
            ann_phase_fluc[i, lat, lon] = np.mean(jfm_data[np.where(year == y)[0]])

corrs = np.zeros_like(net.data[0, ...])
for lat in range(net.lats.shape[0]):
    for lon in range(net.lons.shape[0]):
        corrs[lat, lon] = pearsonr(ann_wemo, ann_phase_fluc[:, lat, lon])[0]


def _corrs_surrs_ind(args):
    nao_surr = nao.copy()
    nao_surr.data = get_single_FT_surrogate(nao.data)
    jfm_nao_surr = nao_surr.data[jfm_index]
    ann_nao_surr = []
    for year in np.unique(y):
        ann_nao_surr.append(np.mean(jfm_nao_surr[np.where(year == y)[0]]))
    ann_nao_surr = np.array(ann_nao_surr)

    corrs_surr = np.zeros_like(net.data[0, ...])
    for lat in range(net.lats.shape[0]):
        for lon in range(net.lons.shape[0]):
            corrs_surr[lat, lon] = pearsonr(ann_nao_surr, ann_phase_fluc[:, lat, lon])[0]

    return corrs_surr


pool = Pool(NUM_WORKERS)
args = [1 for i in range(NUM_SURR)]
results = pool.map(_corrs_surrs_ind, args)
pool.close()
pool.join()

with open("NAO-JFM_ann_means-NCEP-phase-fluc-%dFTsurrs-from-indices.bin" % NUM_SURR, "wb") as f:
    cPickle.dump({'data': corrs, 'surrs' : results}, f, protocol = cPickle.HIGHEST_PROTOCOL)