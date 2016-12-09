from scale_network import ScaleSpecificNetwork
from datetime import date
from pathos.multiprocessing import Pool
import numpy as np
from src.data_class import DataField, load_enso_index
from src.surrogates import SurrogateField, get_single_FT_surrogate, get_p_vals
import src.mutual_information as MI
from scipy.stats import pearsonr
from scipy.signal import hilbert, welch
from src.ssa import ssa_class
import cPickle
import matplotlib.pyplot as plt


# path_to_data = "/Users/nikola/work-ui/data"
path_to_data = "/home/nikola/Work/phd/data"


def _get_MI(args):

    i, j, ph1, ph2 = args

    # return i, j, MI.mutual_information(ph1, ph2, 'EQQ2', bins = 16)
    return i, j, MI.knn_mutual_information(ph1, ph2, k = 32, dualtree = True)


net = ScaleSpecificNetwork('%s/air.mon.mean.levels.nc' % path_to_data, 'air', 
                           date(1950,1,1), date(2015,1,1), None, None, 0, dataset = "NCEP", sampling = 'monthly', anom = True)


nao = DataField()
raw = np.loadtxt("%s/NAO.monthly.1950-2016.txt" % (path_to_data))
nao.data = raw[:, 2]

nao.create_time_array(date_from = date(1950, 1, 1), sampling = 'm')
nao.select_date(date(1950, 1, 1), date(2015, 1, 1))
nao.anomalise()

nao.wavelet(8, 'y', cut = 1)
pool = Pool(5)
net.wavelet(8, 'y', pool = pool, cut = 1)

print nao.phase.shape, net.phase.shape

nao_synch = np.zeros(net.get_spatial_dims())
args = [ (i, j, net.phase[:, i, j], nao.phase) for i in range(net.lats.shape[0]) for j in range(net.lons.shape[0])]
results = pool.map(_get_MI, args)
for i, j, res in results:
    nao_synch[i, j] = res

# fname = "NCEP-SAT-NAO-8yr-phase-MIKNN-k=32.png"
# net.quick_render(field_to_plot = nao_synch, tit = "8yr phase synch TEMP x NAO", symm = False, fname = None)
NUM_SURR = 100
P_VAL = 0.05


nao_surrs = np.zeros([NUM_SURR] + net.get_spatial_dims())
for surr_num in range(NUM_SURR):
    print surr_num
    surr = nao.copy()
    surr.data = get_single_FT_surrogate(nao.data)
    surr.wavelet(8, 'y', cut = 1)
    args = [ (i, j, net.phase[:, i, j], surr.phase) for i in range(net.lats.shape[0]) for j in range(net.lons.shape[0])]
    results = pool.map(_get_MI, args)
    for i, j, res in results:
        nao_surrs[surr_num, i, j] = res

pool.close()
pool.join()

with open("NCEP-SAT-NAO-8yr-phase-%dFTsurrs-MIEQQ-k=32.bin" % NUM_SURR, "wb") as f:
    cPickle.dump({'data' : nao_synch, 'surrs' : nao_surrs}, f, protocol = cPickle.HIGHEST_PROTOCOL)
    # raw = cPickle.load(f)

# nao_synch = raw['data']
# nao_surrs = raw['surrs']

# result = nao_synch.copy()
# p_vals = get_p_vals(nao_synch, nao_surrs, one_tailed = True)
# msk = np.less_equal(p_vals, P_VAL)
# result[~msk] = np.nan

# fname = "NCEP-SAT-NAO-8yr-phase-%dFTsurrs-MIEQQ-k=32.png" % NUM_SURR
# net.quick_render(field_to_plot = result, tit = "8yr phase synch TEMP x NAO \n p-value %.2f" % P_VAL, symm = False, fname = fname)