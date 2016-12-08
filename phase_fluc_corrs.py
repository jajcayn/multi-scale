from scale_network import ScaleSpecificNetwork
from datetime import date
from pathos.multiprocessing import Pool
import numpy as np
from src.data_class import DataField, load_enso_index
from src.surrogates import SurrogateField, get_single_FT_surrogate
from scipy.stats import pearsonr
from scipy.signal import hilbert
from src.ssa import ssa_class
import cPickle


INDICES = ['TNA', 'SOI', 'SCAND', 'PNA', 'PDO', 'EA', 'AMO', 'NAO', 'TPI', 'SAM', 'NINO3.4']
# INDICES = ['TPI', 'SAM']
START_DATES = [date(1948, 1, 1), date(1866, 1, 1), date(1950, 1, 1), date(1950, 1, 1), date(1900, 1, 1), 
                date(1950, 1, 1), date(1948, 1, 1), date(1950, 1, 1), date(1895, 1, 1), date(1957, 1, 1), date(1870, 1, 1)]
# START_DATES = [date(1895, 1, 1), date(1957, 1, 1)]
END_YEARS = [2016, 2014, 2016, 2016, 2015, 2015, 2016, 2016, 2014, 2016, 2015]
# END_YEARS = [2014, 2016]
DATE_TYPE = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 2]
# DATE_TYPE = [1, 1]
NUM_SURR = 1000
NUM_WORKERS = 20
# path_to_data = "/Users/nikola/work-ui/data/"
path_to_data = "/home/nikola/Work/phd/data/"


def get_corrs(net, ndx, cut_ndx = None):
    if cut_ndx is None:
        if ndx.data.shape[0] != net.phase.shape[0]:
            raise Exception("WRONG SHAPES!")
    else:
        if ndx.data.shape[0] != np.sum(cut_ndx):
            raise Exception("WRONG SHAPES!")

    corrs = np.zeros_like(net.data[0, ...])
    for lat in range(corrs.shape[0]):
        for lon in range(corrs.shape[1]):
            if cut_ndx is None:
                corrs[lat, lon] = pearsonr(net.phase[:, lat, lon], ndx.data)[0]
            else:
                corrs[lat, lon] = pearsonr(net.phase[cut_ndx, lat, lon], ndx.data)[0]

    return corrs



net = ScaleSpecificNetwork('%sair.mon.mean.levels.nc' % path_to_data, 'air', 
                           date(1949,1,1), date(2015,1,1), None, None, 0, dataset = "NCEP", sampling = 'monthly', anom = False)

# net = ScaleSpecificNetwork('%sERA/ERAconcat.t2m.mon.means.1958-2014.bin' % path_to_data, 't2m', 
                       # date(1958,1,1), date(2015,1,1), None, None, None, 'monthly', anom = False, pickled = True)

# net_surrs = ScaleSpecificNetwork('%sERAconcat.t2m.mon.means.1958-2014.bin' % path_to_data, 't2m', 
#                        date(1958,1,1), date(2015,1,1), None, None, None, 'monthly', anom = False, pickled = True)

# net = ScaleSpecificNetwork('%s20CR/20CR.t2m.mon.nc' % path_to_data, 't2m', 
                           # date(1949,1,1), date(2011,1,1), None, None, dataset = "ERA", sampling = 'monthly', anom = False)

# net_surrs = ScaleSpecificNetwork('%sair.mon.mean.levels.nc' % path_to_data, 'air', 
                           # date(1949,1,1), date(2015,1,1), None, None, 0, dataset = "NCEP", sampling = 'monthly', anom = False)

# net = ScaleSpecificNetwork('%sECAD.tg.daily.nc' % path_to_data, 'tg', date(1950, 1, 1), date(2015,1,1), None, 
#     None, None, dataset = 'ECA-reanalysis', anom = False)
# net.get_monthly_data()
# print net.data.shape
# print net.get_date_from_ndx(0), net.get_date_from_ndx(-1)


# surr_field = SurrogateField()
# a = net.get_seasonality(detrend = True)
# surr_field.copy_field(net)
# net.return_seasonality(a[0], a[1], a[2])

def _hilbert_ssa(args):
    i, j, data = args
    ssa = ssa_class(data, M = 12)
    _, _, _, rc = ssa.run_ssa()
    real_part = rc[:, 0] + rc[:, 1]
    imag_part = np.imag(hilbert(real_part))
    phase_hilb_rc = np.arctan2(imag_part[12:-12], real_part[12:-12])
        
    return i, j, phase_hilb_rc


pool = Pool(NUM_WORKERS)
net.wavelet(1, 'y', pool = pool, cut = 1)

# Hilbert on RC SSA fluctuations
args = [ (i, j, net.data[:, i, j]) for i in range(net.lats.shape[0]) for j in range(net.lons.shape[0]) ]
results = pool.map(_hilbert_ssa, args)
for i, j, res in results:
    net.phase[:, i, j] = res

net.get_continuous_phase(pool = pool)
net.get_phase_fluctuations(rewrite = True, pool = pool)
pool.close()
pool.join()

index_correlations = {}
index_datas = {}

# # SURROGATES
for index, ndx_type, start_date, end_year in zip(INDICES, DATE_TYPE, START_DATES, END_YEARS):
    # load index
    print index

    if index != 'NINO3.4':
        index_data = DataField()
        raw = np.loadtxt("%s%s.monthly.%d-%d.txt" % (path_to_data, index, start_date.year, end_year))
        if ndx_type == 0:
            index_data.data = raw[:, 2]
        elif ndx_type == 1:
            raw = raw[:, 1:]
            index_data.data = raw.reshape(-1)

        index_data.create_time_array(date_from = start_date, sampling = 'm')
    
    elif index == 'NINO3.4':
        index_data = load_enso_index("%snino34raw.txt" % path_to_data, '3.4', date(1950, 1, 1), date(2014, 1, 1), anom = True)

    if index == 'SAM':
        index_data.select_date(date(1957, 1, 1), date(2014, 1, 1))
        index_data.anomalise()
        index_datas[index] = index_data
        ndx_sam = net.select_date(date(1957, 1, 1), date(2014, 1, 1), apply_to_data = False)[12:-12]
        index_correlations[index] = get_corrs(net, index_datas[index], cut_ndx = ndx_sam)
    else:
        index_data.select_date(date(1950, 1, 1), date(2014, 1, 1))
        index_data.anomalise()
        index_datas[index] = index_data
        index_correlations[index] = get_corrs(net, index_datas[index])

    # with open("20CRtemp-phase-fluct-corr-with-%sindex-1950-2014.bin" % index, "wb") as f:
        # cPickle.dump({('%scorrs' % index) : index_correlations[index].reshape(np.prod(index_correlations[index].shape))}, f)

    # plotting
    # tit = ("NCEP annual phase SSA RC fluctuations x %s correlations" % index)
    # fname = ("../scale-nets/NCEP-SAT-annual-phase-fluc-SSA-RC-%scorrs.png" % index)
    # net.quick_render(field_to_plot = index_correlations[index], tit = tit, symm = True, whole_world = True, fname = fname)


# def _corrs_surrs(args):
#     index_correlations_surrs = {}
#     surr_field.construct_fourier_surrogates()
#     surr_field.add_seasonality(a[0], a[1], a[2])

#     net_surrs.data = surr_field.get_surr()
#     net_surrs.wavelet(1, 'y', cut = 1)
#     net_surrs.get_continuous_phase()
#     net_surrs.get_phase_fluctuations(rewrite = True)
#     for index in INDICES:
#         if index == 'SAM':
#             index_correlations_surrs[index] = get_corrs(net_surrs, index_datas[index], cut_ndx = ndx_sam)
#         else:
#             index_correlations_surrs[index] = get_corrs(net_surrs, index_datas[index])
        

#     return index_correlations_surrs



# pool = Pool(NUM_WORKERS)
# args = [1 for i in range(NUM_SURR)]
# results = pool.map(_corrs_surrs, args)
# pool.close()
# pool.join()

# with open("NCEP-SAT-annual-phase-fluc-%dFTsurrs.bin" % NUM_SURR, "wb") as f:
#     cPickle.dump({'data': index_correlations, 'surrs' : results}, f, protocol = cPickle.HIGHEST_PROTOCOL)


def _corrs_surrs_ind(args):
    index_correlations_surrs = {}
    for index in INDICES:
        index_surr = DataField()
        index_surr.data = get_single_FT_surrogate(index_datas[index].data)
        if index == 'SAM':
            index_correlations_surrs[index] = get_corrs(net, index_surr, cut_ndx = ndx_sam)
        else:
            index_correlations_surrs[index] = get_corrs(net, index_surr)

    return index_correlations_surrs



pool = Pool(NUM_WORKERS)
args = [1 for i in range(NUM_SURR)]
results = pool.map(_corrs_surrs_ind, args)
pool.close()
pool.join()

with open("NCEP-SAT-annual-phase-fluc-SSA-RC-%dFTsurrs-from-indices.bin" % NUM_SURR, "wb") as f:
    cPickle.dump({'data': index_correlations, 'surrs' : results}, f, protocol = cPickle.HIGHEST_PROTOCOL)


