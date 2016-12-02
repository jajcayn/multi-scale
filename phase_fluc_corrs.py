from scale_network import ScaleSpecificNetwork
from datetime import date
from pathos.multiprocessing import Pool
import numpy as np
from src.data_class import DataField, load_enso_index
from src.surrogates import SurrogateField
from scipy.stats import pearsonr
import cPickle


INDICES = ['TNA', 'SOI', 'SCAND', 'PNA', 'PDO', 'EA', 'AMO', 'NAO', 'NINO3.4']
START_DATES = [date(1948, 1, 1), date(1866, 1, 1), date(1950, 1, 1), date(1950, 1, 1), date(1900, 1, 1), 
                date(1950, 1, 1), date(1948, 1, 1), date(1950, 1, 1), date(1870, 1, 1)]
END_YEARS = [2016, 2014, 2016, 2016, 2015, 2015, 2016, 2016, 2015]
DATE_TYPE = [1, 1, 0, 0, 1, 0, 1, 0, 2]
NUM_SURR = 1000
NUM_WORKERS = 20
path_to_data = "/Users/nikola/work-ui/data/"
path_to_data = "/home/nikola/Work/phd/data/"


def get_corrs(net, ndx):
    if ndx.data.shape[0] != net.phase.shape[0]:
        raise Exception("WRONG SHAPES!")

    corrs = np.zeros_like(net.data[0, ...])
    for lat in range(corrs.shape[0]):
        for lon in range(corrs.shape[1]):
            corrs[lat, lon] = pearsonr(net.phase[:, lat, lon], ndx.data)[0]

    return corrs



net = ScaleSpecificNetwork('%sair.mon.mean.levels.nc' % path_to_data, 'air', 
                           date(1949,1,1), date(2015,1,1), None, None, 0, 'monthly', anom = False)

net_surrs = ScaleSpecificNetwork('%sair.mon.mean.levels.nc' % path_to_data, 'air', 
                           date(1949,1,1), date(2015,1,1), None, None, 0, 'monthly', anom = False)


surr_field = SurrogateField()
a = net.get_seasonality(detrend = True)
surr_field.copy_field(net)
net.return_seasonality(a[0], a[1], a[2])


pool = Pool(NUM_WORKERS)
net.wavelet(1, 'y', pool = pool, cut = 1)
net.get_continuous_phase(pool = pool)
print "wavelet done"
net.get_phase_fluctuations(rewrite = True, pool = pool)
print "fluctuations done"
pool.close()
pool.join()

index_correlations = {}
index_datas = {}

# SURROGATES
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

    index_data.select_date(date(1950, 1, 1), date(2014, 1, 1))
    index_data.anomalise()

    index_datas[index] = index_data
    index_correlations[index] = get_corrs(net, index_datas[index])

    ## plotting
    # tit = ("NCEP annual phase fluctuations x %s correlations" % index)
    # fname = ("../scale-nets/NCEP-SAT-annual-phase-fluc-%scorrs.png" % index)
    # net.quick_render(field_to_plot = index_correlations[index], tit = tit, fname = fname)


def _corrs_surrs(args):
    index_correlations_surrs = {}
    surr_field.construct_fourier_surrogates()
    surr_field.add_seasonality(a[0], a[1], a[2])

    net_surrs.data = surr_field.get_surr()
    net_surrs.wavelet(1, 'y', cut = 1)
    net_surrs.get_continuous_phase()
    net_surrs.get_phase_fluctuations(rewrite = True)
    for index in INDICES:
        index_correlations_surrs[index] = get_corrs(net_surrs, index_datas[index])

    return index_correlations_surrs



pool = Pool(NUM_WORKERS)
args = [1 for i in range(NUM_SURR)]
results = pool.map(_corrs_surrs, args)
pool.close()
pool.join()

with open("NCEP-SAT-annual-phase-fluc-%dFTsurrs.bin" % NUM_SURR, "wb") as f:
    cPickle.dump({'data': index_correlations, 'surrs' : results})


