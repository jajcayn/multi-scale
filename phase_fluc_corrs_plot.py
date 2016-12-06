from scale_network import ScaleSpecificNetwork
from datetime import date
import numpy as np
import cPickle
from src.surrogates import get_p_vals


net = ScaleSpecificNetwork('/Users/nikola/work-ui/data/NCEP/air.mon.mean.levels.nc', 'air', 
                           date(1949,1,1), date(2015,1,1), None, None, 0, sampling = 'monthly', anom = False)

# net = ScaleSpecificNetwork('/Users/nikola/work-ui/data/ERA/ERAconcat.t2m.mon.means.1958-2014.bin', 't2m', 
                       # date(1958,1,1), date(2015,1,1), None, None, None, 'monthly', anom = False, pickled = True)

with open("../scale-nets/bins/NCEP-SAT-annual-phase-fluc-1000FTsurrs.bin", "rb") as f:
    surr_res = cPickle.load(f)

INDICES = ['TNA', 'SOI', 'SCAND', 'PNA', 'PDO', 'EA', 'AMO', 'NAO', 'NINO3.4']
P_VAL = 0.05

data_corrs = surr_res['data']
surr_corrs = surr_res['surrs']

for index in INDICES:
    result = data_corrs[index].copy()
    surrs_tmp = np.array([surr_corrs[i][index] for i in range(len(surr_corrs))])
    
    p_vals = get_p_vals(data_corrs[index], surrs_tmp, one_tailed = False)
    msk = np.less_equal(p_vals, P_VAL)

    result[~msk] = np.nan

    tit = ("NCEP annual phase fluctuations x %s correlations \n p-value %.2f" % (index, P_VAL))
    fname = ("../scale-nets/NCEP-SAT-annual-phase-fluc-%scorrs-sig.png" % index)
    net.quick_render(field_to_plot = result, tit = tit, fname = fname, symm = True, whole_world = True)


