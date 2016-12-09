from scale_network import ScaleSpecificNetwork
from datetime import date
import numpy as np
import cPickle
from src.surrogates import get_p_vals


# net = ScaleSpecificNetwork('/Users/nikola/work-ui/data/NCEP/air.mon.mean.levels.nc', 'air', 
                           # date(1949,1,1), date(2015,1,1), None, None, 0, sampling = 'monthly', anom = False)

# net = ScaleSpecificNetwork('/Users/nikola/work-ui/data/ERA/ERAconcat.t2m.mon.means.1958-2014.bin', 't2m', 
                       # date(1958,1,1), date(2015,1,1), None, None, None, 'monthly', anom = False, pickled = True)

net = ScaleSpecificNetwork('/Users/nikola/work-ui/data/ECAD.tg.daily.nc', 'tg', date(1950, 1, 1), date(2015,1,1), None, 
        None, None, dataset = 'ECA-reanalysis', anom = False)
net.get_monthly_data()
print net.data.shape
print net.get_date_from_ndx(0), net.get_date_from_ndx(-1)


with open("../scale-nets/bins/ECAD-SAT-annual-phase-fluc-SSA-RC-1000FTsurrs-from-indices.bin", "rb") as f:
    surr_res = cPickle.load(f)

INDICES = ['TNA', 'SOI', 'SCAND', 'PNA', 'PDO', 'EA', 'AMO', 'NAO', 'NINO3.4', 'TPI', 'SAM']
P_VAL = 0.05

data_corrs = surr_res['data']
surr_corrs = surr_res['surrs']

no_sigs = np.zeros_like(data_corrs['TNA'])
msk = np.isnan(data_corrs['TNA'])
no_sigs[msk] = np.nan

for index in INDICES:
    result = data_corrs[index].copy()
    surrs_tmp = np.array([surr_corrs[i][index] for i in range(len(surr_corrs))])
    
    p_vals = get_p_vals(data_corrs[index], surrs_tmp, one_tailed = False)
    msk = np.less_equal(p_vals, P_VAL)

    result[~msk] = np.nan

    no_sigs[msk] += 1

    tit = ("ECA&D annual phase fluctuations x %s correlations \n p-value %.2f" % (index, P_VAL))
    fname = ("../scale-nets/ECAD-SAT-annual-phase-fluc-SSA-RC-%scorrs-sig-from-indices.png" % index)
    net.quick_render(field_to_plot = result, tit = tit, fname = fname, symm = True, whole_world = False)


tit = ("ECA&D number of significant with p-value %.2f" % (P_VAL))
fname = ("../scale-nets/ECAD-SAT-annual-phase-fluc-SSA-RC-number-of-sig-from-indices.png")
net.quick_render(field_to_plot = no_sigs, tit = tit, fname = fname, symm = False, whole_world = False)



