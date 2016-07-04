from src.data_class import load_ERSST_data
from src.empirical_model import EmpiricalModel
from datetime import date

sst = load_ERSST_data('/Users/nikola/work-ui/data/ersstv4/', date(1900,1,1), date(2011,1,1), [-60, 60])
em = EmpiricalModel(no_levels = 3, verbose = True)
em.copy_existing_datafield(sst)
em.remove_low_freq_variability(mean_over = 50, cos_weights = True, no_comps = 5)
sel_sst = [0, 3, 4, 5, 7, 6, 8, 9, 11, 13]#, 14, 16, 17, 19, 24, 27, 30, 34, 37, 39]
em.prepare_input(anom = True, no_input_ts = len(sel_sst), cos_weights = True, sel = sel_sst)
em.train_model(harmonic_pred = 'first', quad = True, delay_model = True)

em.integrate_model(100, int_length = em.data.shape[0]+1000, sigma = 0.1, n_workers = 5, diagnostics = True, noise_type = 'seasonal')
em.reconstruct_simulated_field(lats = [-5, 5], lons = [190, 240], mean = True)

print em.reconstructions.shape
import scipy.io as sio
sio.savemat("Nino34-delay-quad-10PCsel-L3-seasonal-d:8mon-k:50.mat", {'N34s' : em.reconstructions})