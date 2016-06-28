from src.data_class import load_ERSST_data
from src.empirical_model import EmpiricalModel
from datetime import date

sst = load_ERSST_data('/Users/nikola/work-ui/data/ersstv4/', date(1900,1,1), date(2011,1,1), [-60, 60])
em = EmpiricalModel(3, True)
em.copy_existing_datafield(sst)
em.remove_low_freq_variability(mean_over = 50, cos_weights = True, no_comps = 5)
em.prepare_input(anom = True, no_input_ts = 20, cos_weights = True, sel = None)
em.train_model(harmonic_pred = 'none', quad = False, delay_model = True)

em.integrate_model(1000, int_length = em.data.shape[0]+1000, sigma = 1., n_workers = 7, diagnostics = True, noise_type = 'white')
em.reconstruct_simulated_field(lats = [-5, 5], lons = [190, 240], mean = True)

print em.reconstructions.shape
import scipy.io as sio
sio.savemat("Nino34-delay-linear-20PC-L3-seasonal-d:5mon-k:50-surrs.mat", {'N34s' : em.reconstructions})