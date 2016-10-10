from src.data_class import load_ERSST_data
from src.empirical_model import EmpiricalModel
from datetime import date
import numpy as np

sst = load_ERSST_data('/Users/nikola/work-ui/data/ersstv4/', date(1900,1,1), date(2011,1,1), [-60, 60])
em = EmpiricalModel(no_levels = 3, verbose = True)
em.copy_existing_datafield(sst)
# em.remove_low_freq_variability(mean_over = 50, cos_weights = True, no_comps = 5)
# sel_sst = [0, 3, 4, 5, 7, 6, 8, 9, 11, 13]#, 14, 16, 17, 19, 24, 27, 30, 34, 37, 39]
em.prepare_input(anom = True, no_input_ts = 40, cos_weights = True, sel = None)
em.train_model(harmonic_pred = 'first', quad = False, delay_model = False)

em.integrate_model(50, int_length = em.data.shape[0], sigma = 1, n_workers = 5, diagnostics = True, noise_type = 'white')
em.reconstruct_simulated_field(lats = [-5, 5], lons = [190, 240], mean = True)

print em.reconstructions.shape
import scipy.io as sio
sio.savemat("Nino34-PLS.mat", {'N34s' : em.reconstructions})

# def read_rossler(fname):
#     import csv
#     r = {}
#     f = open(fname, 'r')
#     reader = csv.reader(f, lineterminator = "\n")
#     eps = None
#     for row in reader:
#         if 'eps1' in row[0]:
#             if eps is not None:
#                 r[eps] = np.array(r[eps])
#             eps = float(row[0][8:])
#             r[eps] = []
#         if not "#" in row[0]:
#             n = row[0].split()
#             r[eps].append([float(n[0]), float(n[1])])
#     r[eps] = np.array(r[eps])
#     f.close()

#     return r

# fname = "conceptualRossler1:2monthlysampling_100eps0-0.25.dat"
# r = read_rossler(fname)
# # x, y = r[0.202][20000:22048, 0], r[0.202][20000:22048, 1] # x is biennal, y is annual
# # print x.shape, y.shape
# x = r[0.202][20000:22052, :].T
# print x.shape
# em = EmpiricalModel(no_levels = 3, verbose = True)
# em.input_pcs = x.copy()
# em.input_pcs /= np.std(em.input_pcs[0, :], ddof = 1)
# em.train_model(harmonic_pred = 'first', quad = True, delay_model = False)
# em.integrate_model(10, sigma = 0.1, n_workers = 5, diagnostics = False, noise_type = 'seasonal')
# res = em.integration_results
# res *= np.std(x[0, :], ddof = 1)
# print res.shape

# import matplotlib.pyplot as plt

# plt.plot(x[0, :], linewidth = 1.2, color = 'b')
# for i in range(res.shape[0]):
#     plt.plot(res[i, 0, :], linewidth = 0.7, color = 'k')
# plt.show()
# plt.close()

# plt.plot(x[1, :], linewidth = 1.2, color = 'b')
# for i in range(res.shape[0]):
#     plt.plot(res[i, 1, :], linewidth = 0.7, color = 'k')
# plt.show()