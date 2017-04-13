import numpy as np
from src.data_class import load_station_data
from datetime import date
import matplotlib.pyplot as plt
from src.ssa import ssa_class

prg = load_station_data('../data/ECAstation-TG/TG_STAID000027.txt', date(1775, 1, 1), date(2016, 5, 1), 
    anom = False, offset = 1)
# prg.get_monthly_data()

# prg.temporal_filter(cutoff = 24, btype = 'lowpass', ftype = 'bessel', cut_time = True)
# prg.data = prg.filtered_data.copy()

# prg_ssa = ssa_class(prg.filtered_data, M = 365*2, compute_rc = False)

# lam, e, pc = prg_ssa.run_ssa()
# print lam
# print e.shape

# plt.plot(e[:, 0])
# plt.plot(e[:, 1])
# plt.plot(e[:, 2])
# plt.show()

# # prg_ssa.run_Monte_Carlo(n_realizations = 1000, p_value = 0.05)
# print prg.filtered_data.shape

# prg.plot_FFT_spectrum(ts = [prg.filtered_data, prg.data[365:-365]], log = True)
# prg.plot_FFT_spectrum(ts = prg.filtered_data, log = True)

