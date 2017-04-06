import numpy as np
from src.data_class import load_station_data
from datetime import date
import matplotlib.pyplot as plt
from src.ssa import ssa_class

prg = load_station_data('../data/ECAstation-TG/TG_STAID000027.txt', date(1775, 1, 1), date(2016, 5, 1), 
    anom = False, offset = 1)
prg.get_monthly_data()

prg.plot_welch_spectrum(log = True)

# prg_ssa = ssa_class(prg.data, M = 12, compute_rc = False)

# lam, e, pc = prg_ssa.run_ssa()
# print lam

# prg_ssa.run_Monte_Carlo(n_realizations = 100, p_value = 0.05)

