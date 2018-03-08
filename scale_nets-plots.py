import h5py
import numpy as np
import pyclits as clt
from scale_network import ScaleSpecificNetwork
import matplotlib.pyplot as plt
from datetime import date
plt.style.use('ipython')

net = ScaleSpecificNetwork('/Users/nikola/work-ui/data/NCEP/air.mon.mean.levels.nc', 
                                    'air', date(1950,1,1), date(2014,1,1), None, None, 
                                    level = 0, dataset="NCEP", sampling='monthly', anom=False)

tits = ["NAO", "NINO3.4", "sunspot #", "PDO"]
with h5py.File("networks/phase_synch_eqq_bins=8_all_periods.h5") as hf:
    to_do_periods = np.arange(2,15.5,0.5)
    for period in to_do_periods:
        plt.figure(figsize=(15,7))
        synch = hf['period_%.1fy' % (period)][:]
        for i, tit in zip(range(synch.shape[0]), tits):
            plt.subplot(2,2,i+1)
            vmax, vmin = np.nanmax(synch), np.nanmin(synch)
            net.quick_render(field_to_plot=synch[i, ...], tit=tit, symm=False, whole_world=True, subplot=True,
                                cmap='hot', vminmax=[vmin,0.8*vmax], extend = "max")

        plt.suptitle("SYNCHRONIZATION SAT vs. X: %.1fyr phase [eqq 8]" % (period))
        plt.savefig("networks/plots-synch/SATphase_synch%.1fyr_phase_eqq8.png" % (period), bbox_inches='tight')

