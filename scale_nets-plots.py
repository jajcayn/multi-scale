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
with h5py.File("networks/phase_synch_eqq_bins=8_1958-2014.h5") as hf:
    to_do_periods = [8]
    for period in to_do_periods:
        plt.figure(figsize=(15,7))
        synch = hf['period_%dy' % (period)][:]
        for i, tit in zip(range(synch.shape[0]), tits):
            plt.subplot(2,2,i+1)
            vmax, vmin = np.nanmax(synch), np.nanmin(synch)
            net.quick_render(field_to_plot=synch[i, ...], tit=tit, symm=False, whole_world=True, subplot=True,
                                cmap='hot')#, vminmax=[vmin,vmax])

        plt.suptitle("SYNCHRONIZATION SAT vs. X: %dyr phase [eqq 8] 1958-2014" % (period))
        plt.savefig("networks/plots-synch/SATphase_synch%dyr_phase_eqq8_1958-2014.png" % (period), bbox_inches='tight')

