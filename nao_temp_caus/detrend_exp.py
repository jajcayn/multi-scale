import pyclits as clt
import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
plt.style.use('ipython')

prg_raw = np.loadtxt("nao_temp_caus/PRG_from_ECAD_tg_pp.txt")
prg_tg = clt.geofield.DataField()
prg_tg.data = prg_raw[:, 3].copy()
prg_tg.create_time_array(date_from=date(1950,1,1), sampling='d')
prg_tg.select_date(date(1990,1,1), date(2017,1,1))


#
a, b, c = prg_tg.get_seasonality(detrend=True)
tg = prg_tg.data.copy()
plt.plot(prg_tg.data, label="detrend last", linewidth=0.9)
prg_tg.return_seasonality(a,b,c)

prg_tg.data = ss.detrend(prg_tg.data, type='linear')
prg_tg.get_seasonality(detrend=False)
plt.plot(prg_tg.data, label="detrend first", linewidth=0.9)

plt.twinx()
plt.plot(tg - prg_tg.data, label="difference", color = 'k', linewidth=1.5)
plt.ylim([-0.5,0.5])

plt.legend()
print np.allclose(prg_tg.data, tg)
plt.title("PRG TG deseasonalised")
plt.show()