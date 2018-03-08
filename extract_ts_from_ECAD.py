#%%
import pyclits as clt
import numpy as np
from datetime import date

#%%
tg = clt.geofield.DataField()
tg.load(filename='../data/ECAD.tg.daily.0.50deg.nc', variable_name='tg')
pp = clt.geofield.DataField()
pp.load(filename='../data/ECAD.pp.daily.0.50deg.nc', variable_name='pp')

#%%
print tg.shape()
print pp.shape()
# PRG
la, lo = tg.get_closest_lat_lon(52.23, 21.01)
prg_tg = tg.data[:, la, lo].copy()
prg_pp = pp.data[:, la, lo].copy()
d, m, y = tg.extract_day_month_year()
print prg_tg.shape, prg_pp.shape, d.shape, m.shape
to_write = np.vstack([y, m, d, prg_tg, prg_pp])

#%%
print to_write.shape
np.savetxt("WARSAW_from_ECAD_tg_pp.txt", to_write.T, fmt="%.4f")