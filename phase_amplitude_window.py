"""
created on May 6, 2014

@author: Nikola Jajcay
"""

from src.oscillatory_time_series import OscillatoryTimeSeries
from datetime import date
import matplotlib.pyplot as plt
import numpy as np


ts = OscillatoryTimeSeries('TG_STAID000027.txt', date(1924,1,1), date(2014,1,1), True)
idx1 = ts.g.find_date_ndx(date(1958,1,1))
idx2 = ts.g.find_date_ndx(date(2002,11,10))
ts.wavelet(8, PAD = True)
ts.phase = ts.phase[idx1 : idx2]
ts.amplitude = ts.amplitude[idx1:idx2]
ts.plot_phase_amplitude()


ts2 = OscillatoryTimeSeries('TG_STAID000027.txt', date(1958,1,1), date(2002,11,10), True)
ts2.wavelet(8, PAD = False)
ts2.plot_phase_amplitude()
