"""
created on May 6, 2014

@author: Nikola Jajcay
"""

from src.oscillatory_time_series import OscillatoryTimeSeries
from datetime import date

start_dates = [date(1834,7,28), date(1958,1,1)]
end_dates = [date(2014,1,1), date(2002,11,10)]

ts = OscillatoryTimeSeries('TG_STAID000027.txt', start_dates[0], end_dates[0])
ts.wavelet(8)
ts.get_conditional_means(8, False)


sur = ts.get_conditional_means_surrogates(20)
ts.plot_conditional_means(None, False)