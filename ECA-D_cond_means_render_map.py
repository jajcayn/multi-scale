"""
created on June 13, 2014

@author: Nikola Jajcay
"""

import cPickle
from datetime import datetime, date



SURR_TYPE = 'MF' # MF, FT or AR
START_DATE = date(1960,1,1)
MEANS = True
ANOMALISE = True



# load data 
print("[%s] Loading data..." % (str(datetime.now())))
fname = ('result/ECA-D_%s_cond_%s_data_from_%s_16k.bin' % ('SATA' if ANOMALISE else 'SAT', 
         'means' if MEANS else 'std', str(START_DATE)))
with open(fname, 'rb') as f:
    data = cPickle.load(f)
difference_data = data['difference_data']
mean_data = data['mean_data']
del data

# load surrogates
print("[%s] Data loaded. Now loading surrogates..." % (str(datetime.now())))
fname = ('result/ECA-D_%s_cond_%s_%ssurrogates_from_%s_16k.bin' % ('SATA' if ANOMALISE else 'SAT', 
             'means' if MEANS else 'std', SURR_TYPE, str(START_DATE)))
with open(fname, 'rb') as f:
    data = cPickle.load(f) 
difference_surrogates = data['difference_surrogates']
mean_surrogates = data['mean surrogates']
del data
print("[%s] Surrogates loaded." % (str(datetime.now())))

