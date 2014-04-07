import numpy as np
#import pywt
#import matplotlib.pyplot as plt
from datetime import date
from scipy.signal import detrend


from src.data_class import load_station_data


start_date = date(1924, 4, 15)
end_date = date(2014, 1, 1) # exclusive
g = load_station_data('../data/Spain/TG_STAID000230.txt', start_date, end_date, True)
        

    
def nandetrend(arr, axis = 0):
    """
    Removes the linear trend along the axis, ignoring Nans.
    """
    a = arr.copy()
    rnk = len(a.shape)
    if axis < 0:
        axis += rnk # axis -1 means along last dimension
    newdims = np.r_[axis, 0:axis, axis + 1:rnk]
    newdata = np.reshape(np.transpose(a, tuple(newdims)), (a.shape[axis], np.prod(a.shape, axis = 0) // a.shape[axis]))
    newdata = newdata.copy()
    x = np.arange(0, a.shape[axis], 1)
    print 'x', x.shape
    print 'data', newdata.shape
    A = np.vstack([x, np.ones(len(x))]).T
    print A.shape
    m, c = np.linalg.lstsq(A, newdata)[0]
    for i in range(a.shape[axis]):
        newdata[i, ...] = newdata[i, ...] - (m*x[i] + c)
    tdshape = np.take(a.shape, newdims, 0)
    ret = np.reshape(newdata, tuple(tdshape))
    vals = list(range(1,rnk))
    olddims = vals[:axis] + [0] + vals[axis:]
    ret = np.transpose(ret, tuple(olddims))
    
    return ret, m
    
    
a = np.random.rand(1024, 2563, 3)

d1 = detrend(a, axis = 0)
d2 = detrend(a, axis = 1)