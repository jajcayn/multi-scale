import numpy as np
import marshal
import hickle as hkl


a = np.random.rand(1000, 500, 200).astype(np.float64)
b = np.random.rand(1000, 500, 200).astype(np.float64)

load = True

if not load:
    hkl.dump({'a' : a, 'b' : b}, 'test_hickle.hkl', mode = 'w')
else:
    data = hkl.load('test_hickle.hkl')

print data['a'].shape, data['b'].shape


