import cPickle
import numpy as np
import marshal


a = np.random.rand(1000, 500, 200)
b = np.random.rand(1000, 500, 200)

pickle = True

if pickle:
    with open('pickle_test.bin', 'wb') as f:
        cPickle.dump({'a' : a, 'b' : b}, f, protocol=cPickle.HIGHEST_PROTOCOL)
else:
    with open('marhsal_test.bin', 'wb') as f:
        marshal.dump({'a' : a, 'b' : b}, f)


