import numpy as np
import pywt



n = 14
ts_length = np.power(2, n)

ts = np.random.rand(ts_length)
print 'data: ', ts.shape

coeffs_db1 = pywt.wavedec(ts, 'db1', level = n-1)
#coeffs_db12 = pywt.wavedec(ts, 'db12', level = n-1)



print len(coeffs_db1)
for i in range(len(coeffs_db1)):
    k_coef = np.power(2,i) if i != 0 else 2
    print i, "th coeff's (k <= ", k_coef, ") shape is ", coeffs_db1[i].shape

coeffs_tilde = []
coeffs_tilde.append(coeffs_db1[0])
coeffs_tilde.append(coeffs_db1[1])

for j in range(2,len(coeffs_db1)):
    multip = np.zeros_like(coeffs_db1[j])
    for k in range(coeffs_db1[j-1].shape[0]):
        multip[2*k] = coeffs_db1[j][2*k] / coeffs_db1[j-1][k]
        multip[2*k+1] = coeffs_db1[j][2*k+1] / coeffs_db1[j-1][k]
    coefs = np.zeros_like(multip)
    multip = np.random.permutation(multip)
    for k in range(coeffs_db1[j-1].shape[0]):
        coefs[2*k] = multip[2*k] * coeffs_tilde[j-1][k]
        coefs[2*k+1] = multip[2*k+1] * coeffs_tilde[j-1][k]
    coeffs_tilde.append(coefs)
        
    
    

        

    
