"""
created on Mar 4, 2014

@author: Nikola Jajcay
"""

import numpy as np

def construct_fourier_surrogates(ts):
    """
    Constructs Fourier Transform (FT) surrogates (independent realizations which preserve
    linear structure)
    """
    
    xf = np.fft.rfft(ts, axis = 0)
    
    angle = np.random.uniform(0, 2 * np.pi, xf.shape)
    
    angle[0] = 0
    
    cxf = xf * np.exp(1j * angle)
    
    return np.fft.irfft(cxf, axis = 0)

