cimport cython
import numpy as np
cimport numpy as np
from scipy.fftpack import dct

dummy_gobal_name = "foo"

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def fejer(int N1):
    """
    Compute the Fejér quadrature nodes and weights using the 
    Discrete Cosine Transform. The nodes are the roots of the Chebyshev
    polynomials.
    
    Reference: Jörg Waldvogel, "Fast Construction of the Fejér and
               Clenshaw-Curtis Quadrature Rules," BIT Numerical Mathematics 
               46 (1), p. 195-202 (2006).

    http://www.sam.math.ethz.ch/~waldvoge/Papers/fejer.pdf
    """

    if N<1:
        raise ValueError('Quadrature rule must have at least one point')   

    cdef int N = N1-1
    cdef np.ndarray[DTYPE_t,ndim=1] x
    cdef np.ndarray[DTYPE_t,ndim=1] w
    cdef np.ndarray[DTYPE_t,ndim=1] c
    cdef np.ndarray[DTYPE_t,ndim=1] k

    if N1 == 1: # Midpoint rule
        x = np.array([0.0])
        w = np.array([2.0])
    else:
       
        c = np.zeros(N1,DTYPE)
        k = 2.0*(1.0+np.arange(np.floor(0.5*N)))
        c[::2] = (2.0/N1)/np.hstack((1, 1-k*k))
        x = -np.cos(np.pi*(2.0*np.arange(N1)+1.0)/(2.0*N1)) 
        w = dct(c,type=3,axis=0,norm=None)
    return x,w

            
    
