cimport cython
import numpy as np
cimport numpy as np
from scipy.fftpack import ifft

dummy_gobal_name = "foo"

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def clencurt(int N1):
    """
    Compute the Clenshaw Curtis quadrature nodes and weights using the 
    Fast Fourier Transorm. The nodes are the extrema of the Chebyshev
    polynomials, -1, and +1.
    
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
    cdef np.ndarray[DTYPE_t,ndim=2] C
    cdef np.ndarray[DTYPE_t,ndim=1] k
 
    if N1 == 1: # Midpoint rule
        x = np.array([0.0])
        w = np.array([2.0])
    else:    
        C = np.zeros((N1,2),DTYPE)
        k = 2.0*(1.0+np.arange(np.floor(0.5*N)))
        C[::2,0] = 2.0/np.hstack((1.0, 1.0-k*k))
        C[1,1] = -N
        V = np.vstack((C,np.flipud(C[1:N,:])))
        F = np.real(ifft(V, n=None, axis=0))
        x = F[0:N1,1]
        w = np.hstack((F[0,0],2*F[1:N,0],F[N,0]))
    return x,w   
 

