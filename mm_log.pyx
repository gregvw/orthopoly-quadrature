cimport cython
import numpy as np
cimport numpy as np
from scipy.special import gamma

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def mm_log(int N, double a):
    """ 
    Modified moments for a logarithmic weight function.
    
    The call mm=MM_LOG(n,a) computes the first n modified moments of the
    logarithmic weight function w(t)=t^a log(1/t) on [0,1] relative to 
    shifted Legendre polynomials. 

    REFERENCE:  Walter Gautschi,``On the preceding paper `A Legendre 
                polynomial integral' by James L. Blue'', 
                Math. Comp. 33 (1979), 742-743.

    Adapted from the MATLAB implementation: 
    https://www.cs.purdue.edu/archives/2002/wxg/codes/mm_log.m 
    """
    cdef int ai = int(a)
    cdef int n
    cdef DTYPE_t c
    cdef np.ndarray[DTYPE_t,ndim=1] mm
    cdef np.ndarray[DTYPE_t,ndim=1] p
    cdef np.ndarray[DTYPE_t,ndim=1] k

    if a <= -1:
        raise ValueError('Parameter a must be greater than -1')
    if N<1:
        raise ValueError('N must be at least 1')

    # Compute product of elements
    prod = lambda z: reduce(lambda x,y:x*y,z,1)

    c = 1.0
    mm = np.zeros(N,DTYPE)

    for n in range(N):
        if ai == a and a<n:
            p = np.arange(n-a,n+a+2)
            mm[n] = (-1)**(n-a)/prod(p)
            mm[n] *= gamma(a+1)**2            
        else:
            if n == 0:
                mm[0] = 1.0/(a+1.0)**2.0
            else:
                k = np.arange(1.0,n+1.0)
                s = 1.0/(a+1.0+k)-1.0/(a+1.0-k)
                p = (a+1.0-k)/(a+1.0+k)
                mm[n] = (1.0/(a+1.0)+sum(s))*prod(p)/(a+1.0);
        
        mm[n] *= c
        c *= 0.5*(n+1.0)/(2.0*n+1.0)                 

    return mm
