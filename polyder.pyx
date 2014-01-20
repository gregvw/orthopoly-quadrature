cimport cython
import numpy as np
cimport numpy as np
from polyvander import polyvander
from scipy.misc import factorial

dummy_gobal_name = "foo"

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def polyder(np.ndarray[DTYPE_t,ndim=1] alpha,
            np.ndarray[DTYPE_t,ndim=1] beta,
            np.ndarray[DTYPE_t,ndim=1] x,
            int d=0):
    """ Evaluate dth derivative of the monic polynomials 
        constructed from the three-term recursion relation
       
        p_{k+1}(x) = (x-alpha_k)p_k(x) - beta_k p_{k-1}(x)
        
        where p_0(x) = 1 and p_1(x) = x-alpha_k

        Inputs:
        
        alpha - array of recursion coefficients
        beta - array of recursion coefficients
        x - array of grid points on which to evaluate the polynomials
        d - the order of the derivative to compute
    """
    if alpha.shape[0] != beta.shape[0]:
        raise ValueError('Unequal length input arrays')    

    cdef int N = alpha.shape[0]
    cdef int m = x.shape[0]
    cdef np.ndarray[DTYPE_t,ndim=2] P
    cdef np.ndarray[DTYPE_t,ndim=2] Px
   
    if d == 0:
        P = polyvander(alpha,beta,x)
        Px = P
    else:
        Px = np.zeros((m,N+1),DTYPE)
        P = polyder(alpha,beta,x,d-1)         
        Px[:,d] = factorial(d)
        for k in range(d,N):
            Px[:,k+1] = d*P[:,k] + (x-alpha[k])*Px[:,k] - beta[k]*Px[:,k-1]
    return Px
    

