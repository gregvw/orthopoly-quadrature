cimport cython
import numpy as np
cimport numpy as np

dummy_gobal_name = "foo"

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def polyvander(np.ndarray[DTYPE_t,ndim=1] alpha,
            np.ndarray[DTYPE_t,ndim=1] beta,
            np.ndarray[DTYPE_t,ndim=1] x):

    """ Evaluate generalized Vandermonde matrix of monic polynomials 
        constructed from the three-term recursion relation

        p_{k+1}(x) = (x-alpha_k)p_k(x) - beta_k p_{k-1}(x)
        
        where p_0(x) = 1 and p_1(x) = x-alpha_k

        Inputs:
        
        alpha - array of recursion coefficients
        beta - array of recursion coefficients
        x - array of grid points on which to evaluate the polynomials
    """

    if alpha.shape[0] != beta.shape[0]:
        raise ValueError('Unequal length input arrays')    

    cdef int N = alpha.shape[0]
    cdef int m = x.shape[0]
    cdef np.ndarray[DTYPE_t,ndim=2] P
 
    P = np.zeros((m,N+1),DTYPE)    

    P[:,0] = 1
    P[:,1] = x-alpha[0]

    for k in range(1,N):
        P[:,k+1] = (x-alpha[k])*P[:,k] - beta[k]*P[:,k-1]

    return P



    
   




