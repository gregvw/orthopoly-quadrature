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
    """ Evaluate polynomial Vandermonde matrix on x given the three-term 
        recursion coefficients alpha and beta """

    cdef int N = alpha.shape[0]
    cdef int m = x.shape[0]
    cdef np.ndarray[DTYPE_t,ndim=2] V
 
    V = np.zeros((m,N+1),DTYPE)    

    V[:,0] = 1
    V[:,1] = (x-alpha[0])*V[:,0] 

    for k in range(1,N):
        V[:,k+1] = (x-alpha[k])*V[:,k] - beta[k]*V[:,k-1]

    return V 



    
   




