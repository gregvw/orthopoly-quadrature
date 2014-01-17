cimport cython
from scipy.linalg import solve_banded 
from gauss import gauss
import numpy as np
cimport numpy as np

dummy_global_name = "foo"

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def radau(np.ndarray[DTYPE_t,ndim=1] alpha,
          np.ndarray[DTYPE_t,ndim=1] beta,
          double xr):
    """
        Compute the Radau nodes and weights with the preassigned 
        node xr
        
        Inputs: 
        alpha - recursion coefficients
        beta - recursion coefficients
        xr - assigned node location

        Outputs: 
        x - quadrature nodes		
        w - quadrature weights
   
        Based on the section 7 of the paper "Some modified matrix 
        eigenvalue problems" by Gene Golub, SIAM Review Vol 15, 
        No. 2, April 1973, pp.318--334
    """


    cdef int N = alpha.shape[0]-1
    cdef np.ndarray[DTYPE_t,ndim=1] x
    cdef np.ndarray[DTYPE_t,ndim=1] w
    cdef np.ndarray[DTYPE_t,ndim=1] f
    cdef np.ndarray[DTYPE_t,ndim=1] delta
    cdef np.ndarray[DTYPE_t,ndim=2] A
    cdef np.ndarray[DTYPE_t,ndim=2] J
    cdef np.ndarray[DTYPE_t,ndim=1] alphar

    f = np.zeros(N,DTYPE)
    f[N-1] = beta[N]
    A = np.vstack((np.sqrt(beta),alpha-xr))
    J = np.vstack((A[:,0:N],A[0,1:]))
    delta = solve_banded((1,1),J,f)
    alphar = alpha
    alphar[N] = xr + delta[N-1]
    x,w = gauss(alphar,beta)

    return x,w


