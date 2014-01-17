cimport cython
from scipy.linalg import solve_banded, solve
from gauss import gauss
import numpy as np
cimport numpy as np

dummy_global_name = "foo"

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def lobatto(np.ndarray[DTYPE_t,ndim=1] alpha,
          np.ndarray[DTYPE_t,ndim=1] beta,
          double xl1, double xl2):
    """
        Compute the Lobatto nodes and weights with the preassigned 
        nodea xl1,xl2
        
        Inputs: 
        alpha - recursion coefficients
        beta - recursion coefficients
        xl1 - assigned node location
        xl2 - assigned node location

        Outputs: 
        x - quadrature nodes        
        w - quadrature weights
   
        Based on the section 7 of the paper 
        "Some modified matrix eigenvalue problems" 
        by Gene Golub, SIAM Review Vol 15, No. 2, April 1973, 
        pp.318--334
    """

    cdef int N = alpha.shape[0]-1
    cdef np.ndarray[DTYPE_t,ndim=1] en
    cdef np.ndarray[DTYPE_t,ndim=2] A1
    cdef np.ndarray[DTYPE_t,ndim=2] J1
    cdef np.ndarray[DTYPE_t,ndim=2] A2 
    cdef np.ndarray[DTYPE_t,ndim=2] J2
    cdef np.ndarray[DTYPE_t,ndim=1] g1
    cdef np.ndarray[DTYPE_t,ndim=1] g2
    cdef np.ndarray[DTYPE_t,ndim=2] C
    cdef np.ndarray[DTYPE_t,ndim=1] xl 
    cdef np.ndarray[DTYPE_t,ndim=1] ab


    en = np.zeros(N,DTYPE)
    en[N-1] = 1
    A1 = np.vstack((np.sqrt(beta),alpha-xl1))
    J1 = np.vstack((A1[:,0:N],A1[0,1:]))
    A2 = np.vstack((np.sqrt(beta),alpha-xl2))
    J2 = np.vstack((A2[:,0:N],A2[0,1:]))
    g1 = solve_banded((1,1),J1,en)
    g2 = solve_banded((1,1),J2,en)
    C = np.array(((1,-g1[N-1]),(1,-g2[N-1])))
    xl = np.array((xl1,xl2))  
    ab = solve(C,xl)
    alpha[N] = ab[0]
    beta[N] = ab[1]
    x,w = gauss(alpha,beta)
    return x,w


