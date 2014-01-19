cimport cython
from scipy.linalg import eig_banded
import numpy as np
cimport numpy as np

dummy_global_name = "foo"

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def gauss(np.ndarray[DTYPE_t,ndim=1] alpha,
          np.ndarray[DTYPE_t,ndim=1] beta):
    """ 
        Compute the Gauss nodes and weights from the recursion 
        coefficients associated with a set of orthogonal polynomials 

        Inputs: 
        alpha - recursion coefficients
        beta - recursion coefficients

        Outputs: 
        x - quadrature nodes		
        w - quadrature weights

        Adapted from the MATLAB code by Walter Gautschi
        http://www.cs.purdue.edu/archives/2002/wxg/codes/gauss.m
    """
    if alpha.shape[0] != beta.shape[0]:
        raise ValueError('Unequal length input arrays')    
 
    cdef int N = alpha.shape[0]
    cdef np.ndarray[DTYPE_t,ndim=1] x
    cdef np.ndarray[DTYPE_t,ndim=1] w
    cdef np.ndarray[DTYPE_t,ndim=2] A

   

    A = np.vstack((np.sqrt(beta),alpha))
    x,V = eig_banded(A,lower=False) 
    w = beta[0]*np.real(V[0,:]**2) 
    return x,w


        
