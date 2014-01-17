cimport cython
import numpy as np
cimport numpy as np
from rec_jacobi import rec_jacobi


dummy_gobal_name = "foo"

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def rec_jacobi01(int N, double a, double b):
    """
    Generate the recursion coefficients for the Jacobi polynomials
    defined on the interval [0,1]. These polynomials are 
    orthogonal with respect to a weight function w(x)=[(1-x)]^a*[x]^b

    Inputs: 
    N - polynomial order
    a - weight parameter
    b - weight parameter
 
    Outputs: 
    alpha01 - recursion coefficients
    beta01 - recursion coefficients

    Adapted from the MATLAB code by Walter Gautschi
    http://www.cs.purdue.edu/archives/2002/wxg/codes/r_jacobi01.m 

    """

    cdef np.ndarray[DTYPE_t,ndim=1] alpha = np.zeros(N)
    cdef np.ndarray[DTYPE_t,ndim=1] beta = np.zeros(N)
    cdef np.ndarray[DTYPE_t,ndim=1] alpha01 = np.zeros(N)
    cdef np.ndarray[DTYPE_t,ndim=1] beta01 = np.zeros(N)

    alpha,beta = rec_jacobi(N,a,b)
    alpha01 = 0.5*(1.0+alpha)    
    beta01 = 0.25*beta
    beta01[0] = beta[0]/2*(a+b+1)
    return alpha01, beta01


    


