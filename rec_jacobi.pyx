cimport cython
from scipy.special import gamma
import numpy as np
cimport numpy as np

dummy_gobal_name = "foo"

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def rec_jacobi(int N, double a, double b):
    """ Generate the recursion coefficients alpha_k, beta_k 

        P_{k+1}(x) = (x-alpha_k)*P_{k}(x) - beta_k P_{k-1}(x)
 
        for the Jacobi polynomials which are orthogonal on [-1,1] 
        with respect to the weight w(x)=[(1-x)^a]*[(1+x)^b]  

        Inputs: 
        N - polynomial order
        a - weight parameter
        b - weight parameter
 
        Outputs: 
        alpha - recursion coefficients
        beta - recursion coefficients

        Adapted from the MATLAB code by Dirk Laurie and Walter Gautschi
        http://www.cs.purdue.edu/archives/2002/wxg/codes/r_jacobi.m 
    """


    cdef DTYPE_t mu,nu
    cdef np.ndarray[DTYPE_t,ndim=1] alpha = np.zeros(N)
    cdef np.ndarray[DTYPE_t,ndim=1] beta  = np.zeros(N)
    cdef np.ndarray[DTYPE_t,ndim=1] n = np.arange(1.0,N)
    cdef np.ndarray[DTYPE_t,ndim=1] nab =  2.0*n+a+b

    mu = 2.0**(a+b+1.0)*gamma(a+1.0)*gamma(b+1.0)/gamma(a+b+2.0)
    nu = (b-a)/(a+b+2.0)
        
    if N == 1:
        alpha = nu
        beta = mu
    else:
        alpha = np.hstack((nu,(b**2-a**2)/(nab*(nab+2.0))))
        n = n[1:]
        nab = nab[1:]
        B1 = 4.0*(a+1.0)*(b+1.0)/((a+b+2.0)**2.0*(a+b+3.0))
        B = 4.0*(n+a)*(n+b)*n*(n+a+b)/(nab**2.0*(nab+1.0)*(nab-1.0)) 
        beta = np.hstack((mu,B1,B))
    
    return alpha, beta    

