# setup.py

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup( name = 'OPQ',
       version = '0.1',
       author = 'Greg von Winckel',
       description = 'Orthogonal Polynomials and Quadrature Methods',
       ext_modules=[Extension("rec_jacobi",["rec_jacobi.pyx"]),
                    Extension("gauss",["gauss.pyx"]),
                    Extension("radau",["radau.pyx"])],
                        include_dirs=[numpy.get_include()],
       cmdclass = {'build_ext': build_ext},
)          