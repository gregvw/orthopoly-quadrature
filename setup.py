# setup.py

from distutils.core import setup
from distutils.extension import Extension
import numpy

try:
    from Cython.Distutils import build_ext
except ImportError:
    from distutils.command import build_ext

setup( name = 'OPQ',
       version = '0.1',
       author = 'Greg von Winckel',
       description = 'Orthogonal Polynomials and Quadrature Methods',
       ext_modules=[Extension("rec_jacobi",["rec_jacobi.pyx"]),
                    Extension("rec_jacobi01",["rec_jacobi01.pyx"]),
                    Extension("gauss",["gauss.pyx"]),
                    Extension("radau",["radau.pyx"]),
                    Extension("lobatto",["lobatto.pyx"]),
                    Extension("fejer",["fejer.pyx"]),
                    Extension("clencurt",["clencurt.pyx"]),
                    Extension("polyvander",["polyvander.pyx"]),
                    Extension("polyder",["polyder.pyx"]),
                    Extension("mm_log",["mm_log.pyx"])],
                        include_dirs=[numpy.get_include()],
       cmdclass = {'build_ext': build_ext},
)          
