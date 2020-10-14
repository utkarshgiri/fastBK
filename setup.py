from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
        name = 'mpi_cythonized_cic',
        ext_modules=cythonize('cycic.pyx'),
        include_dirs=[numpy.get_include()]
        )

