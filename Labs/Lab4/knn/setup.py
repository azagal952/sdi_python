from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

extensions = [
    Extension("knn_pure_python", ["knn_pure_python.pyx"], include_dirs=[numpy.get_include()]),
    Extension("knn_typed", ["knn_typed.pyx"], include_dirs=[numpy.get_include()]),
    Extension("knn_memview", ["knn_memview.pyx"], include_dirs=[numpy.get_include()]),
    Extension("knn_contiguous", ["knn_contiguous.pyx"], include_dirs=[numpy.get_include()]),
    Extension("knn_infer_types", ["knn_infer_types.pyx"], include_dirs=[numpy.get_include()]),
    Extension("knn_tuning_indexing", ["knn_tuning_indexing.pyx"], include_dirs=[numpy.get_include()]),
    Extension("knn_several_types", ["knn_several_types.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    name="compute_cy",
    ext_modules=cythonize(extensions, annotate=True, language_level="3"),
)
