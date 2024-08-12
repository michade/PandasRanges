from setuptools import setup
from Cython.Build import cythonize


extensions = [
    'src/PandasRanges/overlaps_cy.pyx'
]


setup(
    name='PandasRanges',
    ext_modules = cythonize(
        extensions,
        build_dir = 'build',
        annotate = True,
        compiler_directives={
            "language_level": 3,
            "profile": False
        }
    )
)
