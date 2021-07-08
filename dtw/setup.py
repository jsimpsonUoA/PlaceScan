"""Build DTW library"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name='dtw',
    version='1.0.4',
    author='Paul Freeman',
    author_email='paul.freeman.cs@gmail.com',
    license='MIT License',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    url='https://github.com/paul-freeman/dtw',
    description=('An implementation of the Dynamic Time Warping algorithm'),
    ext_modules=cythonize("dtw/dtw.pyx",include_path = [numpy.get_include()])
)
