from setuptools import setup, find_packages
from Cython.Build import cythonize


setup(
    name='variational_bayes',
    version='0.1',
    author='tillahoffmann',
    packages=find_packages(),
    ext_modules=cythonize(["variational_bayes/*.pyx"], annotate=True)
)
