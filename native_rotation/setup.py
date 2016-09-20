from setuptools import setup, Extension
import numpy

native_rotation = Extension("native_rotation",
                            sources=["native_rotation.c"],
                            include_dirs=[numpy.get_include()])

setup(name="native_rotation", ext_modules=[native_rotation])
