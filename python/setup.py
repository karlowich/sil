from setuptools import Extension, setup
import numpy

setup(
    ext_modules=[
        Extension(
            name="sil",
            sources=["silmodule.c"],
            include_dirs=[numpy.get_include()],
            extra_link_args=["-lsil"],
            libraries=["sil"]
        ),
    ]
)