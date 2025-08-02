"""
Setup script for wave_attenuation_1d package
Authors: Sandy Herho, Iwan Anwar, Theo Ndruru, Rusmawan Suwarman, Dasapta Irawan
Date: 08/01/2025
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wave-attenuation-1d",
    version="0.1.0",
    author="Sandy Herho, Iwan Anwar, Theo Ndruru",
    author_email="sandy.herho@email.ucr.edu",
    description="Simple 1D Wave Attenuation Model for Coastal Vegetation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sandyherho/wave_attenuation_1d",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.54.0",
        "netCDF4>=1.5.0",
        "tqdm>=4.62.0",
    ],
    entry_points={
        "console_scripts": [
            "wave-attenuation-1d=wave_attenuation_1d.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.txt"],
    },
)
