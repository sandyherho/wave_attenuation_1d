"""
wave_attenuation_1d - Simple 1D Wave Attenuation Model for Coastal Vegetation

Authors: Sandy Herho <sandy.herho@email.ucr.edu>, Iwan Anwar, Faruq Khadami, Theo Ndruru, Rusmawan Suwarman, Dasapta Irawan
Date: 08/01/2025
License: MIT
"""

from .solver import Config, WaveSolver
from .cli import __version__

__all__ = ['Config', 'WaveSolver', '__version__']

# Package metadata
__author__ = 'Sandy Herho, Iwan Anwar, Theo Ndruru, Rusmawan Suwarman, Dasapta Irawan'
__email__ = 'sandy.herho@email.ucr.edu'
__license__ = 'MIT'
