# wave-attenuation-1d: Simple 1D Wave Attenuation Model for Coastal Vegetation

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/wave-attenuation-1d.svg)](https://badge.fury.io/py/wave-attenuation-1d)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/1029883701.svg)](https://doi.org/10.5281/zenodo.16729568)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A fast numerical solver for simulating wave attenuation through coastal vegetation, solving the linearized shallow water equations with vegetation-induced drag.

## Overview

This package models how coastal vegetation (e.g., mangroves, salt marshes) reduces wave energy, providing a nature-based solution for coastal protection. The model uses high-order numerical methods with Numba acceleration for efficient computation.

## Mathematical Model

The solver implements the 1D linearized shallow water equations with vegetation drag:

$\frac{\partial \eta}{\partial t} + h\frac{\partial u}{\partial x} = 0$

$\frac{\partial u}{\partial t} + g\frac{\partial \eta}{\partial x} = -c_D \chi_{\text{veg}}(x) u$

where:
- $\eta(x,t)$ - free surface elevation [m]
- $u(x,t)$ - depth-averaged velocity [m/s]  
- $h$ - water depth [m]
- $g$ - gravitational acceleration (9.81 m/s²)
- $c_D$ - linearized drag coefficient [1/s]
- $\chi_{\text{veg}}(x)$ - vegetation indicator function

## Installation

```bash
pip install wave-attenuation-1d
```

### Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- Numba ≥ 0.54.0
- netCDF4 ≥ 1.5.0
- tqdm ≥ 4.62.0

## Quick Start

```bash
# Run with example configuration
wave-attenuation-1d configs/sparse_config.txt

# Run with custom config
wave-attenuation-1d my_config.txt

# Check version
wave-attenuation-1d --version
```

## Features

- **4th-order Runge-Kutta** time integration with implicit drag treatment
- **Numba JIT compilation** for ~100x speedup over pure Python
- **NetCDF output** with CF-1.8 compliant metadata
- **Automatic transmission coefficient** calculation
- **Configurable vegetation patches** with varying drag coefficients

## Configuration

Create a text configuration file:

```ini
[DOMAIN]
L = 200.0      # Domain length [m]
d = 2.0        # Water depth [m]
dx = 0.5       # Grid spacing [m]
T = 500.0      # Simulation time [s]

[WAVE]
A = 0.3        # Wave amplitude [m]
omega = 0.628  # Angular frequency [rad/s]

[VEGETATION]
start = 80.0   # Vegetation start [m]
end = 120.0    # Vegetation end [m]
cD = 0.14      # Drag coefficient [1/s]

[NUMERICAL]
cfl_target = 0.4    # Target CFL number
output_dt = 1.0     # Output interval [s]
```

## Output Format

Results are saved as NetCDF files containing:
- Time series of surface elevation and velocity
- Wave envelope and RMS velocity
- Metadata including all parameters and transmission coefficients

## Numerical Methods

- **Spatial discretization**: 2nd-order centered differences
- **Time integration**: 4th-order Runge-Kutta (RK4)
- **Drag treatment**: Implicit scheme for unconditional stability
- **Boundary conditions**: 
  - Left: Sinusoidal wave generation
  - Right: Sommerfeld radiation condition

## Citation

If you use this model in your research, please cite:

```bibtex
@software{herho2025wave,
  author = {Herho, Sandy H. S. and Anwar, Iwan P. and Ndruru, Theo R. E. B. N. 
            and Suwarman, Rusmawan and Irawan, Dasapta E.},
  title = {{wave-attenuation-1d: Simple 1D Wave Attenuation Model for Coastal Vegetation}},
  year = {2025},
  url = {https://github.com/sandyherho/wave_attenuation_1d}
}
```

## Authors

- Sandy H. S. Herho (sandy.herho@email.ucr.edu)
- Iwan P. Anwar
- Faruq Khadami
- Theo R. E. B. N. Ndruru  
- Rusmawan Suwarman
- Dasapta E. Irawan

## License

MIT License - see [LICENSE](LICENSE) file for details.
