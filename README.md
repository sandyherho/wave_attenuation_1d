# wave_attenuation_1d

Simple 1D Wave Attenuation Model for Coastal Vegetation

## Description

This package provides a simplified 1D numerical model for simulating wave attenuation through coastal vegetation (e.g., seagrass, mangroves). It solves the shallow water equations with vegetation-induced drag using a 4th-order Runge-Kutta scheme.

## Features

- Fast numerical solver using Numba acceleration
- Configurable via simple text files
- NetCDF output format
- Automatic transmission coefficient calculation
- Support for different vegetation densities

## Installation

```bash
pip install wave-attenuation-1d
```

## Usage

```bash
# Run with example configuration
wave-attenuation-1d configs/sparse_config.txt

# Check version
wave-attenuation-1d --version

# Get help
wave-attenuation-1d --help
```

## Output

Results are saved in:
- `outputs/`: NetCDF files with simulation results
- `logs/`: Simulation log files

## Configuration

Example configuration files are provided in the `configs/` directory:
- `sparse_config.txt`: Sparse vegetation (e.g., seagrass)
- `dense_config.txt`: Dense vegetation (e.g., mangroves)

## Physics

The model solves the linearized shallow water equations:
- ∂η/∂t + ∂(hu)/∂x = 0
- ∂u/∂t + g∂η/∂x = -cD·u (in vegetation)

Where:
- η: surface elevation
- u: horizontal velocity
- h: water depth
- g: gravitational acceleration
- cD: vegetation drag coefficient

## License

MIT License - see LICENSE file

## Authors

- Sandy Herho <sandy.herho@email.ucr.edu>
- Iwan Anwar
- Theo Ndruru

## Citation

If you use this model in your research, please cite:
```
Herho, S., Anwar, I., & Ndruru, T. (2025). wave_attenuation_1d: 
Simple 1D Wave Attenuation Model for Coastal Vegetation. 
https://github.com/sandyherho/wave_attenuation_1d
```
