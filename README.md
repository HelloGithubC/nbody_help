# nbody_help

A Python package to help with N-body simulation tasks, including parameter handling and data I/O.

## Features

- Create linear power spectrum for simulations using CAMB or CLASSY
- Read and write simulation parameter files (Lua and TXT formats)
- Load default parameters for common simulations (COLA, Rockstar)
- Read Gadget2 and Rockstar catalog formats

## Installation

```bash
pip install nbody_help
```

Or install from source:

```bash
git clone https://github.com/yourusername/nbody_help.git
cd nbody_help
pip install .
```

## Quick Start

```python
from nbody_help import create_PS

# Create power spectrum with default cosmology
ps = create_PS(omega_m0=0.3, w0=-1.0, kmax=10.0)

# Read default parameters
from nbody_help import get_default_params
params = get_default_params("cola_halo")
```

## Dependencies

Required:
- numpy
- astropy

Optional:
- colibri (for power spectrum computation, install with `pip install nbody_help[cosmo]`)

## License

MIT License
