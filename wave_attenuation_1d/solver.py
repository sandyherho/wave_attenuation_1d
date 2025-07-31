"""
Simple 1D Wave Attenuation Solver
Solves shallow water equations with vegetation-induced dissipation

Authors: Sandy Herho <sandy.herho@email.ucr.edu>, Iwan Anwar, Theo Ndruru
Date: 08/01/2025
License: MIT
"""

import numpy as np
import numba as nb
import netCDF4 as nc
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
import logging
from pathlib import Path

# Constants
GRAVITY = 9.81


@dataclass
class Config:
    """Model configuration parameters"""
    # Domain
    L: float          # Domain length [m]
    d: float          # Water depth [m]
    dx: float         # Spatial step [m]
    T: float          # Simulation time [s]
    
    # Wave
    A: float          # Wave amplitude [m]
    omega: float      # Wave frequency [rad/s]
    
    # Vegetation patch
    veg_start: float  # Vegetation start [m]
    veg_end: float    # Vegetation end [m]
    cD: float         # Drag coefficient [m^2/s]
    
    # Numerical
    cfl_target: float # Target CFL number
    output_dt: float  # Output interval [s]


@nb.njit
def compute_fluxes(eta, u, d, g, dx):
    """Compute numerical fluxes using centered differences"""
    nx = len(eta)
    flux_eta = np.zeros(nx)
    flux_u = np.zeros(nx-1)
    
    # Flux for eta equation (continuity)
    for i in range(1, nx-1):
        q_left = d * u[i-1]
        q_right = d * u[i]
        flux_eta[i] = -(q_right - q_left) / dx
    
    # Flux for u equation (momentum)
    for i in range(nx-1):
        flux_u[i] = -g * (eta[i+1] - eta[i]) / dx
    
    return flux_eta, flux_u


@nb.njit
def apply_vegetation_drag(u, veg_mask, cD, dt):
    """Apply vegetation drag implicitly"""
    u_new = np.zeros_like(u)
    for i in range(len(u)):
        if veg_mask[i]:
            # Implicit treatment for stability
            u_new[i] = u[i] / (1.0 + cD * dt)
        else:
            u_new[i] = u[i]
    return u_new


@nb.njit  
def rk4_step(eta, u, veg_mask, cD, g, d, dx, dt, A, omega, t):
    """4th order Runge-Kutta time step"""
    nx = len(eta)
    
    # Stage 1
    flux_eta1, flux_u1 = compute_fluxes(eta, u, d, g, dx)
    k1_eta = dt * flux_eta1
    k1_u = dt * flux_u1
    
    # Update with drag for u
    u_temp1 = apply_vegetation_drag(u + 0.5*k1_u, veg_mask, cD, 0.5*dt)
    
    # Stage 2
    flux_eta2, flux_u2 = compute_fluxes(eta + 0.5*k1_eta, u_temp1, d, g, dx)
    k2_eta = dt * flux_eta2
    k2_u = dt * flux_u2
    
    u_temp2 = apply_vegetation_drag(u + 0.5*k2_u, veg_mask, cD, 0.5*dt)
    
    # Stage 3
    flux_eta3, flux_u3 = compute_fluxes(eta + 0.5*k2_eta, u_temp2, d, g, dx)
    k3_eta = dt * flux_eta3
    k3_u = dt * flux_u3
    
    u_temp3 = apply_vegetation_drag(u + k3_u, veg_mask, cD, dt)
    
    # Stage 4
    flux_eta4, flux_u4 = compute_fluxes(eta + k3_eta, u_temp3, d, g, dx)
    k4_eta = dt * flux_eta4
    k4_u = dt * flux_u4
    
    # Combine stages
    eta_new = eta + (k1_eta + 2*k2_eta + 2*k3_eta + k4_eta) / 6
    u_new = u + (k1_u + 2*k2_u + 2*k3_u + k4_u) / 6
    
    # Final drag application
    u_new = apply_vegetation_drag(u_new, veg_mask, cD, dt)
    
    # Apply boundary conditions
    # Wave maker at left boundary
    eta_new[0] = A * np.sin(omega * (t + dt))
    u_new[0] = np.sqrt(g/d) * eta_new[0]
    
    # Radiation BC at right boundary
    c0 = np.sqrt(g * d)
    eta_new[-1] = eta_new[-2] - c0 * dt/dx * (eta_new[-1] - eta_new[-2])
    u_new[-1] = np.sqrt(g/d) * eta_new[-1]
    
    return eta_new, u_new


class WaveSolver:
    """Main wave attenuation solver class"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.setup_domain()
        self.setup_vegetation()
        
    def setup_domain(self):
        """Setup computational domain"""
        # Spatial grid
        self.nx = int(self.config.L / self.config.dx) + 1
        self.x = np.linspace(0, self.config.L, self.nx)
        
        # Wave properties
        self.g = GRAVITY
        self.c0 = np.sqrt(self.g * self.config.d)
        self.k = self.config.omega / self.c0
        self.wavelength = 2 * np.pi / self.k
        self.period = 2 * np.pi / self.config.omega
        
        # Time step from CFL condition
        self.dt = self.config.cfl_target * self.config.dx / self.c0
        self.nt = int(self.config.T / self.dt) + 1
        
        # Output times
        self.output_interval = max(1, int(self.config.output_dt / self.dt))
        self.output_times = list(range(0, self.nt, self.output_interval))
        if self.nt - 1 not in self.output_times:
            self.output_times.append(self.nt - 1)
        
        self.logger.info(f"Domain setup:")
        self.logger.info(f"  Grid: nx={self.nx}, dx={self.config.dx:.3f} m")
        self.logger.info(f"  Time: dt={self.dt:.4f} s, nt={self.nt}")
        self.logger.info(f"  Wave: λ={self.wavelength:.1f} m, T={self.period:.1f} s")
        self.logger.info(f"  CFL number: {self.c0 * self.dt / self.config.dx:.3f}")
        
    def setup_vegetation(self):
        """Setup vegetation patch"""
        # Vegetation mask at cell faces (velocity points)
        self.veg_mask = np.zeros(self.nx - 1, dtype=bool)
        x_faces = 0.5 * (self.x[:-1] + self.x[1:])
        
        self.veg_mask = ((x_faces >= self.config.veg_start) & 
                        (x_faces <= self.config.veg_end))
        
        veg_length = self.config.veg_end - self.config.veg_start
        self.logger.info(f"Vegetation patch:")
        self.logger.info(f"  Location: [{self.config.veg_start:.1f}, {self.config.veg_end:.1f}] m")
        self.logger.info(f"  Length: {veg_length:.1f} m ({veg_length/self.config.L*100:.1f}% of domain)")
        self.logger.info(f"  Drag coefficient: {self.config.cD:.3f} m²/s")
        
    def solve(self):
        """Run the numerical simulation"""
        self.logger.info("Starting simulation...")
        
        # Initialize arrays
        eta = np.zeros(self.nx)
        u = np.zeros(self.nx - 1)
        
        # Storage for outputs
        n_outputs = len(self.output_times)
        self.eta_history = np.zeros((n_outputs, self.nx))
        self.u_history = np.zeros((n_outputs, self.nx - 1))
        self.t_output = np.zeros(n_outputs)
        
        # Initial condition
        eta[0] = self.config.A * np.sin(0)
        
        # Time stepping with progress bar
        output_idx = 0
        with tqdm(total=self.nt, desc="Time stepping", unit="steps") as pbar:
            for n in range(self.nt):
                t = n * self.dt
                
                # Store output
                if n in self.output_times:
                    self.eta_history[output_idx] = eta
                    self.u_history[output_idx] = u
                    self.t_output[output_idx] = t
                    output_idx += 1
                
                # RK4 time step
                eta, u = rk4_step(eta, u, self.veg_mask, self.config.cD,
                                self.g, self.config.d, self.config.dx, 
                                self.dt, self.config.A, self.config.omega, t)
                
                pbar.update(1)
        
        self.logger.info("Simulation completed")
        
    def calculate_transmission(self):
        """Calculate transmission coefficient"""
        # Use last 20% of simulation (steady state)
        n_steady = int(0.2 * len(self.output_times))
        
        # Measurement locations
        x_in = self.config.veg_start - 2 * self.wavelength
        x_out = self.config.veg_end + 2 * self.wavelength
        
        # Ensure within domain
        x_in = max(self.wavelength, x_in)
        x_out = min(self.config.L - self.wavelength, x_out)
        
        idx_in = np.argmin(np.abs(self.x - x_in))
        idx_out = np.argmin(np.abs(self.x - x_out))
        
        # Calculate wave heights
        H_in = 2 * np.std(self.eta_history[-n_steady:, idx_in])
        H_out = 2 * np.std(self.eta_history[-n_steady:, idx_out])
        
        self.Kt = H_out / H_in
        
        self.logger.info(f"Transmission coefficient:")
        self.logger.info(f"  Measurement locations: x_in={x_in:.1f} m, x_out={x_out:.1f} m")
        self.logger.info(f"  Wave heights: H_in={H_in:.3f} m, H_out={H_out:.3f} m")
        self.logger.info(f"  Kt = {self.Kt:.4f} (reduction = {(1-self.Kt)*100:.1f}%)")
        
    def save_results(self, filename: str):
        """Save results to NetCDF file"""
        self.logger.info(f"Saving results to {filename}")
        
        # Calculate additional fields
        # Wave envelope (maximum over last few periods)
        n_periods = min(5, int(5 * self.period / self.config.output_dt))
        n_periods = min(n_periods, len(self.output_times))
        self.envelope = np.max(np.abs(self.eta_history[-n_periods:]), axis=0)
        
        # Phase-averaged velocity magnitude
        self.u_rms = np.sqrt(np.mean(self.u_history[-n_periods:]**2, axis=0))
        
        # Ensure output directory exists
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with nc.Dataset(filename, 'w', format='NETCDF4') as ds:
            # Dimensions
            ds.createDimension('x', self.nx)
            ds.createDimension('x_face', self.nx - 1)
            ds.createDimension('time', len(self.output_times))
            
            # Global attributes
            ds.title = "Simple 1D Wave Attenuation Model Results"
            ds.model = "wave_attenuation_1d"
            ds.authors = "Sandy Herho, Iwan Anwar, Theo Ndruru"
            ds.created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Model parameters
            ds.domain_length = self.config.L
            ds.water_depth = self.config.d
            ds.wave_amplitude = self.config.A
            ds.wave_period = self.period
            ds.wavelength = self.wavelength
            ds.vegetation_start = self.config.veg_start
            ds.vegetation_end = self.config.veg_end
            ds.drag_coefficient = self.config.cD
            ds.transmission_coefficient = self.Kt
            
            # Coordinate variables
            x = ds.createVariable('x', 'f8', ('x',))
            x.units = 'm'
            x.long_name = 'distance along domain'
            x[:] = self.x
            
            x_face = ds.createVariable('x_face', 'f8', ('x_face',))
            x_face.units = 'm'
            x_face.long_name = 'distance at cell faces'
            x_face[:] = 0.5 * (self.x[:-1] + self.x[1:])
            
            time = ds.createVariable('time', 'f8', ('time',))
            time.units = 's'
            time.long_name = 'time'
            time[:] = self.t_output
            
            # 2D fields (space-time)
            eta = ds.createVariable('eta', 'f4', ('time', 'x'))
            eta.units = 'm'
            eta.long_name = 'surface elevation'
            eta[:] = self.eta_history
            
            u = ds.createVariable('u', 'f4', ('time', 'x_face'))
            u.units = 'm/s'
            u.long_name = 'horizontal velocity'
            u[:] = self.u_history
            
            # 1D fields (space only)
            envelope = ds.createVariable('envelope', 'f4', ('x',))
            envelope.units = 'm'
            envelope.long_name = 'wave envelope'
            envelope[:] = self.envelope
            
            u_rms = ds.createVariable('u_rms', 'f4', ('x_face',))
            u_rms.units = 'm/s'
            u_rms.long_name = 'RMS velocity'
            u_rms[:] = self.u_rms
            
            veg_mask = ds.createVariable('vegetation', 'i1', ('x_face',))
            veg_mask.long_name = 'vegetation presence'
            veg_mask[:] = self.veg_mask.astype(int)
            
        self.logger.info("Results saved successfully")
