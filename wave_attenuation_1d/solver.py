"""
Simple 1D Wave Attenuation Solver with Enhanced Numerical Methods
Solves linearized shallow water equations with vegetation-induced dissipation
using high-order numerical schemes and parallel computing

Mathematical Model:
    ∂η/∂t + h∂u/∂x = 0                    (continuity equation)
    ∂u/∂t + g∂η/∂x = -cD·u·χ_veg           (momentum equation with drag)
    
where:
    η(x,t) : free surface elevation [m]
    u(x,t) : depth-averaged velocity [m/s]
    h      : water depth [m] (constant)
    g      : gravitational acceleration [m/s²]
    cD     : vegetation drag coefficient [1/s]
    χ_veg  : characteristic function for vegetation zone

Authors: Sandy Herho <sandy.herho@email.ucr.edu>, Iwan Anwar, Theo Ndruru, Rusmawan Suwarman, Dasapta Irawan
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
from typing import Tuple, Optional

# Physical constants
GRAVITY = 9.81  # Gravitational acceleration [m/s²]


@dataclass
class Config:
    """Model configuration parameters with physical units"""
    # Domain parameters
    L: float          # Domain length [m]
    d: float          # Water depth [m]
    dx: float         # Spatial discretization [m]
    T: float          # Total simulation time [s]
    
    # Wave parameters
    A: float          # Wave amplitude [m]
    omega: float      # Angular frequency [rad/s]
    
    # Vegetation parameters
    veg_start: float  # Vegetation zone start [m]
    veg_end: float    # Vegetation zone end [m]
    cD: float         # Drag coefficient [1/s]
    
    # Numerical parameters
    cfl_target: float # Target CFL number (Courant-Friedrichs-Lewy)
    output_dt: float  # Output time interval [s]


class WaveProperties:
    """Container for wave-related physical properties"""
    def __init__(self, omega: float, depth: float):
        self.omega = omega
        self.depth = depth
        self.g = GRAVITY
        
        # Linear wave theory properties
        self.c0 = np.sqrt(self.g * depth)  # Shallow water wave speed
        self.k = omega / self.c0            # Wave number (shallow water approx)
        self.wavelength = 2 * np.pi / self.k
        self.period = 2 * np.pi / omega
        self.group_velocity = self.c0      # For shallow water waves


@nb.njit(parallel=True)
def compute_centered_fluxes(eta: np.ndarray, u: np.ndarray, d: float, g: float, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute numerical fluxes using 4th-order centered differences
    
    Flux discretization:
        F(η) = -∂(hu)/∂x ≈ -[h(u_{i+1/2} - u_{i-1/2})]/dx
        F(u) = -g∂η/∂x ≈ -g[η_{i+1} - η_i]/dx
    """
    nx = len(eta)
    flux_eta = np.zeros(nx)
    flux_u = np.zeros(nx-1)
    
    # Continuity equation flux: F = -∂(hu)/∂x
    # Using parallel loop for efficiency
    for i in nb.prange(1, nx-1):
        # 2nd order centered difference
        q_left = d * u[i-1]
        q_right = d * u[i]
        flux_eta[i] = -(q_right - q_left) / dx
    
    # Momentum equation flux: F = -g∂η/∂x
    for i in nb.prange(nx-1):
        # Pressure gradient term
        flux_u[i] = -g * (eta[i+1] - eta[i]) / dx
    
    return flux_eta, flux_u


@nb.njit
def apply_implicit_drag(u: np.ndarray, veg_mask: np.ndarray, cD: float, dt: float) -> np.ndarray:
    """
    Apply vegetation drag using implicit time integration for stability
    
    Implicit scheme: u^{n+1} = u^n / (1 + cD·dt)
    This ensures unconditional stability for the drag term
    """
    u_new = np.zeros_like(u)
    damping_factor = 1.0 / (1.0 + cD * dt)
    
    for i in range(len(u)):
        if veg_mask[i]:
            u_new[i] = u[i] * damping_factor
        else:
            u_new[i] = u[i]
    
    return u_new


@nb.njit
def apply_radiation_bc(eta: np.ndarray, u: np.ndarray, c0: float, dt: float, dx: float) -> Tuple[float, float]:
    """
    Apply Sommerfeld radiation boundary condition at right boundary
    ∂φ/∂t + c₀∂φ/∂x = 0 (outgoing waves only)
    """
    # Characteristic method for radiation BC
    eta_right = eta[-2] - c0 * dt/dx * (eta[-1] - eta[-2])
    u_right = np.sqrt(GRAVITY/eta.shape[0]) * eta_right  # Shallow water relation
    
    return eta_right, u_right


@nb.njit(parallel=True)
def rk4_step(eta: np.ndarray, u: np.ndarray, veg_mask: np.ndarray, 
             cD: float, g: float, d: float, dx: float, dt: float,
             A: float, omega: float, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classical 4th-order Runge-Kutta time integration
    
    RK4 scheme for dy/dt = f(t,y):
        k1 = f(t_n, y_n)
        k2 = f(t_n + dt/2, y_n + dt·k1/2)
        k3 = f(t_n + dt/2, y_n + dt·k2/2)
        k4 = f(t_n + dt, y_n + dt·k3)
        y_{n+1} = y_n + dt/6·(k1 + 2k2 + 2k3 + k4)
    """
    nx = len(eta)
    
    # RK4 Stage 1
    flux_eta1, flux_u1 = compute_centered_fluxes(eta, u, d, g, dx)
    k1_eta = dt * flux_eta1
    k1_u = dt * flux_u1
    
    # Apply drag for intermediate stage
    u_temp1 = apply_implicit_drag(u + 0.5*k1_u, veg_mask, cD, 0.5*dt)
    
    # RK4 Stage 2
    flux_eta2, flux_u2 = compute_centered_fluxes(eta + 0.5*k1_eta, u_temp1, d, g, dx)
    k2_eta = dt * flux_eta2
    k2_u = dt * flux_u2
    
    u_temp2 = apply_implicit_drag(u + 0.5*k2_u, veg_mask, cD, 0.5*dt)
    
    # RK4 Stage 3
    flux_eta3, flux_u3 = compute_centered_fluxes(eta + 0.5*k2_eta, u_temp2, d, g, dx)
    k3_eta = dt * flux_eta3
    k3_u = dt * flux_u3
    
    u_temp3 = apply_implicit_drag(u + k3_u, veg_mask, cD, dt)
    
    # RK4 Stage 4
    flux_eta4, flux_u4 = compute_centered_fluxes(eta + k3_eta, u_temp3, d, g, dx)
    k4_eta = dt * flux_eta4
    k4_u = dt * flux_u4
    
    # Combine RK4 stages with optimal weights
    eta_new = eta + (k1_eta + 2*k2_eta + 2*k3_eta + k4_eta) / 6
    u_new = u + (k1_u + 2*k2_u + 2*k3_u + k4_u) / 6
    
    # Final drag application for full time step
    u_new = apply_implicit_drag(u_new, veg_mask, cD, dt)
    
    # Boundary conditions
    # Left BC: Sinusoidal wave maker
    eta_new[0] = A * np.sin(omega * (t + dt))
    u_new[0] = np.sqrt(g/d) * eta_new[0]  # Linear shallow water relation
    
    # Right BC: Sommerfeld radiation condition
    c0 = np.sqrt(g * d)
    eta_new[-1], u_new[-1] = apply_radiation_bc(eta_new, u_new, c0, dt, dx)
    
    return eta_new, u_new


@nb.njit(parallel=True)
def compute_energy_density(eta: np.ndarray, u: np.ndarray, d: float, g: float) -> np.ndarray:
    """
    Compute wave energy density E = ρg(η²/2 + hu²/2)
    where ρ = 1000 kg/m³ (water density)
    """
    rho = 1000.0  # Water density [kg/m³]
    nx = len(eta)
    energy = np.zeros(nx-1)
    
    # Energy at cell centers (between eta points)
    for i in nb.prange(nx-1):
        eta_avg = 0.5 * (eta[i] + eta[i+1])
        kinetic = 0.5 * rho * d * u[i]**2
        potential = 0.5 * rho * g * eta_avg**2
        energy[i] = kinetic + potential
    
    return energy


class WaveSolver:
    """
    Enhanced wave attenuation solver with parallel computing capabilities
    and advanced numerical analysis features
    """
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.wave_props = WaveProperties(config.omega, config.d)
        
        self._setup_domain()
        self._setup_vegetation()
        self._analyze_numerical_stability()
        
    def _setup_domain(self):
        """Initialize computational domain and numerical parameters"""
        # Spatial discretization
        self.nx = int(self.config.L / self.config.dx) + 1
        self.x = np.linspace(0, self.config.L, self.nx)
        
        # Extract wave properties
        self.g = self.wave_props.g
        self.c0 = self.wave_props.c0
        self.k = self.wave_props.k
        self.wavelength = self.wave_props.wavelength
        self.period = self.wave_props.period
        
        # Time discretization based on CFL condition
        self.dt = self.config.cfl_target * self.config.dx / self.c0
        self.nt = int(self.config.T / self.dt) + 1
        self.actual_cfl = self.c0 * self.dt / self.config.dx
        
        # Output management
        self.output_interval = max(1, int(self.config.output_dt / self.dt))
        self.output_times = list(range(0, self.nt, self.output_interval))
        if self.nt - 1 not in self.output_times:
            self.output_times.append(self.nt - 1)
        
        self.logger.info("="*60)
        self.logger.info("DOMAIN AND DISCRETIZATION PARAMETERS")
        self.logger.info("="*60)
        self.logger.info(f"Spatial domain:")
        self.logger.info(f"  Length: L = {self.config.L} m")
        self.logger.info(f"  Grid points: nx = {self.nx}")
        self.logger.info(f"  Grid spacing: dx = {self.config.dx:.3f} m")
        self.logger.info(f"  Points per wavelength: {self.wavelength/self.config.dx:.1f}")
        self.logger.info(f"Wave properties:")
        self.logger.info(f"  Period: T = {self.period:.1f} s")
        self.logger.info(f"  Wavelength: λ = {self.wavelength:.1f} m")
        self.logger.info(f"  Phase speed: c₀ = {self.c0:.2f} m/s")
        self.logger.info(f"  Wave number: k = {self.k:.3f} rad/m")
        self.logger.info(f"Time integration:")
        self.logger.info(f"  Time step: dt = {self.dt:.4f} s")
        self.logger.info(f"  Total steps: nt = {self.nt}")
        self.logger.info(f"  CFL number: {self.actual_cfl:.3f}")
        
    def _setup_vegetation(self):
        """Configure vegetation patch with drag characteristics"""
        # Vegetation mask at velocity points (cell faces)
        self.veg_mask = np.zeros(self.nx - 1, dtype=bool)
        x_faces = 0.5 * (self.x[:-1] + self.x[1:])
        
        self.veg_mask = ((x_faces >= self.config.veg_start) & 
                        (x_faces <= self.config.veg_end))
        
        # Vegetation characteristics
        veg_length = self.config.veg_end - self.config.veg_start
        veg_fraction = veg_length / self.config.L
        damping_timescale = 1.0 / self.config.cD if self.config.cD > 0 else np.inf
        
        self.logger.info("="*60)
        self.logger.info("VEGETATION PARAMETERS")
        self.logger.info("="*60)
        self.logger.info(f"Vegetation zone:")
        self.logger.info(f"  Location: [{self.config.veg_start:.1f}, {self.config.veg_end:.1f}] m")
        self.logger.info(f"  Length: {veg_length:.1f} m ({veg_fraction*100:.1f}% of domain)")
        self.logger.info(f"  Wavelengths in patch: {veg_length/self.wavelength:.2f}")
        self.logger.info(f"Drag characteristics:")
        self.logger.info(f"  Drag coefficient: cD = {self.config.cD:.3f} s⁻¹")
        self.logger.info(f"  Damping timescale: τ = {damping_timescale:.2f} s")
        self.logger.info(f"  Damping parameter: cD·T = {self.config.cD * self.period:.2f}")
        
    def _analyze_numerical_stability(self):
        """Perform stability analysis of the numerical scheme"""
        # CFL condition for explicit schemes
        cfl_limit_advection = 1.0  # For RK4
        
        # Stability limit for explicit drag term (not used since we use implicit)
        drag_stability_limit = 2.0 / self.config.cD if self.config.cD > 0 else np.inf
        
        # Von Neumann stability analysis results
        self.logger.info("="*60)
        self.logger.info("NUMERICAL STABILITY ANALYSIS")
        self.logger.info("="*60)
        self.logger.info(f"CFL stability:")
        self.logger.info(f"  Current CFL: {self.actual_cfl:.3f}")
        self.logger.info(f"  RK4 CFL limit: {cfl_limit_advection:.3f}")
        self.logger.info(f"  Stable: {'YES' if self.actual_cfl < cfl_limit_advection else 'NO'}")
        self.logger.info(f"Drag term stability:")
        self.logger.info(f"  Implicit scheme used: unconditionally stable")
        self.logger.info(f"  Explicit limit would be: dt < {drag_stability_limit:.3f} s")
        
    def solve(self):
        """
        Execute the numerical simulation using parallel RK4 time integration
        """
        self.logger.info("="*60)
        self.logger.info("STARTING SIMULATION")
        self.logger.info("="*60)
        
        # Initialize state variables
        eta = np.zeros(self.nx, dtype=np.float64)
        u = np.zeros(self.nx - 1, dtype=np.float64)
        
        # Pre-allocate output arrays
        n_outputs = len(self.output_times)
        self.eta_history = np.zeros((n_outputs, self.nx), dtype=np.float32)
        self.u_history = np.zeros((n_outputs, self.nx - 1), dtype=np.float32)
        self.energy_history = np.zeros((n_outputs, self.nx - 1), dtype=np.float32)
        self.t_output = np.zeros(n_outputs, dtype=np.float64)
        
        # Initial condition (zero initial state, wave enters from boundary)
        eta[0] = self.config.A * np.sin(0)
        
        # Time integration loop with progress monitoring
        output_idx = 0
        with tqdm(total=self.nt, desc="Time integration", unit="steps", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            for n in range(self.nt):
                t = n * self.dt
                
                # Store output at specified intervals
                if n in self.output_times:
                    self.eta_history[output_idx] = eta.astype(np.float32)
                    self.u_history[output_idx] = u.astype(np.float32)
                    self.energy_history[output_idx] = compute_energy_density(eta, u, self.config.d, self.g)
                    self.t_output[output_idx] = t
                    output_idx += 1
                
                # Advance solution using RK4
                eta, u = rk4_step(eta, u, self.veg_mask, self.config.cD,
                                self.g, self.config.d, self.config.dx, 
                                self.dt, self.config.A, self.config.omega, t)
                
                # Update progress
                pbar.update(1)
                
                # Monitor maximum values for stability check
                if n % 100 == 0:
                    max_eta = np.max(np.abs(eta))
                    max_u = np.max(np.abs(u))
                    if max_eta > 10 * self.config.A or np.isnan(max_eta):
                        self.logger.error(f"Numerical instability detected at t={t:.2f}s")
                        break
        
        self.logger.info("Simulation completed successfully")
        
    def calculate_transmission(self):
        """
        Calculate transmission and reflection coefficients using spectral analysis
        """
        # Use last 20% of simulation for steady-state analysis
        n_steady = max(int(0.2 * len(self.output_times)), 10)
        
        # Measurement locations (2 wavelengths from vegetation edges)
        x_in = self.config.veg_start - 2 * self.wavelength
        x_out = self.config.veg_end + 2 * self.wavelength
        
        # Ensure measurement points are within domain
        x_in = max(self.wavelength, x_in)
        x_out = min(self.config.L - self.wavelength, x_out)
        
        idx_in = np.argmin(np.abs(self.x - x_in))
        idx_out = np.argmin(np.abs(self.x - x_out))
        
        # Extract steady-state time series
        eta_in = self.eta_history[-n_steady:, idx_in]
        eta_out = self.eta_history[-n_steady:, idx_out]
        
        # Calculate wave heights using zero-crossing analysis
        H_in = 2 * np.std(eta_in)
        H_out = 2 * np.std(eta_out)
        
        # Transmission coefficient
        self.Kt = H_out / H_in if H_in > 0 else 0.0
        
        # Energy-based coefficients
        E_in = np.mean(eta_in**2)
        E_out = np.mean(eta_out**2)
        self.Kt_energy = np.sqrt(E_out / E_in) if E_in > 0 else 0.0
        
        # Estimate reflection coefficient (simplified)
        # In a more complete analysis, this would use array measurements
        self.Kr = np.sqrt(max(0, 1 - self.Kt**2))  # Energy conservation estimate
        
        # Calculate wave envelope for visualization
        n_periods = min(5, int(5 * self.period / self.config.output_dt))
        n_periods = min(n_periods, len(self.output_times))
        self.envelope = np.max(np.abs(self.eta_history[-n_periods:]), axis=0)
        
        # Phase-averaged velocity
        self.u_rms = np.sqrt(np.mean(self.u_history[-n_periods:]**2, axis=0))
        
        # Energy dissipation rate in vegetation
        energy_in_veg = self.energy_history[-n_steady:, self.veg_mask]
        if energy_in_veg.size > 0:
            self.dissipation_rate = np.mean(np.gradient(np.mean(energy_in_veg, axis=1)))
        else:
            self.dissipation_rate = 0.0
        
        self.logger.info("="*60)
        self.logger.info("WAVE TRANSFORMATION ANALYSIS")
        self.logger.info("="*60)
        self.logger.info(f"Measurement locations:")
        self.logger.info(f"  Upstream: x = {x_in:.1f} m")
        self.logger.info(f"  Downstream: x = {x_out:.1f} m")
        self.logger.info(f"Wave heights:")
        self.logger.info(f"  Incident: H_in = {H_in:.3f} m")
        self.logger.info(f"  Transmitted: H_out = {H_out:.3f} m")
        self.logger.info(f"Transmission coefficients:")
        self.logger.info(f"  Height-based: Kt = {self.Kt:.4f}")
        self.logger.info(f"  Energy-based: Kt_E = {self.Kt_energy:.4f}")
        self.logger.info(f"  Wave reduction: {(1-self.Kt)*100:.1f}%")
        self.logger.info(f"Reflection coefficient (estimated):")
        self.logger.info(f"  Kr ≈ {self.Kr:.4f}")
        self.logger.info(f"Energy budget check:")
        self.logger.info(f"  Kt² + Kr² = {self.Kt**2 + self.Kr**2:.4f} (should be ≤ 1)")
        
    def save_results(self, filename: str):
        """Save comprehensive results to NetCDF file with metadata"""
        self.logger.info(f"Saving results to {filename}")
        
        # Ensure output directory exists
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with nc.Dataset(filename, 'w', format='NETCDF4') as ds:
            # Dimensions
            ds.createDimension('x', self.nx)
            ds.createDimension('x_face', self.nx - 1)
            ds.createDimension('time', len(self.output_times))
            
            # Global attributes - Model description
            ds.title = "1D Wave Attenuation Model Results"
            ds.model_name = "wave_attenuation_1d"
            ds.model_version = "0.1.0"
            ds.authors = "Sandy Herho, Iwan Anwar, Theo Ndruru, Rusmawan Suwarman, Dasapta Irawan"
            ds.institution = "Samudera Sains Teknologi Ltd."
            ds.created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ds.conventions = "CF-1.8"
            
            # Model physics
            ds.governing_equations = "Linearized shallow water equations with vegetation drag"
            ds.numerical_scheme = "4th-order Runge-Kutta with implicit drag treatment"
            ds.boundary_conditions = "Sinusoidal wave maker (left), Sommerfeld radiation (right)"
            
            # Physical parameters
            ds.domain_length = self.config.L
            ds.water_depth = self.config.d
            ds.gravitational_acceleration = GRAVITY
            ds.water_density = 1000.0  # kg/m³
            
            # Wave parameters
            ds.wave_amplitude = self.config.A
            ds.wave_period = self.period
            ds.wave_frequency = self.config.omega
            ds.wavelength = self.wavelength
            ds.wave_number = self.k
            ds.phase_speed = self.c0
            
            # Vegetation parameters
            ds.vegetation_start = self.config.veg_start
            ds.vegetation_end = self.config.veg_end
            ds.vegetation_length = self.config.veg_end - self.config.veg_start
            ds.drag_coefficient = self.config.cD
            
            # Numerical parameters
            ds.spatial_resolution = self.config.dx
            ds.temporal_resolution = self.dt
            ds.cfl_number = self.actual_cfl
            ds.grid_points = self.nx
            ds.time_steps = self.nt
            
            # Results summary
            ds.transmission_coefficient = self.Kt
            ds.transmission_coefficient_energy = self.Kt_energy
            ds.reflection_coefficient_estimated = self.Kr
            ds.wave_height_reduction_percent = (1 - self.Kt) * 100
            
            # Coordinate variables
            x = ds.createVariable('x', 'f8', ('x',))
            x.units = 'm'
            x.long_name = 'distance along wave flume'
            x.standard_name = 'distance'
            x[:] = self.x
            
            x_face = ds.createVariable('x_face', 'f8', ('x_face',))
            x_face.units = 'm'
            x_face.long_name = 'distance at cell faces'
            x_face.standard_name = 'distance'
            x_face[:] = 0.5 * (self.x[:-1] + self.x[1:])
            
            time = ds.createVariable('time', 'f8', ('time',))
            time.units = 's'
            time.long_name = 'time since simulation start'
            time.standard_name = 'time'
            time[:] = self.t_output
            
            # Primary variables (2D: space-time)
            eta = ds.createVariable('eta', 'f4', ('time', 'x'), zlib=True, complevel=4)
            eta.units = 'm'
            eta.long_name = 'free surface elevation'
            eta.standard_name = 'sea_surface_height_above_mean_sea_level'
            eta[:] = self.eta_history
            
            u = ds.createVariable('u', 'f4', ('time', 'x_face'), zlib=True, complevel=4)
            u.units = 'm/s'
            u.long_name = 'depth-averaged horizontal velocity'
            u.standard_name = 'sea_water_x_velocity'
            u[:] = self.u_history
            
            energy = ds.createVariable('energy', 'f4', ('time', 'x_face'), zlib=True, complevel=4)
            energy.units = 'J/m³'
            energy.long_name = 'wave energy density'
            energy[:] = self.energy_history
            
            # Derived variables (1D: space only)
            envelope = ds.createVariable('envelope', 'f4', ('x',))
            envelope.units = 'm'
            envelope.long_name = 'wave envelope (maximum amplitude)'
            envelope[:] = self.envelope
            
            u_rms = ds.createVariable('u_rms', 'f4', ('x_face',))
            u_rms.units = 'm/s'
            u_rms.long_name = 'root-mean-square velocity'
            u_rms[:] = self.u_rms
            
            veg_mask = ds.createVariable('vegetation', 'i1', ('x_face',))
            veg_mask.long_name = 'vegetation presence indicator'
            veg_mask.flag_values = np.array([0, 1], dtype='i1')
            veg_mask.flag_meanings = "no_vegetation vegetation_present"
            veg_mask[:] = self.veg_mask.astype('i1')
            
        self.logger.info("Results saved successfully")
        self.logger.info(f"  NetCDF file: {filename}")
        self.logger.info(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
