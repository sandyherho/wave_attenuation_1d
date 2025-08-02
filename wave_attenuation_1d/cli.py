"""
Command-line interface for wave_attenuation_1d

Authors: Sandy Herho <sandy.herho@email.ucr.edu>, Iwan Anwar, Theo Ndruru, Rusmawan Suwarman, Dasapta Irawan
Date: 08/01/2025
License: MIT
"""

import argparse
import configparser
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

from .solver import Config, WaveSolver

__version__ = "0.1.0"


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging configuration"""
    # Ensure logs directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('wave_attenuation_1d')
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def load_config(config_file: Path) -> Config:
    """Load configuration from file"""
    parser = configparser.ConfigParser()
    parser.read(config_file)
    
    return Config(
        L=float(parser['DOMAIN']['L']),
        d=float(parser['DOMAIN']['d']),
        dx=float(parser['DOMAIN']['dx']),
        T=float(parser['DOMAIN']['T']),
        A=float(parser['WAVE']['A']),
        omega=float(parser['WAVE']['omega']),
        veg_start=float(parser['VEGETATION']['start']),
        veg_end=float(parser['VEGETATION']['end']),
        cD=float(parser['VEGETATION']['cD']),
        cfl_target=float(parser['NUMERICAL']['cfl_target']),
        output_dt=float(parser['NUMERICAL']['output_dt'])
    )


def write_summary_log(log_file: Path, config: Config, solver: WaveSolver):
    """Write simulation summary to log file"""
    with open(log_file, 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write("SIMULATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Domain: L = {config.L} m\n")
        f.write(f"  Water depth: d = {config.d} m\n")
        f.write(f"  Wave: A = {config.A} m, T = {solver.period:.1f} s\n")
        f.write(f"  Wavelength: λ = {solver.wavelength:.1f} m\n")
        f.write(f"  Vegetation: [{config.veg_start}, {config.veg_end}] m\n")
        f.write(f"  Drag coefficient: cD = {config.cD} m²/s\n\n")
        
        f.write("Numerical parameters:\n")
        f.write(f"  Grid points: nx = {solver.nx}\n")
        f.write(f"  Grid spacing: dx = {config.dx} m\n")
        f.write(f"  Time step: dt = {solver.dt:.4f} s\n")
        f.write(f"  CFL number: {solver.c0 * solver.dt / config.dx:.3f}\n")
        f.write(f"  Total steps: {solver.nt}\n\n")
        
        f.write("Results:\n")
        f.write(f"  Transmission coefficient: Kt = {solver.Kt:.4f}\n")
        f.write(f"  Wave height reduction: {(1-solver.Kt)*100:.1f}%\n")
        f.write(f"  Maximum envelope at x=0: {solver.envelope[0]:.3f} m\n")
        f.write(f"  Maximum envelope at x=L: {solver.envelope[-1]:.3f} m\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Simple 1D Wave Attenuation Model for Coastal Vegetation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wave-attenuation-1d configs/sparse_config.txt
  wave-attenuation-1d dense_config.txt
  
Authors: Sandy Herho, Iwan Anwar, Theo Ndruru
License: MIT (2025)
        """
    )
    
    parser.add_argument(
        'config',
        type=str,
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'wave-attenuation-1d {__version__}'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file '{config_path}' not found")
        sys.exit(1)
    
    # Get the directory where the config file is located
    config_dir = config_path.parent
    
    # Create output directories at the same level as configs directory
    # If running from project root, this creates outputs/ and logs/ alongside configs/
    if config_dir.name == 'configs':
        base_dir = config_dir.parent
    else:
        base_dir = Path.cwd()
    
    output_dir = base_dir / 'outputs'
    log_dir = base_dir / 'logs'
    
    # Create directories if they don't exist
    output_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    # Generate output filenames WITHOUT timestamp
    base_name = config_path.stem
    output_file = output_dir / f"{base_name}.nc"
    log_file = log_dir / f"{base_name}.log"
    
    # Setup logging
    logger = setup_logging(log_file)
    
    logger.info("="*50)
    logger.info("Simple 1D Wave Attenuation Model")
    logger.info("="*50)
    logger.info(f"Version: {__version__}")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(config_path)
        
        # Create and run solver
        solver = WaveSolver(config, logger)
        solver.solve()
        solver.calculate_transmission()
        
        # Save results
        solver.save_results(str(output_file))
        
        # Write summary to log
        write_summary_log(log_file, config, solver)
        
        logger.info("")
        logger.info("Simulation completed successfully!")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Log saved to: {log_file}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
