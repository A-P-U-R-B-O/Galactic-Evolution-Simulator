"""
parameters.py

Advanced parameters and default configuration management for the Galactic Evolution Simulator.
Allows for easy extension and configuration of simulations, including physical constants,
galaxy model presets, and run-time simulation settings.
"""

from types import SimpleNamespace

# --- Physical constants (in simulation units or astrophysically meaningful units) ---
CONSTANTS = SimpleNamespace(
    G=4.302e-6,  # Gravitational constant [kpc * (km/s)^2 / Msun]
    year=3.154e7,  # seconds in a year
    kpc=3.086e16,  # meters in 1 kiloparsec
    Msun=1.989e30,  # kg in one solar mass
    c=3.0e5,       # Speed of light (km/s)
)

# --- Default simulation parameters ---
DEFAULTS = {
    "num_particles": 1000,
    "galaxy_type": "Spiral",  # "Spiral" or "Elliptical"
    "dark_matter_fraction": 0.8,
    "gas_fraction": 0.1,
    "disk_scale_length": 5.0,         # kpc
    "bulge_fraction": 0.2,            # for spiral
    "velocity_dispersion": 60.0,      # km/s
    "rotation_curve": "flat",         # "flat" or "keplerian"
    "star_formation_efficiency": 0.1, # dimensionless, 0-1
    "feedback_strength": 0.1,         # dimensionless, 0-1
    "SFR_threshold": 0.1,             # gas density threshold for star formation
    "cooling_rate": 1e-3,             # per timestep
    "timesteps": 200,
    "dt": 0.1,                        # time increment (simulation units)
    "integrate_method": "leapfrog",   # "leapfrog" or "euler"
    "softening": 0.05,                # softening parameter for gravity
    "seed": None,                     # random seed
}

# --- Preset galaxy models for quick setup ---
GALAXY_PRESETS = {
    "Milky Way": {
        "num_particles": 1500,
        "galaxy_type": "Spiral",
        "dark_matter_fraction": 0.85,
        "gas_fraction": 0.12,
        "disk_scale_length": 6.5,
        "bulge_fraction": 0.15,
        "velocity_dispersion": 80.0,
        "rotation_curve": "flat",
    },
    "Elliptical Giant": {
        "num_particles": 1200,
        "galaxy_type": "Elliptical",
        "dark_matter_fraction": 0.7,
        "gas_fraction": 0.05,
        "disk_scale_length": 10.0,
        "velocity_dispersion": 140.0,
        "rotation_curve": "keplerian",
    },
    "Dwarf Galaxy": {
        "num_particles": 400,
        "galaxy_type": "Spiral",
        "dark_matter_fraction": 0.9,
        "gas_fraction": 0.2,
        "disk_scale_length": 1.5,
        "bulge_fraction": 0.05,
        "velocity_dispersion": 20.0,
        "rotation_curve": "flat",
    }
}

def get_parameters(preset=None, **overrides):
    """
    Returns a dict of simulation parameters, optionally using a preset and applying overrides.
    """
    params = DEFAULTS.copy()
    if preset and preset in GALAXY_PRESETS:
        params.update(GALAXY_PRESETS[preset])
    params.update(overrides)
    return params

def parameter_summary(params):
    """
    Returns a formatted string summary of simulation parameters for display/logging.
    """
    lines = []
    for key in sorted(params.keys()):
        lines.append(f"{key:>24}: {params[key]}")
    return "\n".join(lines)
