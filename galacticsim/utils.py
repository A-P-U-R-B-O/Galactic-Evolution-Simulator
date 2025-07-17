import numpy as np

def get_masses(num_particles, dark_matter_fraction=0.8, gas_fraction=0.1, 
               star_mass=1.0, dm_mass=5.0, gas_mass=0.5):
    """
    Assign masses to particles according to their component: stars, dark matter, gas.
    Returns masses array and types array.
    """
    n_stars = int(num_particles * (1 - dark_matter_fraction - gas_fraction))
    n_dm = int(num_particles * dark_matter_fraction)
    n_gas = num_particles - n_stars - n_dm

    masses = np.concatenate([
        np.full(n_stars, star_mass),
        np.full(n_dm, dm_mass),
        np.full(n_gas, gas_mass)
    ])
    types = np.array(
        ["star"] * n_stars + ["dm"] * n_dm + ["gas"] * n_gas
    )
    return masses, types

def kinetic_energy(velocities, masses):
    """
    Calculate total kinetic energy of a system.
    """
    return 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))

def potential_energy(positions, masses, G=4.302e-6, softening=0.05):
    """
    Calculate total gravitational potential energy of the system.
    """
    N = len(masses)
    pot = 0.0
    for i in range(N):
        diff = positions[i] - positions[i+1:]
        dist = np.sqrt(np.sum(diff**2, axis=1) + softening**2)
        pot -= np.sum(G * masses[i] * masses[i+1:] / dist)
    return pot

def random_seed(seed=None):
    """
    Sets numpy's random seed if provided, for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

def center_of_mass(positions, masses):
    """
    Calculate the center of mass of the system.
    """
    total_mass = np.sum(masses)
    return np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass

def shift_to_center_of_mass_frame(positions, velocities, masses):
    """
    Shift positions and velocities so that the center of mass is at (0, 0) and the net momentum is zero.
    """
    com = center_of_mass(positions, masses)
    positions -= com
    total_mass = np.sum(masses)
    momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    v_com = momentum / total_mass
    velocities -= v_com
    return positions, velocities

def compute_radial_profile(positions, masses, nbins=30, rmax=None):
    """
    Compute radial mass profile for the system.
    Returns bin centers, enclosed mass, and surface density.
    """
    r = np.linalg.norm(positions, axis=1)
    if rmax is None:
        rmax = np.max(r) * 1.1
    bins = np.linspace(0, rmax, nbins+1)
    enclosed_mass = np.zeros(nbins)
    surface_density = np.zeros(nbins)
    for i in range(nbins):
        mask = (r >= bins[i]) & (r < bins[i+1])
        enclosed_mass[i] = np.sum(masses[r < bins[i+1]])
        area = np.pi * (bins[i+1]**2 - bins[i]**2)
        surface_density[i] = np.sum(masses[mask]) / area if area > 0 else 0
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers, enclosed_mass, surface_density

def velocity_dispersion(velocities, types=None, component="star"):
    """
    Calculate velocity dispersion for a given component.
    """
    if types is not None:
        mask = types == component
        vels = velocities[mask]
    else:
        vels = velocities
    return np.std(vels, axis=0)

def angular_momentum(positions, velocities, masses):
    """
    Calculate the total angular momentum vector of the system.
    """
    # Assume 2D: L_z = sum m * (x*v_y - y*v_x)
    L = np.sum(masses * (positions[:, 0] * velocities[:, 1] - positions[:, 1] * velocities[:, 0]))
    return L

def print_diagnostics(step, history, verbose=True):
    """
    Print or log summary diagnostics for a simulation step.
    """
    if not verbose:
        return
    print(
        f"Step {step:4d}: "
        f"Star Mass={history['star_mass'][-1]:.2e} "
        f"Gas Mass={history['gas_mass'][-1]:.2e} "
        f"Tgas={history['gas_temperature'][-1]:.1f}K "
        f"E_kin={history['kinetic_energy'][-1]:.2e} "
        f"E_pot={history['potential_energy'][-1]:.2e}"
    )
