import numpy as np

# Optional: Uncomment the next two lines if you install numba for speedup!
# from numba import njit

G = 4.302e-6  # Gravitational constant in kpc * (km/s)^2 / Msun

class SimulationConfig:
    def __init__(
        self,
        steps=200,
        dt=0.1,
        softening=0.05,
        integrate_method="leapfrog",
        use_gpu=False,
        star_formation=True,
        star_formation_efficiency=0.1,
        feedback=True,
        feedback_efficiency=0.1,
        SFR_threshold=0.1,
        cooling=True,
        cooling_rate=1e-3,
        verbose=False
    ):
        self.steps = steps
        self.dt = dt
        self.softening = softening
        self.integrate_method = integrate_method
        self.use_gpu = use_gpu
        self.star_formation = star_formation
        self.star_formation_efficiency = star_formation_efficiency
        self.feedback = feedback
        self.feedback_efficiency = feedback_efficiency
        self.SFR_threshold = SFR_threshold
        self.cooling = cooling
        self.cooling_rate = cooling_rate
        self.verbose = verbose

# Optionally add @njit for big speedup if you have numba installed!
# @njit
def compute_accelerations(positions, masses, softening):
    """
    Compute gravitational acceleration for all particles using softening.
    O(N^2) direct-sum.
    """
    N = positions.shape[0]
    acc = np.zeros_like(positions)
    for i in range(N):
        diff = positions - positions[i]
        dist2 = np.sum(diff**2, axis=1) + softening**2
        inv_dist3 = dist2 ** -1.5
        inv_dist3[i] = 0  # ignore self-interaction
        acc[i] = G * np.sum((diff.T * masses * inv_dist3).T, axis=0)
    return acc

def leapfrog_step(positions, velocities, masses, dt, softening):
    """
    Leapfrog integrator (kick-drift-kick).
    """
    acc = compute_accelerations(positions, masses, softening)
    velocities_half = velocities + 0.5 * acc * dt
    positions_new = positions + velocities_half * dt
    acc_new = compute_accelerations(positions_new, masses, softening)
    velocities_new = velocities_half + 0.5 * acc_new * dt
    return positions_new, velocities_new

def euler_step(positions, velocities, masses, dt, softening):
    acc = compute_accelerations(positions, masses, softening)
    velocities_new = velocities + acc * dt
    positions_new = positions + velocities_new * dt
    return positions_new, velocities_new

def star_formation(gas_mass, gas_density, threshold, efficiency, dt):
    """
    Simple Schmidt-Kennicutt law: SFR ∝ gas_density^1.4 if above threshold.
    Returns mass of new stars formed.
    """
    if gas_density > threshold:
        sfr = efficiency * gas_density ** 1.4
        new_stars = sfr * dt
        new_stars = min(new_stars, gas_mass)
        return new_stars
    return 0.0

def apply_feedback(star_mass, feedback_efficiency):
    """
    Returns energy injected back into the ISM from young stars (e.g., supernovae).
    """
    feedback_energy = feedback_efficiency * star_mass  # toy model
    return feedback_energy

def apply_cooling(gas_temperature, cooling_rate, dt):
    """
    Simple exponential cooling.
    """
    return gas_temperature * np.exp(-cooling_rate * dt)

def compute_kinetic_energy(velocities, masses):
    return 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))

def compute_potential_energy(positions, masses, softening):
    """
    Compute total potential energy (pairwise sum).
    """
    N = positions.shape[0]
    pot = 0.0
    for i in range(N):
        diff = positions[i] - positions[i+1:]
        dist = np.sqrt(np.sum(diff**2, axis=1) + softening**2)
        pot -= np.sum(G * masses[i] * masses[i+1:] / dist)
    return pot

def run_simulation(
    init_positions,
    init_velocities,
    masses,
    config: SimulationConfig,
    gas_mass=None,
    gas_density=None,
    gas_temperature=None,
    callback=None,
    progress_callback=None
):
    """
    Runs the full simulation with advanced features:
    - Leapfrog or Euler integration
    - Star formation and feedback
    - ISM cooling
    - Energy tracking

    Optional:
    - callback: called each step with (step, positions, velocities, masses, star_mass, gas_mass, gas_density, gas_temperature)
    - progress_callback: called with (step, steps) for UI progress (e.g. Streamlit st.progress)
    """
    positions = init_positions.copy()
    velocities = init_velocities.copy()
    masses = masses.copy()
    N = positions.shape[0]
    steps = config.steps
    dt = config.dt

    # For history
    history = {
        "positions": [],
        "velocities": [],
        "masses": [],
        "star_mass": [],
        "gas_mass": [],
        "gas_density": [],
        "gas_temperature": [],
        "kinetic_energy": [],
        "potential_energy": [],
        "time": [],
    }

    # ISM properties, if enabled
    star_mass = np.sum(masses)
    if gas_mass is None:
        gas_mass = 0.1 * star_mass  # initial gas mass
    if gas_density is None:
        gas_density = gas_mass / (np.pi * (np.max(np.linalg.norm(positions, axis=1))**2))
    if gas_temperature is None:
        gas_temperature = 1e4  # K

    for step in range(steps):
        # Integrate positions and velocities
        if config.integrate_method == "leapfrog":
            positions, velocities = leapfrog_step(
                positions, velocities, masses, dt, config.softening
            )
        else:
            positions, velocities = euler_step(
                positions, velocities, masses, dt, config.softening
            )

        # Star Formation
        new_stars = 0.0
        if config.star_formation and gas_mass > 0:
            new_stars = star_formation(
                gas_mass, gas_density, config.SFR_threshold, config.star_formation_efficiency, dt
            )
            gas_mass -= new_stars
            star_mass += new_stars

        # Feedback
        feedback_energy = 0.0
        if config.feedback and new_stars > 0:
            feedback_energy = apply_feedback(new_stars, config.feedback_efficiency)
            gas_temperature += feedback_energy  # toy: increase ISM temperature

        # Cooling
        if config.cooling:
            gas_temperature = apply_cooling(gas_temperature, config.cooling_rate, dt)

        # Update gas density (toy: assume constant area)
        gas_density = gas_mass / (np.pi * (np.max(np.linalg.norm(positions, axis=1))**2))

        # Energies
        kinetic = compute_kinetic_energy(velocities, masses)
        potential = compute_potential_energy(positions, masses, config.softening)

        # Save history
        history["positions"].append(positions.copy())
        history["velocities"].append(velocities.copy())
        history["masses"].append(masses.copy())
        history["star_mass"].append(star_mass)
        history["gas_mass"].append(gas_mass)
        history["gas_density"].append(gas_density)
        history["gas_temperature"].append(gas_temperature)
        history["kinetic_energy"].append(kinetic)
        history["potential_energy"].append(potential)
        history["time"].append(step * dt)

        if config.verbose and step % 10 == 0:
            print(
                f"Step {step:4d}: stars={star_mass:.2e} gas={gas_mass:.2e} "
                f"Tgas={gas_temperature:.1f}K Ekin={kinetic:.2e} Epot={potential:.2e}"
            )

        # Optional callback for e.g. visualization or external logging
        if callback is not None:
            callback(step, positions, velocities, masses, star_mass, gas_mass, gas_density, gas_temperature)

        # Progress bar support
        if progress_callback is not None:
            progress_callback(step, steps)

    # Convert lists to arrays
    for key in history:
        history[key] = np.array(history[key])

    return history
