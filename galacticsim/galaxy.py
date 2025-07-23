import numpy as np

class Galaxy:
    """
    Represents a galaxy with stars, gas, and dark matter components.
    Supports initialization of spiral, elliptical, and custom galaxies,
    and holds particle data and ISM properties for simulation.
    Now supports full 3D initialization.
    """

    def __init__(
        self,
        num_particles,
        galaxy_type="Spiral",
        dark_matter_fraction=0.8,
        gas_fraction=0.1,
        disk_scale_length=5.0,
        bulge_fraction=0.2,
        velocity_dispersion=60.0,
        rotation_curve="flat",
        seed=None,
        custom_init=None,
    ):
        """
        Parameters:
            num_particles: Total number of particles (stars+dark matter+gas)
            galaxy_type: "Spiral", "Elliptical", or "Custom"
            dark_matter_fraction: Fraction of mass in dark matter
            gas_fraction: Fraction of baryonic mass in gas
            disk_scale_length: Disk scale length (for spiral)
            bulge_fraction: Fraction of stellar mass in bulge (for spiral)
            velocity_dispersion: Initial random velocity dispersion (km/s)
            rotation_curve: "flat" or "keplerian"
            seed: (Optional) Random seed for reproducibility
            custom_init: (Optional) Function to initialize custom galaxies
        """
        self.num_particles = int(num_particles)
        self.galaxy_type = galaxy_type
        self.dark_matter_fraction = float(dark_matter_fraction)
        self.gas_fraction = float(gas_fraction)
        self.disk_scale_length = float(disk_scale_length)
        self.bulge_fraction = float(bulge_fraction)
        self.velocity_dispersion = float(velocity_dispersion)
        self.rotation_curve = rotation_curve
        self.seed = seed
        self.custom_init = custom_init

        # Validate fractions
        for frac_name, frac_val in [
            ("dark_matter_fraction", self.dark_matter_fraction),
            ("gas_fraction", self.gas_fraction),
            ("bulge_fraction", self.bulge_fraction),
        ]:
            if not (0.0 <= frac_val <= 1.0):
                raise ValueError(f"{frac_name} must be between 0 and 1. Got {frac_val}.")

        total_fraction = self.dark_matter_fraction + self.gas_fraction
        if total_fraction > 1.0:
            raise ValueError(
                f"Sum of dark_matter_fraction ({self.dark_matter_fraction}) and gas_fraction ({self.gas_fraction}) exceeds 1. "
                "There must be some stellar mass (stars = 1 - dark_matter_fraction - gas_fraction)."
            )

        # Set random seed for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)

        # Compute number of each component and ensure non-negative, integer values
        n_stars_raw = self.num_particles * (1 - self.dark_matter_fraction - self.gas_fraction)
        self.n_stars = max(int(round(n_stars_raw)), 0)
        n_gas_raw = self.num_particles * self.gas_fraction
        self.n_gas = max(int(round(n_gas_raw)), 0)
        self.n_dm = max(self.num_particles - self.n_stars - self.n_gas, 0)

        if self.n_stars + self.n_gas + self.n_dm != self.num_particles:
            # Fix rounding mismatches by adding/subtracting from DM
            diff = self.num_particles - (self.n_stars + self.n_gas + self.n_dm)
            self.n_dm = max(self.n_dm + diff, 0)

        # Initialize particles
        if self.custom_init is not None:
            self.positions, self.velocities, self.masses, self.types = self.custom_init(self)
        elif self.galaxy_type == "Spiral":
            self.positions, self.velocities, self.masses, self.types = self._init_spiral()
        elif self.galaxy_type == "Elliptical":
            self.positions, self.velocities, self.masses, self.types = self._init_elliptical()
        else:
            raise ValueError(f"Unsupported galaxy_type: {self.galaxy_type}")

        # ISM properties (toy model)
        if np.any(self.types == "gas"):
            self.gas_mass = np.sum(self.masses[self.types == "gas"])
        else:
            self.gas_mass = 0.0
        pos_norm = np.linalg.norm(self.positions, axis=1)
        if pos_norm.size > 0:
            self.gas_density = self.gas_mass / (4 / 3 * np.pi * (np.max(pos_norm) ** 3))
        else:
            self.gas_density = 0.0
        self.gas_temperature = 1e4  # Kelvin

    def _init_spiral(self):
        """Spiral galaxy: disk + bulge + dark matter halo + gas. Now 3D."""
        positions = []
        velocities = []
        masses = []
        types = []

        # Disk stars (thin disk, z ~ Gaussian)
        n_disk_raw = self.n_stars * (1 - self.bulge_fraction)
        n_disk = max(int(round(n_disk_raw)), 0)
        if n_disk > 0:
            disk_pos, disk_vel = self._sample_exponential_disk_3d(n_disk, z_std=0.1 * self.disk_scale_length)
            positions.append(disk_pos)
            velocities.append(disk_vel)
            masses.append(np.ones(n_disk))
            types += ["star"] * n_disk

        # Bulge stars (spherical)
        n_bulge = self.n_stars - n_disk
        if n_bulge > 0:
            bulge_pos = np.random.normal(0, self.disk_scale_length * 0.3, (n_bulge, 3))
            bulge_vel = np.random.normal(0, self.velocity_dispersion, (n_bulge, 3))
            positions.append(bulge_pos)
            velocities.append(bulge_vel)
            masses.append(np.ones(n_bulge))
            types += ["star"] * n_bulge

        # Gas particles (disk, thin)
        if self.n_gas > 0:
            gas_pos, gas_vel = self._sample_exponential_disk_3d(
                self.n_gas,
                scale=1.2 * self.disk_scale_length,
                z_std=0.15 * self.disk_scale_length,
            )
            positions.append(gas_pos)
            velocities.append(gas_vel)
            masses.append(np.full(self.n_gas, 0.5))  # Each gas cloud is less massive
            types += ["gas"] * self.n_gas

        # Dark matter halo (spherical isothermal sphere)
        if self.n_dm > 0:
            dm_radius = np.abs(np.random.normal(self.disk_scale_length * 3, self.disk_scale_length, self.n_dm))
            phi = np.random.uniform(0, 2 * np.pi, self.n_dm)
            costheta = np.random.uniform(-1, 1, self.n_dm)
            theta = np.arccos(costheta)
            dm_x = dm_radius * np.sin(theta) * np.cos(phi)
            dm_y = dm_radius * np.sin(theta) * np.sin(phi)
            dm_z = dm_radius * np.cos(theta)
            dm_pos = np.column_stack([dm_x, dm_y, dm_z])
            dm_vel = np.random.normal(0, self.velocity_dispersion, (self.n_dm, 3))
            positions.append(dm_pos)
            velocities.append(dm_vel)
            masses.append(np.ones(self.n_dm) * 5.0)  # DM particles are more massive
            types += ["dm"] * self.n_dm

        # If all lists are empty, return empty arrays
        if len(positions) == 0:
            positions = np.empty((0, 3))
            velocities = np.empty((0, 3))
            masses = np.array([])
            types = np.array([], dtype=str)
            return positions, velocities, masses, types

        # Stack arrays
        positions = np.vstack(positions)
        velocities = np.vstack(velocities)
        masses = np.concatenate(masses)
        types = np.array(types)

        # Assign disk rotation (in-plane only)
        self._apply_disk_rotation_3d(positions, velocities, types)

        return positions, velocities, masses, types

    def _init_elliptical(self):
        """Elliptical galaxy: single spheroidal distribution, no disk. Now 3D."""
        positions = []
        velocities = []
        masses = []
        types = []

        if self.n_stars > 0:
            star_pos = np.random.normal(0, self.disk_scale_length, (self.n_stars, 3))
            star_vel = np.random.normal(0, self.velocity_dispersion, (self.n_stars, 3))
            positions.append(star_pos)
            velocities.append(star_vel)
            masses.append(np.ones(self.n_stars))
            types += ["star"] * self.n_stars

        # Gas (if present)
        if self.n_gas > 0:
            gas_pos = np.random.normal(0, self.disk_scale_length, (self.n_gas, 3))
            gas_vel = np.random.normal(0, self.velocity_dispersion, (self.n_gas, 3))
            positions.append(gas_pos)
            velocities.append(gas_vel)
            masses.append(np.full(self.n_gas, 0.5))
            types += ["gas"] * self.n_gas

        # Dark matter halo (if present)
        if self.n_dm > 0:
            dm_radius = np.abs(np.random.normal(self.disk_scale_length * 3, self.disk_scale_length, self.n_dm))
            phi = np.random.uniform(0, 2 * np.pi, self.n_dm)
            costheta = np.random.uniform(-1, 1, self.n_dm)
            theta = np.arccos(costheta)
            dm_x = dm_radius * np.sin(theta) * np.cos(phi)
            dm_y = dm_radius * np.sin(theta) * np.sin(phi)
            dm_z = dm_radius * np.cos(theta)
            dm_pos = np.column_stack([dm_x, dm_y, dm_z])
            dm_vel = np.random.normal(0, self.velocity_dispersion, (self.n_dm, 3))
            positions.append(dm_pos)
            velocities.append(dm_vel)
            masses.append(np.ones(self.n_dm) * 5.0)
            types += ["dm"] * self.n_dm

        # If all lists are empty, return empty arrays
        if len(positions) == 0:
            positions = np.empty((0, 3))
            velocities = np.empty((0, 3))
            masses = np.array([])
            types = np.array([], dtype=str)
            return positions, velocities, masses, types

        # Stack arrays
        positions = np.vstack(positions)
        velocities = np.vstack(velocities)
        masses = np.concatenate(masses)
        types = np.array(types)

        return positions, velocities, masses, types

    def _sample_exponential_disk_3d(self, n, scale=None, z_std=None):
        """
        Sample positions and velocities for a 3D thin exponential disk.
        z_std: standard deviation of z (vertical thickness)
        """
        n = max(int(round(n)), 0)
        if n <= 0:
            # Return empty arrays if no particles requested
            return np.empty((0, 3)), np.empty((0, 3))
        if scale is None:
            scale = self.disk_scale_length
        if z_std is None:
            z_std = 0.1 * scale  # thin disk

        # Radial positions: exponential distribution
        r = np.random.exponential(scale, n)
        theta = np.random.uniform(0, 2 * np.pi, n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.normal(0, z_std, n)
        pos = np.column_stack([x, y, z])

        # Set initial velocities (rotation in xy + vertical random)
        v_circ = self._rotation_curve(r)
        vx = -v_circ * np.sin(theta) + np.random.normal(0, self.velocity_dispersion * 0.1, n)
        vy = v_circ * np.cos(theta) + np.random.normal(0, self.velocity_dispersion * 0.1, n)
        vz = np.random.normal(0, self.velocity_dispersion * 0.1, n)
        vel = np.column_stack([vx, vy, vz])
        return pos, vel

    def _apply_disk_rotation_3d(self, positions, velocities, types):
        """
        Adjust velocities so that disk stars and gas rotate in the disk plane (xy).
        z-velocity is left unchanged.
        """
        is_disk = (types == "star") | (types == "gas")
        if np.sum(is_disk) == 0:
            return
        disk_positions = positions[is_disk]
        x = disk_positions[:, 0]
        y = disk_positions[:, 1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        v_circ = self._rotation_curve(r)
        vx = -v_circ * np.sin(theta)
        vy = v_circ * np.cos(theta)
        velocities[is_disk, 0] = vx
        velocities[is_disk, 1] = vy
        # velocities[is_disk, 2] remains
