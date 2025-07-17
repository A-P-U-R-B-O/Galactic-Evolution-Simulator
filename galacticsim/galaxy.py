import numpy as np

class Galaxy:
    """
    Represents a galaxy with stars, gas, and dark matter components.
    Supports initialization of spiral, elliptical, and custom galaxies,
    and holds particle data and ISM properties for simulation.
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
        self.num_particles = num_particles
        self.galaxy_type = galaxy_type
        self.dark_matter_fraction = dark_matter_fraction
        self.gas_fraction = gas_fraction
        self.disk_scale_length = disk_scale_length
        self.bulge_fraction = bulge_fraction
        self.velocity_dispersion = velocity_dispersion
        self.rotation_curve = rotation_curve
        self.seed = seed
        self.custom_init = custom_init

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Compute number of each component
        self.n_stars = int(num_particles * (1 - dark_matter_fraction - gas_fraction))
        self.n_gas = int(num_particles * gas_fraction)
        self.n_dm = num_particles - self.n_stars - self.n_gas

        # Initialize particles
        if custom_init is not None:
            self.positions, self.velocities, self.masses, self.types = custom_init(self)
        elif galaxy_type == "Spiral":
            self.positions, self.velocities, self.masses, self.types = self._init_spiral()
        elif galaxy_type == "Elliptical":
            self.positions, self.velocities, self.masses, self.types = self._init_elliptical()
        else:
            raise ValueError(f"Unsupported galaxy_type: {galaxy_type}")

        # ISM properties (toy model)
        self.gas_mass = np.sum(self.masses[self.types == "gas"]) if self.n_gas > 0 else 0.0
        self.gas_density = self.gas_mass / (np.pi * (np.max(np.linalg.norm(self.positions, axis=1))**2))
        self.gas_temperature = 1e4  # Kelvin

    def _init_spiral(self):
        """Spiral galaxy: disk + bulge + dark matter halo + gas."""
        positions = []
        velocities = []
        masses = []
        types = []

        # Disk stars
        n_disk = int(self.n_stars * (1 - self.bulge_fraction))
        disk_pos, disk_vel = self._sample_exponential_disk(n_disk)
        positions.append(disk_pos)
        velocities.append(disk_vel)
        masses.append(np.ones(n_disk))
        types += ["star"] * n_disk

        # Bulge stars (spherical)
        n_bulge = self.n_stars - n_disk
        bulge_pos = np.random.normal(0, self.disk_scale_length * 0.3, (n_bulge, 2))
        bulge_vel = np.random.normal(0, self.velocity_dispersion, (n_bulge, 2))
        positions.append(bulge_pos)
        velocities.append(bulge_vel)
        masses.append(np.ones(n_bulge))
        types += ["star"] * n_bulge

        # Gas particles (disk)
        if self.n_gas > 0:
            gas_pos, gas_vel = self._sample_exponential_disk(self.n_gas, scale=1.2*self.disk_scale_length)
            positions.append(gas_pos)
            velocities.append(gas_vel)
            masses.append(np.full(self.n_gas, 0.5))  # Each gas cloud is less massive
            types += ["gas"] * self.n_gas

        # Dark matter halo (isothermal sphere)
        if self.n_dm > 0:
            dm_radius = np.random.normal(self.disk_scale_length * 3, self.disk_scale_length, self.n_dm)
            dm_theta = np.random.uniform(0, 2 * np.pi, self.n_dm)
            dm_x = dm_radius * np.cos(dm_theta)
            dm_y = dm_radius * np.sin(dm_theta)
            dm_pos = np.column_stack([dm_x, dm_y])
            dm_vel = np.random.normal(0, self.velocity_dispersion, (self.n_dm, 2))
            positions.append(dm_pos)
            velocities.append(dm_vel)
            masses.append(np.ones(self.n_dm) * 5.0)  # DM particles are more massive
            types += ["dm"] * self.n_dm

        # Stack arrays
        positions = np.vstack(positions)
        velocities = np.vstack(velocities)
        masses = np.concatenate(masses)
        types = np.array(types)

        # Assign disk rotation
        self._apply_disk_rotation(positions, velocities, types)

        return positions, velocities, masses, types

    def _init_elliptical(self):
        """Elliptical galaxy: single spheroidal distribution, no disk."""
        positions = np.random.normal(0, self.disk_scale_length, (self.n_stars, 2))
        velocities = np.random.normal(0, self.velocity_dispersion, (self.n_stars, 2))
        masses = np.ones(self.n_stars)
        types = np.array(["star"] * self.n_stars)

        # Gas (if present)
        if self.n_gas > 0:
            gas_pos = np.random.normal(0, self.disk_scale_length, (self.n_gas, 2))
            gas_vel = np.random.normal(0, self.velocity_dispersion, (self.n_gas, 2))
            positions = np.vstack([positions, gas_pos])
            velocities = np.vstack([velocities, gas_vel])
            masses = np.concatenate([masses, np.full(self.n_gas, 0.5)])
            types = np.concatenate([types, np.array(["gas"] * self.n_gas)])

        # Dark matter halo (if present)
        if self.n_dm > 0:
            dm_radius = np.random.normal(self.disk_scale_length * 3, self.disk_scale_length, self.n_dm)
            dm_theta = np.random.uniform(0, 2 * np.pi, self.n_dm)
            dm_x = dm_radius * np.cos(dm_theta)
            dm_y = dm_radius * np.sin(dm_theta)
            dm_pos = np.column_stack([dm_x, dm_y])
            dm_vel = np.random.normal(0, self.velocity_dispersion, (self.n_dm, 2))
            positions = np.vstack([positions, dm_pos])
            velocities = np.vstack([velocities, dm_vel])
            masses = np.concatenate([masses, np.ones(self.n_dm) * 5.0])
            types = np.concatenate([types, np.array(["dm"] * self.n_dm)])

        return positions, velocities, masses, types

    def _sample_exponential_disk(self, n, scale=None):
        """
        Sample positions and velocities for an exponential disk.
        """
        if scale is None:
            scale = self.disk_scale_length
        # Radial positions: exponential distribution
        r = np.random.exponential(scale, n)
        theta = np.random.uniform(0, 2 * np.pi, n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pos = np.column_stack([x, y])

        # Set initial velocities (rotation + random)
        v_circ = self._rotation_curve(r)
        vx = -v_circ * np.sin(theta) + np.random.normal(0, self.velocity_dispersion*0.1, n)
        vy = v_circ * np.cos(theta) + np.random.normal(0, self.velocity_dispersion*0.1, n)
        vel = np.column_stack([vx, vy])
        return pos, vel

    def _apply_disk_rotation(self, positions, velocities, types):
        """
        Adjust velocities so that disk stars and gas rotate in the disk plane.
        """
        is_disk = (types == "star") | (types == "gas")
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

    def _rotation_curve(self, r):
        """
        Calculate rotation curve at radius r.
        """
        if self.rotation_curve == "flat":
            v0 = 220.0  # km/s
            return np.ones_like(r) * v0
        elif self.rotation_curve == "keplerian":
            v0 = 220.0
            r0 = self.disk_scale_length
            return v0 * np.sqrt(r0 / np.maximum(r, 0.1))
        else:
            raise ValueError(f"Unknown rotation_curve: {self.rotation_curve}")
