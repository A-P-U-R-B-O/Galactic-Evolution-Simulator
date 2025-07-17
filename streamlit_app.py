import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from galacticsim.parameters import get_parameters, GALAXY_PRESETS, parameter_summary
from galacticsim.galaxy import Galaxy
from galacticsim.core import run_simulation, SimulationConfig
from galacticsim.visualization import (
    plot_galaxy,
    plot_density_map,
    plot_time_series,
    animate_galaxy,
)
from galacticsim.utils import (
    shift_to_center_of_mass_frame,
    compute_radial_profile,
    velocity_dispersion,
    get_masses,
)

st.set_page_config(
    page_title="Galactic Evolution Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌌 Galactic Evolution Simulator")
st.markdown(
    """
    **Explore galaxy formation and evolution!**  
    Adjust parameters, simulate, and visualize the cosmic dance of stars, gas, and dark matter.
    - Advanced N-body simulation with star formation and feedback
    - Real-time visualization, radial profiles, and more
    - Share and download your results!
    """
)

# --- Sidebar: Parameter Controls ---
with st.sidebar:
    st.header("Simulation Setup")

    preset = st.selectbox(
        "Preset Galaxy Model",
        ["Custom"] + list(GALAXY_PRESETS.keys()),
        index=0,
    )

    if preset != "Custom":
        params = get_parameters(preset=preset)
        st.success(f"Loaded preset: {preset}")
    else:
        params = get_parameters()
    
    # Particle number
    num_particles = st.slider(
        "Number of Particles", 100, 3000, int(params["num_particles"]), step=100
    )
    galaxy_type = st.selectbox(
        "Galaxy Type", ["Spiral", "Elliptical"], index=0 if params["galaxy_type"] == "Spiral" else 1
    )
    dark_matter_fraction = st.slider(
        "Dark Matter Fraction", 0.0, 0.99, float(params["dark_matter_fraction"]), step=0.01
    )
    gas_fraction = st.slider(
        "Gas Fraction", 0.0, 0.5, float(params.get("gas_fraction", 0.1)), step=0.01
    )
    disk_scale_length = st.number_input(
        "Disk Scale Length (kpc)", 0.5, 20.0, float(params.get("disk_scale_length", 5.0)), step=0.1
    )
    bulge_fraction = st.slider(
        "Bulge Fraction (Spiral)", 0.0, 0.5, float(params.get("bulge_fraction", 0.2)), step=0.01
    )
    velocity_dispersion = st.number_input(
        "Velocity Dispersion (km/s)", 0.0, 300.0, float(params.get("velocity_dispersion", 60.0)), step=1.0
    )
    rotation_curve = st.selectbox(
        "Rotation Curve", ["flat", "keplerian"], index=0 if params.get("rotation_curve", "flat") == "flat" else 1
    )
    seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=int(params.get("seed") or 0), step=1)

    # Simulation options
    st.header("Physical & Numerical Options")
    integrate_method = st.selectbox(
        "Integrator", ["leapfrog", "euler"], index=0 if params.get("integrate_method", "leapfrog") == "leapfrog" else 1
    )
    timesteps = st.slider("Number of Timesteps", 10, 1000, int(params.get("timesteps", 200)), step=10)
    dt = st.number_input("dt (time increment)", 0.001, 2.0, float(params.get("dt", 0.1)), step=0.001)
    softening = st.number_input("Softening Parameter", 0.001, 1.0, float(params.get("softening", 0.05)), step=0.001)
    star_formation_efficiency = st.slider(
        "Star Formation Efficiency", 0.0, 1.0, float(params.get("star_formation_efficiency", 0.1)), step=0.01
    )
    feedback_strength = st.slider(
        "Feedback Strength", 0.0, 1.0, float(params.get("feedback_strength", 0.1)), step=0.01
    )
    SFR_threshold = st.number_input(
        "SFR Gas Density Threshold", 0.0, 10.0, float(params.get("SFR_threshold", 0.1)), step=0.01
    )
    cooling_rate = st.number_input(
        "Cooling Rate", 0.0, 1.0, float(params.get("cooling_rate", 1e-3)), step=1e-3, format="%.4f"
    )
    verbose = st.checkbox("Verbose Output (console/log)", value=False)

    st.markdown("----")
    st.markdown("**Tip:** Use a preset as a starting point, then customize parameters.")

# --- Main Simulation Controls and Visualization ---

if "history" not in st.session_state:
    st.session_state["history"] = None

st.header("1️⃣ Galaxy Initialization")
if st.button("Generate Initial Galaxy", type="primary"):
    galaxy = Galaxy(
        num_particles=num_particles,
        galaxy_type=galaxy_type,
        dark_matter_fraction=dark_matter_fraction,
        gas_fraction=gas_fraction,
        disk_scale_length=disk_scale_length,
        bulge_fraction=bulge_fraction,
        velocity_dispersion=velocity_dispersion,
        rotation_curve=rotation_curve,
        seed=seed,
    )
    # Center-of-mass frame
    galaxy.positions, galaxy.velocities = shift_to_center_of_mass_frame(
        galaxy.positions, galaxy.velocities, galaxy.masses
    )
    st.session_state["galaxy"] = galaxy
    st.session_state["history"] = None
    st.success("Galaxy initialized! Explore the initial state below.")

# Show initial galaxy plots if available
galaxy = st.session_state.get("galaxy", None)
if galaxy is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Particle View")
        fig1 = plot_galaxy(galaxy.positions, types=galaxy.types, title="Initial Galaxy State")
        st.pyplot(fig1)
    with col2:
        st.subheader("Density Map")
        fig2 = plot_density_map(galaxy.positions, title="Initial Density Map")
        st.pyplot(fig2)

    # Show parameter summary
    st.expander("Parameter Summary", expanded=False).markdown(
        f"```yaml\n{parameter_summary(locals())}\n```"
    )

st.header("2️⃣ Run Simulation")
if st.button("Run Simulation", type="primary", disabled=(galaxy is None)):
    config = SimulationConfig(
        steps=timesteps,
        dt=dt,
        softening=softening,
        integrate_method=integrate_method,
        star_formation=True,
        star_formation_efficiency=star_formation_efficiency,  # <-- FIXED: pass to config
        feedback=True,
        feedback_efficiency=feedback_strength,
        SFR_threshold=SFR_threshold,
        cooling=True,
        cooling_rate=cooling_rate,
        verbose=verbose,
    )
    # Run the simulation
    history = run_simulation(
        galaxy.positions,
        galaxy.velocities,
        galaxy.masses,
        config,
        gas_mass=galaxy.gas_mass,
        gas_density=galaxy.gas_density,
        gas_temperature=galaxy.gas_temperature,
    )
    st.session_state["history"] = history
    st.success("Simulation complete! View results and analysis below.")

history = st.session_state.get("history", None)
if history is not None:
    st.header("3️⃣ Simulation Results & Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Final State", "Evolution Animation", "Physical Plots", "Profiles/Diagnostics"]
    )
    with tab1:
        st.subheader("Final Galaxy State")
        idx = -1  # Last frame
        fig3 = plot_galaxy(history["positions"][idx], types=galaxy.types, title="Final State")
        st.pyplot(fig3)
        fig4 = plot_density_map(history["positions"][idx], title="Final Density Map")
        st.pyplot(fig4)

    with tab2:
        st.subheader("Animated Evolution")
        frame = st.slider(
            "Frame", 0, history["positions"].shape[0] - 1, 0, key="anim_frame"
        )
        fig_anim = plot_galaxy(
            history["positions"][frame],
            types=galaxy.types,
            title=f"Galaxy Evolution (Frame {frame})",
        )
        st.pyplot(fig_anim)
        st.caption("Use the slider to scrub through simulation frames.")

    with tab3:
        st.subheader("Physical Properties Over Time")
        fig5 = plot_time_series(
            history["time"],
            star_mass=history["star_mass"],
            gas_mass=history["gas_mass"],
            gas_temperature=history["gas_temperature"],
            kinetic_energy=history["kinetic_energy"],
            potential_energy=history["potential_energy"],
        )
        st.pyplot(fig5)

    with tab4:
        st.subheader("Radial Mass Profile (Final State)")
        radii, enclosed_mass, surf_density = compute_radial_profile(
            history["positions"][-1], galaxy.masses, nbins=30
        )
        plt.figure(figsize=(6, 4))
        plt.plot(radii, enclosed_mass, label="Enclosed Mass")
        plt.xlabel("Radius [kpc]")
        plt.ylabel("Enclosed Mass [a.u.]")
        plt.title("Radial Mass Profile")
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()

        st.subheader("Velocity Dispersion (Final State)")
        disp = velocity_dispersion(history["velocities"][-1], types=galaxy.types, component="star")
        st.write(f"Stellar velocity dispersion: {disp}")

    # Download results
    st.markdown("---")
    st.subheader("Download Simulation Data")
    import io
    import pickle

    data_bytes = io.BytesIO()
    pickle.dump(history, data_bytes)
    st.download_button(
        label="Download Simulation Data (.pkl)",
        data=data_bytes.getvalue(),
        file_name="galaxy_simulation_history.pkl",
        mime="application/octet-stream",
    )

    # Export final snapshot as PNG
    img_bytes = io.BytesIO()
    fig3.savefig(img_bytes, format="png")
    st.download_button(
        label="Download Final State Image (.png)",
        data=img_bytes.getvalue(),
        file_name="galaxy_final_state.png",
        mime="image/png",
    )

st.markdown("---")
st.caption("Galactic Evolution Simulator | Made with ❤️ for Open Science. [GitHub Repo](https://github.com/yourusername/galactic-evolution-simulator)")
