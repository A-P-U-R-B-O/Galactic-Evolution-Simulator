import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import animation

try:
    import plotly.graph_objs as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def plot_galaxy(
    positions,
    types=None,
    title="Galaxy Snapshot",
    show_legend=True,
    figsize=(7, 7),
    color_map=None,
    ax=None,
):
    """
    Plot a snapshot of galaxy particles by type (stars, gas, dark matter).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if types is None:
        # Assume all stars
        ax.scatter(positions[:, 0], positions[:, 1], s=2, c="yellow", alpha=0.7, label="Particles")
    else:
        color_map = color_map or {"star": "orange", "gas": "blue", "dm": "purple"}
        unique_types = np.unique(types)
        for t in unique_types:
            mask = types == t
            ax.scatter(
                positions[mask, 0],
                positions[mask, 1],
                s=2 if t != "dm" else 1,
                c=color_map.get(t, "gray"),
                alpha=0.7,
                label=t.capitalize(),
            )
    ax.set_aspect("equal")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(title)
    if show_legend and types is not None:
        ax.legend()
    return fig

def plot_density_map(
    positions,
    bins=200,
    extent=None,
    cmap="inferno",
    title="Projected Density Map",
    ax=None,
    lognorm=True,
):
    """
    Plot a 2D density (surface brightness) map from particle positions.
    """
    if extent is None:
        R = np.max(np.linalg.norm(positions, axis=1)) * 1.2
        extent = [-R, R, -R, R]
    H, xedges, yedges = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=bins, range=[[extent[0], extent[1]], [extent[2], extent[3]]]
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    norm = LogNorm(vmin=1, vmax=H.max()) if lognorm else None
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="Projected Density")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(title)
    return fig

def plot_time_series(
    time,
    star_mass=None,
    gas_mass=None,
    gas_temperature=None,
    kinetic_energy=None,
    potential_energy=None,
    figsize=(10, 6),
    ax=None,
):
    """
    Plot evolution of physical properties over simulation time.
    """
    if ax is None:
        fig, ax = plt.subplots(2, 2, figsize=figsize)
    else:
        fig = ax[0, 0].figure

    # Star/Gas Mass
    if star_mass is not None and gas_mass is not None:
        ax[0, 0].plot(time, star_mass, label="Stars", color="gold")
        ax[0, 0].plot(time, gas_mass, label="Gas", color="blue")
        ax[0, 0].set_ylabel("Mass [a.u.]")
        ax[0, 0].set_xlabel("Time [sim units]")
        ax[0, 0].set_title("Mass Evolution")
        ax[0, 0].legend()
    elif star_mass is not None:
        ax[0, 0].plot(time, star_mass, label="Stars", color="gold")
        ax[0, 0].set_ylabel("Star Mass [a.u.]")

    # Gas Temperature
    if gas_temperature is not None:
        ax[0, 1].plot(time, gas_temperature, color="red")
        ax[0, 1].set_ylabel("Gas Temperature [K]")
        ax[0, 1].set_xlabel("Time [sim units]")
        ax[0, 1].set_title("Gas Temperature Evolution")

    # Energies
    if kinetic_energy is not None and potential_energy is not None:
        ax[1, 0].plot(time, kinetic_energy, label="Kinetic", color="green")
        ax[1, 0].plot(time, potential_energy, label="Potential", color="purple")
        ax[1, 0].plot(time, kinetic_energy + potential_energy, label="Total", color="black")
        ax[1, 0].set_ylabel("Energy [a.u.]")
        ax[1, 0].set_xlabel("Time [sim units]")
        ax[1, 0].set_title("Energy Evolution")
        ax[1, 0].legend()

    # Blank or custom
    ax[1, 1].axis("off")
    fig.tight_layout()
    return fig

def animate_galaxy(history, types=None, interval=50, save_path=None, density_map=False):
    """
    Animate the evolution of a galaxy simulation.
    If density_map=True, shows a 2D density histogram; else scatter plot.
    """
    pos = history["positions"]
    n_frames = pos.shape[0]
    fig, ax = plt.subplots(figsize=(7, 7))

    if density_map:
        def update_density(frame):
            ax.clear()
            plot_density_map(pos[frame], ax=ax, lognorm=True, title=f"Density Map (Frame {frame})")
            return ax

        anim = animation.FuncAnimation(fig, update_density, frames=n_frames, interval=interval, blit=False)
    else:
        # Particle types for color coding
        if types is not None:
            unique_types = np.unique(types)
            color_map = {"star": "orange", "gas": "blue", "dm": "purple"}
            def update_scatter(frame):
                ax.clear()
                for t in unique_types:
                    mask = types == t
                    ax.scatter(
                        pos[frame][mask, 0],
                        pos[frame][mask, 1],
                        s=2 if t != "dm" else 1,
                        c=color_map.get(t, "gray"),
                        alpha=0.7,
                        label=t.capitalize(),
                    )
                ax.set_aspect("equal")
                ax.set_xlabel("x [kpc]")
                ax.set_ylabel("y [kpc]")
                ax.set_title(f"Galaxy Evolution (Frame {frame})")
                ax.legend()
                return ax
        else:
            def update_scatter(frame):
                ax.clear()
                ax.scatter(pos[frame][:, 0], pos[frame][:, 1], s=2, c="orange", alpha=0.7)
                ax.set_aspect("equal")
                ax.set_xlabel("x [kpc]")
                ax.set_ylabel("y [kpc]")
                ax.set_title(f"Galaxy Evolution (Frame {frame})")
                return ax

        anim = animation.FuncAnimation(fig, update_scatter, frames=n_frames, interval=interval, blit=False)

    if save_path is not None:
        anim.save(save_path, writer="ffmpeg")

    return anim

def plotly_galaxy_3d(positions, types=None, title="Galaxy 3D", color_map=None):
    """
    Generate an interactive 3D scatter plot using Plotly.
    Requires plotly.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is not installed. Please install plotly to use this function.")
    if positions.shape[1] != 3:
        raise ValueError("3D Plot requires positions with 3 columns (x, y, z).")

    color_map = color_map or {"star": "orange", "gas": "blue", "dm": "purple"}
    data = []
    if types is None:
        data.append(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers",
                marker=dict(size=2, color="orange", opacity=0.7),
                name="Particles",
            )
        )
    else:
        unique_types = np.unique(types)
        for t in unique_types:
            mask = types == t
            data.append(
                go.Scatter3d(
                    x=positions[mask, 0],
                    y=positions[mask, 1],
                    z=positions[mask, 2],
                    mode="markers",
                    marker=dict(size=2 if t != "dm" else 1, color=color_map.get(t, "gray"), opacity=0.7),
                    name=t.capitalize(),
                )
            )
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title="x [kpc]",
            yaxis_title="y [kpc]",
            zaxis_title="z [kpc]",
        ),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=30),
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

def save_snapshot(positions, types, filename="galaxy_snapshot.png"):
    """
    Save a snapshot of the galaxy as an image file.
    """
    fig = plot_galaxy(positions, types, title="Galaxy Snapshot")
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_density_map(positions, filename="density_map.png"):
    """
    Save a density map as an image.
    """
    fig = plot_density_map(positions)
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
