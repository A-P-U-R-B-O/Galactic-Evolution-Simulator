# 🌌 Galactic Evolution Simulator

**A modular, open-source simulator for galactic evolution, star formation, and feedback, with interactive Streamlit and Jupyter interfaces.**

---

## 🚀 Overview

The **Galactic Evolution Simulator** enables you to simulate and visualize the formation and dynamical evolution of galaxies—including stars, gas, and dark matter—using advanced N-body physics with options for star formation, feedback, and cooling. This project is designed for research, education, and exploration, and comes with a modern Python codebase, rich documentation, and interactive visualization.

- **N-body gravitational dynamics** (Leapfrog/Euler integrators)
- **Star formation and feedback** (toy models)
- **Gas cooling and evolution**
- **Preset or custom galaxies** (spiral, elliptical, dwarf, etc.)
- **Animated and interactive visualizations** (matplotlib, Plotly, Streamlit)
- **Jupyter and Streamlit interfaces**
- **Open source, extensible, and reproducible**

---

## 🗂️ Project Structure

```
galactic-evolution-simulator/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── setup.py
├── requirements.txt
│
├── galacticsim/
│   ├── __init__.py
│   ├── core.py                # Core simulation logic (N-body, star formation, etc.)
│   ├── galaxy.py              # Galaxy and component classes
│   ├── parameters.py          # Parameter management and defaults
│   ├── visualization.py       # Visualization utilities (matplotlib/plotly)
│   └── utils.py               # Helper functions
│
├── streamlit_app.py           # Streamlit web app entry point
│
├── examples/
│   └── example_notebook.ipynb # Jupyter notebook: advanced usage/demo
│
└── tests/
    ├── __init__.py
    └── test_core.py
```

---

## ⚡ Quickstart

### 1. Install

**Requirements:** Python 3.8+, `numpy`, `matplotlib`, `streamlit`, `plotly`.

```bash
git clone https://github.com/A-P-U-R-B-O/galactic-evolution-simulator.git
cd galactic-evolution-simulator
pip install -e .
```

Or just:

```bash
pip install -r requirements.txt
```

### 2. Run the Interactive App

```bash
streamlit run streamlit_app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Try the Jupyter Example

```bash
cd examples
jupyter notebook example_notebook.ipynb
```

---

## 🧑‍💻 Usage: In Code

```python
from galacticsim.parameters import get_parameters
from galacticsim.galaxy import Galaxy
from galacticsim.core import run_simulation, SimulationConfig

# 1. Get parameters and initialize a galaxy
params = get_parameters(preset="Milky Way", num_particles=1000)
galaxy = Galaxy(**params)

# 2. Run the simulation
config = SimulationConfig(steps=200, dt=0.1, ...)
history = run_simulation(
    galaxy.positions, galaxy.velocities, galaxy.masses, config
)

# 3. Visualize (matplotlib, or see visualization.py for more)
from galacticsim.visualization import plot_galaxy
plot_galaxy(history["positions"][-1], types=galaxy.types)
```

---

## 🎨 Features

- **Galaxy Models:** Spiral, elliptical, and custom, with tunable gas, bulge, dark matter, and velocity profiles.
- **Physical Processes:** Star formation (Schmidt law), feedback (toy supernovae), ISM cooling.
- **Visualization:** 2D/3D scatter, density maps, time evolution, interactive animation (matplotlib/Plotly/Streamlit).
- **Parameter Management:** Presets for Milky Way, dwarfs, etc., or full custom control.
- **Analysis Tools:** Radial mass profiles, velocity dispersion, angular momentum, energy conservation.
- **Export:** Download simulation data and imagery directly from the Streamlit app.
- **Testing:** Automated tests in `/tests` for CI and reliability.

---

## 🧩 Extending

- Add new galaxy models or physics by editing `galacticsim/galaxy.py` and `core.py`.
- Add new visualizations in `galacticsim/visualization.py`.
- Define new parameter presets in `galacticsim/parameters.py`.

---

## 📚 Documentation

- **Example notebook:** `examples/example_notebook.ipynb`
- **Code docs:** Docstrings throughout the codebase.
- **API:** See each module for usage, or run help in Python.

---

## 🛠️ Development

- To run tests: `pytest tests/`
- To add features: Fork, branch, PR welcome!
- To deploy on Render or similar, see `render.yaml`.

---

## 🤝 Contributing

- Issues and PRs are welcome!
- Please include tests for new features.
- See [CONTRIBUTING.md](CONTRIBUTING.md) if available.

---

## 📜 License

MIT License (see `LICENSE`).

---

## 👨‍🔬 Credits

Created by [Your Name](mailto:your.email@example.com).  
Inspired by research and teaching needs in computational astrophysics.

---

## 🌠 Acknowledgments

- Python, NumPy, Matplotlib, Streamlit, Plotly, and the open astrophysics community.

---

## 🪐 FAQ

**Q:** Is this code suitable for professional galactic astrophysics research?  
**A:** It's designed for education and rapid prototyping. For high-precision research, more sophisticated codes (e.g., GADGET, AREPO) are recommended.

**Q:** Can I use this for teaching?  
**A:** Absolutely! The Streamlit app is especially suited for interactive demos.

**Q:** How do I cite this project?  
**A:** See [CITATION.cff](CITATION.cff) if available, or cite the GitHub repo URL.

---

## 🌍 Links

- [Project Home](https://github.com/yourusername/galactic-evolution-simulator)
- [Streamlit App Demo](https://galactic-evolution-simulator.onrender.com) *(if deployed)*
- [Issues](https://github.com/yourusername/galactic-evolution-simulator/issues)

---

Enjoy exploring the universe!
