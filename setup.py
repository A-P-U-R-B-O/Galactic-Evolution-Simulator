from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="galacticsim",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An advanced, interactive Galactic Evolution Simulator with Streamlit frontend.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/galactic-evolution-simulator",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/galactic-evolution-simulator/issues",
        "Documentation": "https://github.com/yourusername/galactic-evolution-simulator#readme",
    },
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "streamlit",
        "plotly",
    ],
    extras_require={
        "dev": ["pytest", "jupyter"],
        "docs": ["sphinx", "nbsphinx"]
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.ipynb"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Framework :: Streamlit",
    ],
    keywords="galaxy simulator astronomy astrophysics streamlit nbody",
)
