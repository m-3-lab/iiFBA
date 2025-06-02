from setuptools import setup, find_packages

setup(
    name="iiFBA_COBRApy_extension",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "cobra>=0.29.1",     # COBRApy, adjust version as needed
        "numpy>=1.23.3",
        "pandas>=2.2.3",
    ],
    author="Multiscale Metabolic Modeling Lab",
    description="iiFBA Functions for COBRApy models",
)