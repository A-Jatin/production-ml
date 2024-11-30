from setuptools import setup, find_packages

setup(
    name="synthetic_data",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "dask>=2021.6.2",
        "pytest>=7.0.0",
        "psutil>=5.8.0",
    ],
) 