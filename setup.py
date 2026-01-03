"""
Setup script for Hybrid-GCS package.

Install with: pip install -e .
Install with dev tools: pip install -e ".[dev]"
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hybrid-gcs",
    version="0.1.0",
    author="Harsh Mulodhia",
    description="Hybrid planning combining GCS and Deep RL for robotics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HarshMulodhia/hybrid-gcs",
    packages=find_packages(),
    python_requires=">=3.9",
    
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=1.10.0",
        "pydantic>=1.8.0",
        "PyYAML>=5.4.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.7b0",
            "pylint>=2.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
        ],
        "sim": [
            "pybullet>=3.1.0",
        ],
        "solvers": [
            "scs>=3.0.0",
        ],
        "rl": [
            "tensorboard>=2.8.0",
            "matplotlib>=3.4.0",
        ],
        "full": [
            "pybullet>=3.1.0",
            "scs>=3.0.0",
            "tensorboard>=2.8.0",
            "matplotlib>=3.4.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "hybrid-gcs-train=hybrid_gcs.cli.train:main",
            "hybrid-gcs-eval=hybrid_gcs.cli.evaluate:main",
            "hybrid-gcs-vis=hybrid_gcs.cli.visualize:main",
        ],
    },
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
