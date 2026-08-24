from setuptools import find_packages, setup

setup(
    name="mle-eval",
    version="1.0.0",
    description="Extensible Benchmark Framework for Autonomous AI ML Engineering Agents",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
    ],
    entry_points={
        "console_scripts": [
            "mle-eval=src.cli:main",
        ],
    },
)