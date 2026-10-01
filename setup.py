from pathlib import Path

from setuptools import find_packages, setup

requirements = [
    line.split("#")[0].strip()
    for line in Path(__file__).with_name("requirements.txt").read_text().splitlines()
    if line.split("#")[0].strip()
]

setup(
    name="gquery",
    version="1.0.0",
    description="Constraint Group Query: Pareto and weighted solvers, experiment runners and data preparation",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=requirements,
)
