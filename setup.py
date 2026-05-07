from setuptools import setup, find_packages

setup(
    name="gono-optimizer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.23.0",
        "matplotlib>=3.6.0",
        "scikit-learn>=1.0.0",
    ],
    author="Victor Daniel Gera",
    description="GONO: Gradient-Oriented Norm-Adaptive Optimizer",
    python_requires=">=3.8",
)
