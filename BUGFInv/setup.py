from setuptools import setup, find_packages

setup(
    name="bugfinv",
    version="1.0.0",
    author="Yingtian Liu",
    author_email="yingtianliu06@outlook.com",
    description="Bayesian Uncertainty-Aware Gradient Fusion for 3D Prestack Three-Parameter Inversion",
    url="https://github.com/Yingtian-Liu/BUGFInv",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
)