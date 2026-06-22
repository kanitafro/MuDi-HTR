"""Setup configuration for MuDi-HTR."""

from setuptools import find_packages, setup

setup(
    name="MuDi-HTR",
    version="0.1.0",
    description="Multi-Modal Digital Handwriting Text Recognition",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "torchvision",
        "opencv-python",
        "pillow",
        "numpy",
        "pandas",
        "streamlit",
        "streamlit-drawable-canvas",
        "datasketch",
        "PyInkML",
        "wandb",
        "tensorboard",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "jupyter",
        "pytest",
        "pyyaml",
        "datasets",
    ],
)
