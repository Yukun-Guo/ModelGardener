"""
Setup configuration for ModelGardener CLI package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    requirements = (this_directory / "requirements.txt").read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="modelgardener",
    version="2.0.0",
    author="ModelGardener Team",
    author_email="contact@modelgardener.com",
    description="A command-line interface for deep learning model training with TensorFlow/Keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yukun-Guo/ModelGardener",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'mgd=modelgardener.cli:run_cli',
        ],
    },
    include_package_data=True,
    package_data={
        "modelgardener": ["*.py"],
    },
    zip_safe=False,
    keywords="deep-learning, machine-learning, tensorflow, keras, cli, training",
)
