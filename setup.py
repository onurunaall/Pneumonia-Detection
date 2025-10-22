"""
Pneumonia Detection Package Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="pneumonia-detection",
    version="1.0.0",
    description="Deep learning-based pneumonia detection from chest X-rays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "api": [
            "fastapi>=0.95.0",
            "uvicorn[standard]>=0.21.0",
            "python-multipart>=0.0.6",
        ],
        "viz": [
            "plotly>=5.0.0",
            "gradio>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pneumonia-train=scripts.train:main",
            "pneumonia-predict=scripts.predict:main",
            "pneumonia-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
