from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="pneumonia-detection",
    version="1.0.0",
    description="Deep learning-based pneumonia detection from chest X-rays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Onur Ünal",
    author_email="onur.unal492@gmail.com",
    url="https://github.com/onurunaall/pneumonia-detection",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "scripts", "docs"]),
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
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "pandas>=1.3.0,<3.0.0",
        "scikit-learn>=1.0.0,<2.0.0",
        "tensorflow>=2.10.0,<3.0.0",
        "pydicom>=2.3.0,<3.0.0",
        "opencv-python>=4.6.0,<5.0.0",
        "scikit-image>=0.19.0,<1.0.0",
        "albumentations>=1.3.0,<2.0.0",
        "Pillow>=9.0.0,<11.0.0",
        "matplotlib>=3.5.0,<4.0.0",
        "seaborn>=0.11.0,<1.0.0",
        "tqdm>=4.64.0,<5.0.0",
        "pyyaml>=6.0,<7.0",
        "python-dotenv>=0.20.0,<2.0.0",
        "click>=8.0.0,<9.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.11.0",
        ],
        "api": [
            "fastapi>=0.95.0",
            "uvicorn[standard]>=0.21.0",
            "python-multipart>=0.0.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "pneumonia-train=scripts.train:main",
            "pneumonia-predict=scripts.predict:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
