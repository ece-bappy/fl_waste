"""
Setup script for Medical Waste Sorting System.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="medical-waste-sorting",
    version="1.0.0",
    author="Medical Waste Sorting Team",
    description="Federated Learning Medical Waste Sorting System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "torch-audio>=2.0.0",
            "tensorrt>=8.6.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "medical-waste-sorting=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="medical waste, federated learning, robotics, computer vision, YOLO",
    project_urls={
        "Bug Reports": "https://github.com/your-org/medical-waste-sorting/issues",
        "Source": "https://github.com/your-org/medical-waste-sorting",
        "Documentation": "https://github.com/your-org/medical-waste-sorting/wiki",
    },
)
