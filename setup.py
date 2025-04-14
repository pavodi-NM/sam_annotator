from setuptools import setup, find_packages
import os
import subprocess
import sys

# Generate PyPI-friendly README if the script exists
readme_path = 'README.md'
pypi_readme_path = 'README_PYPI.md'

# Try to generate PyPI README if the script exists
try:
    if os.path.exists('prepare_pypi.py'):
        print("Generating PyPI-friendly README...")
        subprocess.check_call([sys.executable, 'prepare_pypi.py'])
        if os.path.exists(pypi_readme_path):
            readme_path = pypi_readme_path
except Exception as e:
    print(f"Warning: Could not generate PyPI README: {e}")
    print("Using original README.md for PyPI")

# Read the contents of README file for long_description
with open(readme_path, encoding='utf-8') as f:
    long_description = f.read()

# Get version from pyproject.toml for consistency
version = "0.1.0"  # Default version
try:
    with open('pyproject.toml', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('version'):
                version = line.split('=')[1].strip().replace('"', '').replace("'", '')
                break
except:
    pass  # Use default version if pyproject.toml can't be read

setup(
    name="sam_annotator",  # Use the exact same name as in pyproject.toml
    version=version,
    description="A powerful tool for semi-automatic image annotation based on Meta AI's Segment Anything Model (SAM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SAM Annotator Team",
    author_email="pavodi.mani@fingervision.biz",
    url="https://github.com/pavodi-nm/sam_annotator",
    packages=find_packages(exclude=['test*', 'documentation', 'examples']),
    include_package_data=True,
    install_requires=[
        "torch>=1.10.0",
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "segment-anything>=1.0",
        "PyYAML>=6.0",
    ],
    entry_points={
        'console_scripts': [
            'sam-annotator=sam_annotator.main:main',  # Keep the hyphen for the command name
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
)