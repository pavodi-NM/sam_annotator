from setuptools import setup, find_packages
import os

# Read the contents of README.md for long_description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="sam_annotator",
    version="1.0.0",
    description="A powerful tool for semi-automatic image annotation based on Meta AI's Segment Anything Model (SAM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SAM Annotator Team",
    author_email="pacodi.mani@fingervision.biz",
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
            'sam-annotator=sam_annotator.main:main',
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