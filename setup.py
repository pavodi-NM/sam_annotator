from setuptools import setup, find_packages

setup(
    name="sam-annotator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
        "pandas>=1.2.0"
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=20.8b1',
            'isort>=5.0'
        ]
    },
    author="Pavodi NDOYI MANIAMFU",
    author_email="pavodi.mani@fingervision.biz",
    description="A general-purpose image annotation tool based on SAM",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sam-annotator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)