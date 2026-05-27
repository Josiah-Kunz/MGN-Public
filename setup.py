from setuptools import setup, find_packages

setup(
    name="meshgraphnet",
    version="0.1.0",
    author="Josiah D. Kunz, Kamal Choudhary",
    author_email="josiah.kunz@ic.edu",
    description="Graph neural network surrogates for FEA on arbitrary geometries",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Josiah-Kunz/meshgraphnet",
    packages=find_packages(),
    install_requires=[
        "gmsh",
        "meshio",
        "pint",
        "torch",
        "torch-geometric",
        "scikit-learn",
        "numpy",
        "pandas",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
