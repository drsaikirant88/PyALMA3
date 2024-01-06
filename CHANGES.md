# VERSION CONTROL

## v1.0: 6 Jan 2024

Public release of PyALMA3 along with submission of manuscript.
Moved change log to separate file
Changed params.toml file to package-specific name paramsPyALMA3.toml
Updated PlanetProfile example to latest version (v2.3.19)
Moved obsolete alma_build_model.py file to old/ directory to preserve legacy code
Reconfigured file naming and directory structure to support packaging
Implemented default config file stored in package and copied to PWD
Updated jupyter notebooks to work with the above changes
Modified PlanetProfile layer finder to use phase indices

## v0.1: 20 May 2022

This is the initial conversion of FORTRAN to Python.
FMZM is obsolete so not required in Python Input files
parameters should be set in the "params" file
Added functionality to read PP and BM models.
Removed LU Decomposition and Linear Equation solver subroutines from FORTRAN. Instead I am using modules from Scipy and Numpy.
