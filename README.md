pyALMA3 is a pythonized version of ALMA 3.

pyALMA3 is a free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 or any later version.

pyALMA3 is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for more details.

GNU GPL: http://www.gnu.org/licenses/

Author: Saikiran Tharimena, Daniele Melini, Giorgio Spada, Steve Vance, Marshall Styczinski

Copyright (C) 2022 Saikiran Tharimena

VERSION CONTROL

v0.1: 20 May 2022

This is the initial conversion of FORTRAN to Python
FMZM is obsolete so not required in Python Input files
parameters should be set in the "params" file
Added functionality to read PP and BM models.
Removed LU Decomposition and Linear Equation solver subroutines from FORTRAN. Instead I am using modules from Scipy and Numpy.