"""
pyALMA3 (Python plAnetary Love nuMbers cALculator)
pythonized version of ALMA 3
------------------------------------------------------------------------------

Author: Flavio Petricca, Saikiran Tharimena, Daniele Melini, Giorgio Spada, Amirhossein Bagheri, Marshall J. Styczinski, Steven D. Vance 
Copyright (C) 2024 the authors
"""
from alma import CopyOnlyIfNeeded, _defaultConfig, _userConfig

if __name__ == '__main__':
    CopyOnlyIfNeeded(_defaultConfig, _userConfig)
