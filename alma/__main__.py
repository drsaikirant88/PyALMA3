"""
pyALMA3 (Python plAnetary Love nuMbers cALculator)
pythonized version of ALMA 3
------------------------------------------------------------------------------

Author: Saikiran Tharimena, Daniele Melini, Giorgio Spada, Steven D. Vance, Marshall J. Styczinski
Copyright (C) 2024 the authors
"""
from alma import CopyOnlyIfNeeded, _defaultConfig, _userConfig

if __name__ == '__main__':
    CopyOnlyIfNeeded(_defaultConfig, _userConfig)
