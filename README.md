# Python plAnetary Love nuMbers cALculator
PyALMA3 is a pythonized version of ALMA 3. This software package calculates tidal Love numbers given an appropriate model of interior structure for a planetary body.

## Installation

The recommended installation method is with pip:

`pip install PyALMA3`

After installing with pip, copy over the default config file to your working directory with

`python -m alma`

Then to compute Love numbers, call PyALMA3 functions using e.g.

```
from alma import love_numbers

h, l, k = love_numbers(n, t, 
                       alma_params['mode'], 
                       alma_params['function'],
                       alma_params['tau'],
                       model_params,
                       alma_params['type'],
                       alma_params['gorder'],
                       verbose = alma_params['verbose'],
                       parallel = alma_params['parallel'])
```

See [PP_example.ipynb](https://github.com/drsaikirant88/PyALMA3/blob/main/PP_example.ipynb) for a complete example application.

## Disclaimers and copyright

PyALMA3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 or any later version.

PyALMA3 is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for more details.

GNU GPL: http://www.gnu.org/licenses/

Authors: Saikiran Tharimena, Daniele Melini, Giorgio Spada, Steven D. Vance, Marshall J. Styczinski

Copyright (C) 2024 the authors.
