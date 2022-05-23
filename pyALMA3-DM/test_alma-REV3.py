
import numpy as np
import alma 

import importlib
importlib.reload(alma)


#!------------------------------------------------------------
#! radius,    density,        rigidity      viscosity
#!  (m)       (kg/m^3)          (Pa)          (Pa.s) 
#!------------------------------------------------------------
# 6371e3     3.300e3      0.28e11       1.e21      elastic
# 6271e3     4.518e3	 1.45e11       1.e21      maxwell
# 3480e3    10.977e3        0.e11       0.e21      fluid

r   = np.array( [3480e3, 6271e3, 6371e3] )
rho = np.array( [10.977e3, 4.518e3, 3.300e3] )
mu  = np.array( [    0, 1.45e11, 0.28e11] )
eta = np.array( [    0, 1e21, 1e99] )

reol = [ 'fluid', 'maxwell', 'elastic' ]
par  = np.zeros( (3,2) )

ndigits = 100

n = list( range(2,11) )
#t = [0.1,1,10]
t = [ 1e-4, 1e-3, 1e-2, 1e-1, 1, 10 ]
loadtype='loading'
loadfcn='step'
tau=0
#output='real'
output='complex'
order=8

alma.initialize( ndigits )
alma.build_model( r, rho, mu, eta, reol, par )
h,l,k = alma.love_numbers(n,t,loadtype,loadfcn,tau,output,order)

