"""
pyALMA3 (Python plAnetary Love nuMbers cALculator)
pythonized version of ALMA 3
------------------------------------------------------------------------------

Author: Saikiran Tharimena, Daniele Melini, Giorgio Spada, Steven D. Vance, Marshall J. Styczinski
Copyright (C) 2024 the authors
"""

import os.path
import shutil

# %% Import Libraries
from toml import load as tomlload
from psutil import Process
# Joblib
from os import cpu_count
from os.path import abspath
from joblib import Parallel, delayed
from time import perf_counter
# Numpy
from numpy import (ndarray, array, round, loadtxt,
                   vstack, power, gradient, arange,
                   logspace, linspace, geomspace,
                   flipud, where, ones, insert)
# Math
from math import floor
# MPMATH
from mpmath import mp
from mpmath import (binomial, matrix, factorial, eye,
                    lu_solve, log)

# Set accessible file paths for installation directory and default configs
_ROOT = os.path.abspath(os.path.dirname(__file__))
_defaultConfig = os.path.join(_ROOT, 'defaultParamsPyALMA3.toml')

# Copy user config files to local dir if the user does not have them yet
_userConfig = 'paramsPyALMA3.toml'

# Functions for managing copying of default config file
def CopyOnlyIfNeeded(source, destination):
    if not os.path.isfile(destination):
        try:
            if os.path.dirname(destination) != '':
                os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy(source, destination)
        except OSError as err:
            raise OSError(f'Unable to copy from {source} to {destination}. ' +
                          f'Check that you have write permission for {os.getcwd()}. ' +
                          f'The error reported was:\n{err}')
        else:
            print(f'{destination} was copied from default at {source}.')

    return


def RemoveCarefully(file):
    if os.path.isfile(file):
        os.remove(file)


# Helper function to kill spawned threads
def clear_threads(mode, before=None, verbose=False):
    current_process = Process()

    if mode == 'start':
        return set([p.pid for p in current_process.children(recursive=True)])
    else:
        after = set([p.pid for p in current_process.children(recursive=True)])

        if verbose: print('  >> Clearing spawned threads.')
        for subproc in after - before:
            Process(subproc).terminate()

    return


# %% Initialize
def initialize(ndigits):
    mp.dps = ndigits

    # Set constants
    Gnwt = mp.mpf('6.674e-11')
    iota = mp.mpc(0, 1)

    return Gnwt, iota


# %% Parse Rheology
def get_rheology(rheology, nla, params):
    # Parse rheology
    rheol = []
    for rcode in rheology:
        if type(rcode) == str:
            if rcode.lower() == 'fluid':
                rheol.append(0)
            elif rcode.lower() == 'elastic':
                rheol.append(1)
            elif rcode.lower() == 'maxwell':
                rheol.append(2)
            elif rcode.lower() == 'newton':
                rheol.append(3)
            elif rcode.lower() == 'kelvin':
                rheol.append(4)
            elif rcode.lower() == 'burgers':
                rheol.append(5)
            elif rcode.lower() == 'andrade':
                rheol.append(6)
            else:
                raise ValueError('Unknown rheology name: {}'.format(rcode))
        elif type(rcode) == int:
            if rcode >= 0 and rcode <= 6:
                rheol.append(rcode)
            else:
                raise ValueError('Invalid rheology code: {}'.format(rcode))
        else:
            raise TypeError('Invalid rheology argument')

    # Compute gamma(alpha+1) for Andrade layers
    rpar = mp.matrix(params)

    for i in range(nla):
        if rheol[i] == 6:
            rpar[i, 1] = mp.gamma(params[i, 0] + 1)

    return rheol, rpar


# %% Build Model
def wrapper_layer(rho, ri_1, ri):
    return rho * (ri_1 ** 3 - ri ** 3)


def build_model(r_in, rho_in, mu_in, eta_in, rheology, params, ndigits=128, verbose=True,
                parallel=None):
    if verbose:
        print('> Initializing')
        print(f'  >> Setting precision: {ndigits}')

    Gnwt, iota = initialize(ndigits)

    if verbose:
        print('> Building model')

    # Check arguments
    for arg in [r_in, rho_in, mu_in, eta_in, params]:
        if type(arg) not in [list, tuple, ndarray]:
            raise TypeError("Invalid argument type")

    if type(rheology) not in [list, tuple]:
        raise TypeError("Invalid argument type")

    # Check argument sizes
    nla = len(rho_in)
    for arg in [r_in, rho_in, mu_in, eta_in, rheology, params]:
        if len(arg) != nla:
            raise ValueError("Argument size mismatch")

    # Set model parameters
    r = matrix(nla + 1, 1)
    r[1:] = matrix(r_in)

    rho = matrix(rho_in)
    mu = matrix(mu_in)
    eta = matrix(eta_in)

    # Parse Rheology
    if verbose:
        print('  >> Parsing Rheology')

    rheol, rpar = get_rheology(rheology, nla, params)

    # Build Model
    gra = matrix(nla + 1, 1)
    mlayer = matrix(nla, 1)

    # Planet Mass
    if verbose:
        print('  >> Computing mass of the planet')

    # Sanity check for parallel
    if nla <= 10 and parallel is None:
        parallel = False

    elif nla > 10 and parallel is None:
        parallel = True

    else:
        if bool(parallel):
            parallel = parallel
        else:
            print(f'WARNING: parallel setting should be boolean, it is: {type(parallel)}.')
            print('Reverting to serial operation.')
            parallel = False

    if parallel:
        '''
        def wrapper_layer(rho, ri_1, ri):
            return rho * (ri_1 ** 3 - ri ** 3)
        '''
        mlayer = Parallel(verbose=0, n_jobs=-1)(
            delayed(wrapper_layer)(rho[i], r[i + 1], r[i]) for i in range(nla))
        mlayer = matrix(mlayer)

    else:
        for i in range(nla):
            mlayer[i] = rho[i] * (r[i + 1] ** 3 - r[i] ** 3)

    for i in range(nla):
        mlayer[i] = rho[i] * (r[i + 1] ** 3 - r[i] ** 3)

    mlayer = mlayer * mp.mpf('4') / mp.mpf('3') * mp.pi

    mass = sum(mlayer)

    # Gravity
    if verbose:
        print('  >> Computing gravity at the interface boundaries')
    '''
    if parallel:
        def wrapper_gravity(Gnwt, mlayer, r):
            return Gnwt * sum(mlayer) / (r ** 2)

        if verbose: vbse=1

        gra = Parallel(verbose=vbse, n_jobs=-1)(delayed(wrapper_gravity)(Gnwt, mlayer[0:i], r[i]) for i in range(1, nla + 1))

    else:
        for i in range(1, nla + 1):
            gra[i] = Gnwt * sum(mlayer[0:i]) / (r[i] ** 2)
    '''
    for i in range(1, nla + 1):
        gra[i] = Gnwt * sum(mlayer[0:i]) / (r[i] ** 2)

    # Normalize
    if verbose:
        print('  >> Normalizing model parameters')

    # Reference scales
    r0 = r[nla]
    rho0 = rho[1]
    mu0 = max(mu)
    t0 = mp.mpf('1000') * mp.mpf('365.25') * mp.mpf('24') * mp.mpf('3600')
    eta0 = mu0 * t0
    mass0 = rho0 * (r0 ** 3)

    # Normalize model parameters
    r = r / r0
    rho = rho / rho0
    mu = mu / mu0
    eta = eta / eta0

    # Normalize Newton constant
    G = Gnwt * (rho0 ** 2) * (r0 ** 2) / mu0

    # Normalize mass
    mass = mass / mass0
    mlayer = mlayer / mass0

    # Normalize gravity at internal interfaces
    gra = gra * (r0 ** 2) * (G / Gnwt) / mass0

    # Output as dict object
    model = {'r': r,
             'rho': rho,
             'mu': mu,
             'eta': eta,
             'G': G,
             'mass': mass,
             'mlayer': mlayer,
             'gra': gra,
             'rheol': rheol,
             'rpar': rpar,
             'iota': iota}

    return model


# Parse input parameters
def parse_loadtype(loadtype):
    # Set load type ('tidal' or 'loading')
    if loadtype.lower() == 'tidal':
        iload = 0
    elif loadtype.lower() == 'loading':
        iload = 1
    else:
        raise ValueError('Unknown load type {}'.format(loadtype))

    return iload


def parse_outputtype(output):
    # set output LN type
    if output.lower() == 'real':
        itype = 1
    elif output.lower() == 'complex':
        itype = 2
    elif output.lower() == 'rate':
        itype = 3
    else:
        raise ValueError('Unknown LN mode {}'.format(output))

    return itype


def parse_histtype(loadfcn):
    if loadfcn.lower() == 'step':
        ihist = 1
    elif loadfcn.lower() == 'ramp':
        ihist = 2
    else:
        raise ValueError('Unknown load time-history: {}'.format(loadfcn))
    return ihist


# Subroutines from Fortran
# Fluid Core bc
# computes the boundary conditions at the CMB for a fluid inviscid core
def fluid_core_bc(n, r, rho, gra, G):
    b = matrix(6, 3)

    b[0, 0] = - 1 / gra * (r ** n)
    b[0, 2] = 1

    b[1, 1] = 1

    b[2, 2] = rho * gra

    b[4, 0] = r ** n

    b[5, 0] = 2 * (n - 1) * r ** (n - 1)
    b[5, 2] = 4 * mp.pi * G * rho

    return b


# Surface bc
# computes the boundary conditions at the surface
def surface_bc(n, r, gra, iload, G):
    bs = matrix(3, 1)
    kappa = (2 * n + 1) / (4 * mp.pi * (r ** 2))

    if iload == 1:
        bs[0] = -gra * kappa

    bs[2] = -4 * mp.pi * G * kappa

    return bs


# Salzer weights
# computes the weights zeta(i) to be used in the Salzer accelerated
# Post-Widder Laplace inversions scheme
def salzer_weights(order, verbose=True):
    if verbose:
        print('> Computing Salzer weights')

    if type(order) != int:
        raise TypeError("salzer_weights: order must be integer")

    # m = order
    zeta = matrix(2 * order, 1)

    for k in range(1, 2 * order + 1):

        j1 = floor((k + 1) / 2)
        j2 = min(k, order)

        for j in range(j1, j2 + 1):
            fattm = factorial(order)

            q1 = binomial(order, j)
            q2 = binomial(2 * j, j)
            q3 = binomial(j, k - j)

            zeta[k - 1] = zeta[k - 1] + j ** (order + 1) / fattm * q1 * q2 * q3

        if (order + k) % 2 != 0:
            zeta[k - 1] = -zeta[k - 1]

    return zeta


# Complex rigidity
# computes mu(s) for various rheologies
def complex_rigidity(s, mu, eta, code, par):
    if code == 1:  # Elastic
        mu_s = mu
    elif code == 2:  # Maxwell
        mu_s = mu * s / (s + mu / eta)
    elif code == 3:  # Newton
        mu_s = eta * s
    elif code == 4:  # Kelvin
        mu_s = mu + eta * s
    elif code == 5:  # Burgers
        mu2 = par[0] * mu
        eta2 = par[1] * eta
        mu_s = mu * s * (s + mu2 / eta2) / (
                    s ** 2 + s * (mu / eta + (mu + mu2) / eta2) + (mu * mu2) / (eta * eta2))
    elif code == 6:  # Andrade
        alpha = par[0]
        gam = par[1]
        mu_s = 1 / mu + 1 / (eta * s) + gam * (1 / mu) * (s * eta / mu) ** (-alpha)
        mu_s = 1 / mu_s
    else:
        raise ValueError('Invalid rheology code: {}'.format(code))

    return mu_s


# Direct matrix
# computes the fundamental matrix in a layer
def direct_matrix(n, r, rho, mu, gra, G):
    assert type(n) == int, "harmonic degree shall be integer"

    a1 = 2 * n + 3  # 2n+3
    a2 = n + 1  # n+1
    a3 = n + 3  # n+3
    a4 = n ** 2 - n - 3  # n^2-n-3
    a5 = n + 2  # n+2
    a6 = n - 1  # n-1
    a7 = 2 * n + 1  # 2n+1
    a8 = 2 * n - 1  # 2n-1
    a9 = 2 - n  # 2-n
    a10 = n ** 2 + 3 * n - 1  # n^2+3n-1
    a11 = n ** 2 - 1  # n^2-1

    Y = matrix(6)

    Y[0, 0] = n / (2 * a1) * r ** (n + 1)
    Y[1, 0] = a3 / (2 * a1 * a2) * r ** (n + 1)
    Y[2, 0] = (n * rho * gra * r + 2 * a4 * mu) / (2 * a1) * r ** n
    Y[3, 0] = n * a5 / (a1 * a2) * mu * r ** n
    Y[5, 0] = 2 * mp.pi * G * rho * n / a1 * r ** (n + 1)

    Y[0, 1] = r ** (n - 1)
    Y[1, 1] = 1 / n * r ** (n - 1)
    Y[2, 1] = (rho * gra * r + 2 * a6 * mu) * r ** (n - 2)
    Y[3, 1] = 2 * a6 / n * mu * r ** (n - 2)
    Y[5, 1] = 4 * mp.pi * G * rho * r ** (n - 1)

    Y[2, 2] = rho * r ** n
    Y[4, 2] = r ** n
    Y[5, 2] = a7 * r ** (n - 1)

    Y[0, 3] = a2 / (2 * a8) * r ** (-n)
    Y[1, 3] = a9 / (2 * n * a8) * r ** (-n)
    Y[2, 3] = (a2 * rho * gra * r - 2 * a10 * mu) / (2 * a8) * r ** (-n - 1)
    Y[3, 3] = a11 / (n * a8) * mu * r ** (-n - 1)
    Y[5, 3] = 2 * mp.pi * G * rho * a2 / a8 * r ** (-n)

    Y[0, 4] = r ** (-n - 2)
    Y[1, 4] = - 1.0 / a2 * r ** (-n - 2)
    Y[2, 4] = (rho * gra * r - 2 * a5 * mu) * r ** (-n - 3)
    Y[3, 4] = 2 * a5 / a2 * mu * r ** (-n - 3)
    Y[5, 4] = 4 * mp.pi * G * rho * r ** (-n - 2)

    Y[2, 5] = rho * r ** (-n - 1)
    Y[4, 5] = r ** (-n - 1)

    return Y


# Inverse matrix
# computes the inverse of the fundamental matrix in a layer
def inverse_matrix(n, r, rho, mu, gra, G):
    a1 = 2 * n + 1  # 2n+1
    a2 = 2 * n - 1  # 2n-1
    a3 = n + 1  # n+1
    a4 = n - 1  # n-1
    a5 = n + 2  # n+2
    a6 = n + 3  # n+3
    a7 = n ** 2 + 3 * n - 1  # n^2+3n-1
    a8 = n ** 2 - n - 3  # n^2-n-3
    a9 = 2 - n  # 2-n
    a10 = n ** 2 - 1  # n^2-1
    a11 = 2 * n + 3  # 2n+3

    Yinv = matrix(6)
    D = matrix(6)

    D[0, 0] = a3 * r ** (-(n + 1))
    D[1, 1] = a3 * n / (2 * a2) * r ** (-(n - 1))
    D[2, 2] = -r ** (-(n - 1))
    D[3, 3] = n * r ** n
    D[4, 4] = n * a3 / (2 * a11) * r ** (n + 2)
    D[5, 5] = r ** (n + 1)

    D = D / a1

    Yinv[0, 0] = rho * gra * r / mu - 2 * a5
    Yinv[1, 0] = - rho * gra * r / mu + 2 * a7 / a3
    Yinv[2, 0] = 4 * mp.pi * G * rho
    Yinv[3, 0] = rho * gra * r / mu + 2 * a4
    Yinv[4, 0] = - rho * gra * r / mu - 2 * a8 / n
    Yinv[5, 0] = 4 * mp.pi * G * rho * r

    Yinv[0, 1] = 2 * n * a5
    Yinv[1, 1] = - 2 * a10
    Yinv[3, 1] = 2 * a10
    Yinv[4, 1] = - 2 * n * a5

    Yinv[0, 2] = - r / mu
    Yinv[1, 2] = r / mu
    Yinv[3, 2] = - r / mu
    Yinv[4, 2] = r / mu

    Yinv[0, 3] = n * r / mu
    Yinv[1, 3] = a9 * r / mu
    Yinv[3, 3] = - a3 * r / mu
    Yinv[4, 3] = a6 * r / mu

    Yinv[0, 4] = rho * r / mu
    Yinv[1, 4] = - rho * r / mu
    Yinv[3, 4] = rho * r / mu
    Yinv[4, 4] = - rho * r / mu
    Yinv[5, 4] = a1

    Yinv[2, 5] = - 1
    Yinv[5, 5] = - r

    # Yinv = D * Yinv

    return D * Yinv


# Love number sampler
# computes the Love numbers h,l,k in the Laplace domain for
# a given value of s
def love_numbers_sampler(n, s, iload, model_params, verbose=True):
    # lam = matrix(6, 6)
    # for i in range(6):
    #    lam[i,i] = mp.mpf('1')
    lam = eye(6)

    # Model parameters
    rho = model_params['rho']
    r = model_params['r']
    G = model_params['G']
    mu = model_params['mu']
    eta = model_params['eta']
    gra = model_params['gra']
    mass = model_params['mass']
    rpar = model_params['rpar']
    rheol = model_params['rheol']

    # Build the propagator product
    nla = len(rho)

    for i in range(nla - 1, 0, -1):
        # Ydir = direct_matrix(n, r[i + 1], rho[i], mu_s, gra[i + 1], G)
        # Yinv = inverse_matrix(n, r[i], rho[i], mu_s, gra[i], G)
        # lam = lam * (Ydir * Yinv)
        mu_s = complex_rigidity(s, mu[i], eta[i], rheol[i], rpar[i, :])
        lam = lam * (direct_matrix(int(n), r[i + 1], rho[i], mu_s, gra[i + 1], G) * inverse_matrix(
            int(n), r[i], rho[i], mu_s, gra[i], G))

    if rheol[0] == 0:
        bc = fluid_core_bc(n, r[1], rho[0], gra[1], G)

    else:
        mu_s = complex_rigidity(s, mu[0], eta[0], rheol[0], rpar[0, :])
        Ydir = direct_matrix(int(n), r[1], rho[0], mu_s, gra[1], G)
        bc = Ydir[:, 0:3]

    bs = surface_bc(n, r[nla], gra[nla], iload, G)

    # Compute the 'R' and 'Q' arrays
    prod = lam * bc

    rr = matrix(3)
    qq = matrix(3)

    rr[0, :] = prod[2, :]
    rr[1, :] = prod[3, :]
    rr[2, :] = prod[5, :]

    qq[0, :] = prod[0, :]
    qq[1, :] = prod[1, :]
    qq[2, :] = prod[4, :]

    bb = lu_solve(rr, bs)
    x = qq * bb

    hh = x[0] * mass / r[nla]
    ll = x[1] * mass / r[nla]
    kk = (-1 - x[2] * mass / (r[nla] * gra[nla]))

    return hh, ll, kk


# Compute love numbers
# Putting this wrapper so it's easy to parallelize
def wrapper_love_numbers_timestep(idx_n, params, t1=0, verbose=False):
    # Dummy initiator
    h_love = matrix(1, params['nt'])
    l_love = matrix(1, params['nt'])
    k_love = matrix(1, params['nt'])

    # Harmonic degree
    n = params['degrees'][idx_n]

    if params['itype'] == 1 or params['itype'] == 3:
        for idx_t in range(params['nt']):  # range(nt):
            # print(f'idx_t: {idx_t}')
            t = params['timesteps'][idx_t]  # timesteps[idx_t]
            f = log(2) / t

            for ik in range(1, 2 * params['order'] + 1):  # range(1, 2*order + 1):

                s = f * ik
                # hh, ll, kk = love_numbers_sampler(n, s, iload, model_params)
                hh, ll, kk = love_numbers_sampler(n, s, params['iload'], params['model_params'])

                if params['ihist'] == 1:  # ihist == 1:
                    fh = mp.mpf('1') / s

                elif params['ihist'] == 2:  # ihist == 2:
                    fh = (mp.mpf('1') - mp.exp(-s * params['tau'])) / (params['tau'] * s ** 2)

                if params['itype'] == 3:
                    # h_love[idx_n, idx_t] += fh * (s * hh) * params['zeta'][ik-1] * f
                    # l_love[idx_n, idx_t] += fh * (s * ll) * params['zeta'][ik-1] * f
                    # k_love[idx_n, idx_t] += fh * (s * kk) * params['zeta'][ik-1] * f

                    h_love[idx_t] += fh * (s * hh) * params['zeta'][ik - 1] * f
                    l_love[idx_t] += fh * (s * ll) * params['zeta'][ik - 1] * f
                    k_love[idx_t] += fh * (s * kk) * params['zeta'][ik - 1] * f

                else:
                    # h_love[idx_n, idx_t] += fh * hh * params['zeta'][ik-1] * f
                    # l_love[idx_n, idx_t] += fh * ll * params['zeta'][ik-1] * f
                    # k_love[idx_n, idx_t] += fh * kk * params['zeta'][ik-1] * f

                    h_love[idx_t] += fh * hh * params['zeta'][ik - 1] * f
                    l_love[idx_t] += fh * ll * params['zeta'][ik - 1] * f
                    k_love[idx_t] += fh * kk * params['zeta'][ik - 1] * f

    elif params['itype'] == 2:
        for idx_t in range(params['nt']):
            # print(f'idx_t: {idx_t} > {params["timesteps"][idx_t]}')
            t = params['timesteps'][idx_t]
            omega = mp.mpf('2') * mp.pi / t
            s = params['iota'] * omega
            hh, ll, kk = love_numbers_sampler(n, s, params['iload'], params['model_params'])

            # h_love[idx_n, idx_t] = hh
            # l_love[idx_n, idx_t] = ll
            # k_love[idx_n, idx_t] = kk

            h_love[idx_t] = hh
            l_love[idx_t] = ll
            k_love[idx_t] = kk

    if verbose:
        t2 = perf_counter()
        print('Harmonic degree n = {} ({} s)'.format(n, round(t2 - t1, 2)))
        t1 = t2

    return {'h_love': h_love,
            'l_love': l_love,
            'k_love': k_love}, t1


def love_numbers(degrees, timesteps, loadtype, loadfcn, tau, model_params, output, order,
                 verbose=True, parallel=None):
    # Set parameters in dict
    params = {'nt': len(timesteps),
              'ndeg': len(degrees),
              'iota': model_params['iota'],
              'iload': parse_loadtype(loadtype),
              'itype': parse_outputtype(output),
              'tau': tau,
              'order': order,
              'degrees': degrees,
              'timesteps': matrix(timesteps),
              'model_params': model_params}

    # Parse loading, output and hist types
    if params['itype'] != 2:
        # ihist = parse_histtype(loadfcn)
        params['ihist'] = parse_histtype(loadfcn)
    else:
        # ihist = 0
        params['ihist'] = 0

    # Allocate output arrays
    h_love = matrix(params['ndeg'], params['nt'])
    l_love = matrix(params['ndeg'], params['nt'])
    k_love = matrix(params['ndeg'], params['nt'])

    # Compute LNs
    t1 = perf_counter()

    # Parallelize if ndeg > 2
    params['zeta'] = salzer_weights(order, verbose=verbose)

    # Sanity check for parallel
    if parallel is None and params['ndeg'] > 2 and cpu_count() > 2:
        parallel = True
    elif parallel is None and (params['ndeg'] <= 2 or cpu_count() <= 2):
        parallel = False
    else:
        if type(parallel) == bool:
            parallel = parallel
        else:
            print(f'WARNING: parallel setting should be boolean, it is: {type(parallel)}.')
            print('Reverting to serial operation.')
            parallel = False

    if parallel:
        if verbose: print('> Computing Love Numbers - Parallel')

        # Safety net if code doesn't clear spawned threads
        process_before = clear_threads('start')

        pverbose = 3 if verbose else 0

        results = Parallel(verbose=pverbose, n_jobs=-1, backend='threading')(
            delayed(wrapper_love_numbers_timestep)(idx_n, params) for idx_n in
            range(params['ndeg']))

        _ = clear_threads('end', before=process_before, verbose=verbose)

        # splice all results
        for idx_n, lovens in enumerate(results):
            h_love[idx_n, :] = lovens[0]['h_love']
            l_love[idx_n, :] = lovens[0]['l_love']
            k_love[idx_n, :] = lovens[0]['k_love']

    else:
        if verbose: print('> Computing Love Numbers - Serial')

        for idx_n in range(params['ndeg']):
            if verbose:
                print(f'   >> processing {idx_n + 1} of {params["ndeg"]}')

            lovens, t1 = wrapper_love_numbers_timestep(idx_n, params, t1, verbose=verbose)

            h_love[idx_n, :] = lovens['h_love']
            l_love[idx_n, :] = lovens['l_love']
            k_love[idx_n, :] = lovens['k_love']

    if params['itype'] == 1 or params['itype'] == 3:
        h_love = array(h_love.tolist(), dtype=float)
        l_love = array(l_love.tolist(), dtype=float)
        k_love = array(k_love.tolist(), dtype=float)

    elif params['itype'] == 2:
        h_love = array(h_love.tolist(), dtype=complex)
        l_love = array(l_love.tolist(), dtype=complex)
        k_love = array(k_love.tolist(), dtype=complex)

    return h_love, l_love, k_love


# Read Planet Profile Model
# %% Read velocity model from PlanetProfile v2.0+
def read_model_pp(fname, verbose=False):
    # TODO: correct equations for MU, VIS and RIGIDITY

    if verbose:
        print('> Reading model from PlanetProfile output')

    # Read file
    with open(fname, 'r') as f:
        modelLabel = f.readline()
        nHeadLines = int(f.readline().split('=')[-1])

    # Read data
    P_MPa, T_K, r_m, phase, rho_kgm3, Cp_JkgK, alpha_pK, \
        g_ms2, phi_frac, sigma_Sm, kTherm_WmK, VP_kms, VS_kms, \
        QS, KS_GPa, GS_GPa, Ppore_MPa, rhoMatrix_kgm3, \
        rhoPore_kgm3, MLayer_kg, VLayer_m3, Htidal_Wm3 \
        = loadtxt(fname, skiprows=nHeadLines, unpack=True)

    # List of variable names
    COLUMNS = ['P', 'T', 'r', 'phase', 'rho', 'Cp', 'alpha',
               'g', 'phi', 'sigma', 'kTherm', 'VP', 'VS',
               'QS', 'KS', 'GS', 'Ppore', 'rhoMatrix',
               'rhoPore', 'MLayer', 'VLayer', 'Htidal']

    # List of data units
    UNITS = ['MPa', 'K', 'm', '', 'kg m-3', 'J kg-1 K-1', 'K-1',
             'm s-2', '-', 'S m-1', 'W m-1 K-1', 'km s-1', 'km s-1',
             '', 'GPa', 'GPa', 'MPa', 'kg m-3',
             'kg m-3', 'kg', 'm3', 'W m-3']

    # Combine data into columns to reuse existing infrastructure
    MODEL = vstack([P_MPa, T_K, r_m, phase, rho_kgm3, Cp_JkgK, alpha_pK,
                    g_ms2, phi_frac, sigma_Sm, kTherm_WmK, VP_kms, VS_kms,
                    QS, KS_GPa, GS_GPa, Ppore_MPa, rhoMatrix_kgm3,
                    rhoPore_kgm3, MLayer_kg, VLayer_m3, Htidal_Wm3]).T

    # Check GPA and MPA to Pa
    changeheader = ['P', 'VP', 'VS', 'r', 'rho', 'KS', 'GS']
    for index, (header, unit) in enumerate(zip(COLUMNS, UNITS)):
        if header in changeheader:
            if unit.lower() == 'mpa':
                UNITS[index] = 'Pa'
                MODEL[:, index] = MODEL[:, index] * 1e6

            elif unit.lower() == 'gpa':
                UNITS[index] = 'Pa'
                MODEL[:, index] = MODEL[:, index] * 1e9

            elif unit.lower() == 'km':
                UNITS[index] = 'm'
                MODEL[:, index] = MODEL[:, index] * 1e3

            elif unit.lower() == 'km s-1':
                UNITS[index] = 'm s-1'
                MODEL[:, index] = MODEL[:, index] * 1e3

    # Round radius
    MODEL[:, COLUMNS.index('r')] = round(MODEL[:, COLUMNS.index('r')], 0)

    # Flip model: core at top and surface at bottom
    if MODEL[0, COLUMNS.index('r')] > MODEL[-1, COLUMNS.index('r')]:
        MODEL = flipud(MODEL)

    ## Calculate elastic parameters
    # LAMBDA = rho (Vp^2 - 2 * Vs^2)
    # 1st Lame parameter
    LAMBDA = MODEL[:, COLUMNS.index('rho')] * (
                power(MODEL[:, COLUMNS.index('VP')], 2) - 2. * power(MODEL[:, COLUMNS.index('VS')],
                                                                     2))

    # shear modulus G or MU = rho Vs^2
    # MU = MODEL[:, COLUMNS.index('rho')] * power(MODEL[:, COLUMNS.index('VS')], 2)
    MU = MODEL[:, COLUMNS.index('rho')] * power(MODEL[:, COLUMNS.index('VP')], 2)

    # Bulk Modulus K = lambda + 2/3 mu
    K = LAMBDA + 2 * MU / 3

    # Poissons ratio sigma = lambda / 2*(lambda + mu)
    SIGMA = LAMBDA / (2 * LAMBDA + 2 * MU)

    # Youngs modulus Y = 2 * MU * (1 + sigma)
    Y = 2. * MU * (1 + SIGMA)

    # Rigidity RIG = 2/3 mu
    RIG = 2 * MU / 3

    # Viscosity VIS = RIG / GRAD(Vs)
    gradvs = abs(gradient(MODEL[:, COLUMNS.index('VS')], MODEL[:, COLUMNS.index('r')]))
    gradvs[gradvs == 0.] = 1e-30

    VIS = RIG / gradvs

    # JUST FOR TESTING PP MODELS
    # SET MU and VIS to min value
    # VIS[VIS == 0.] = min(VIS[VIS>0]) / 10000
    # MU[MU == 0.] = min(MU[MU>0]) / 10000
    # VIS[VIS == 0.] = (0. * where(VIS==0.)[0] + 1.) * linspace(min(VIS[VIS>0]) - 20000, min(VIS[VIS>0]) - 150000, num=len(where(VIS==0.)[0]))
    # MU[MU == 0.] = (0. * where(MU==0.)[0] + 1.) * linspace(min(MU[MU>0]) - 20000, min(MU[MU>0]) - 150000, num=len(where(MU==0.)[0]))

    # Return output variables
    return {'columns': COLUMNS,
            'units': UNITS,
            'model': MODEL,
            'lambda': LAMBDA,
            'mu': MU,
            'k': K,
            'sigma': SIGMA,
            'y': Y,
            'rig': RIG,
            'vis': VIS}


# Infer rheology from pp model
# Type should be from core to surface
def infer_rheology_pp(model, structure=None, layer_radius=None, layer_radius_index=False,
                      verbose=False):
    '''model         - model object that is returned by "read_model_pp"
       structure     - rheologies of layers
                        '0 : "fluid"
                        '1 : "elastic"
                        '2 : "maxwell"
                        '3 : "newton"
                        '4 : "kelvin"
                        '5 : "burgers"
                        '6 : "andrade"
        layer_radius - manually define transition in layers. For example: this can be defined as [50, 100]. This will be treated as a 3 layer model with first layer from 1 - 50m, second layer from 50 - 100m, and final layer from 100m to end of model file. Here, unit should be same as unit of radius. Always define layers from inside to outside for a planet profile model.

        layer_radius_index - boolean. If set to true, layer_radius values will be treated as index. In the above example, the model is treated as a 3 layered model. First layer is from index 0 - 50, second layer from 51 - 100, final later from 100 to length of model.
    '''

    if verbose:
        print('> Generating rheology for PlanetProfile model')

    if layer_radius is False or layer_radius is None:
        layer_radius = None

    if layer_radius is None and structure is None:
        print(
            '  >> ERROR: Layer rheology structure of model (from core to surface) should be defined when using Planet Profile model file.\nExample: for enceladus this is defined as a list ["elastic", "newton", "maxwell"] corresponding to 3 layers: core, water, ice.')
        print('\nRheology structure can be defined as code or name:')
        print('Code : Name')
        print('0 : "fluid"')
        print('1 : "elastic"')
        print('2 : "maxwell"')
        print('3 : "newton"')
        print('4 : "kelvin"')
        print('5 : "burgers"')
        print('6 : "andrade"')
        raise ValueError()

    # Sanity check for structure
    print('  >> Verifying validity of structure rheologies')
    if structure is None:
        print('  >> ERROR: Rheology structure must be defined')
        raise ValueError()

    if array([type(i) is str for i in structure]).all():
        for i in structure:
            if i not in ['fluid', 'elastic', 'maxwell', 'newton', 'kelvin', 'burgers', 'andrade']:
                print(f'  >> ERROR: Unknown rheology: {i}')
                raise ValueError()

    elif array([type(i) is int for i in structure]).all() or array(
            [type(i) is float for i in structure]).all():
        # Convert to int
        structure = [int(i) for i in structure]

        for i in structure:
            if i < 0 or i > 6:
                print(f'  >> ERROR: Unknown rheology code: {i}. Code should be between [0, 6]')
                raise ValueError()

    else:
        print(
            f'  >> ERROR: Unknown data type for rheology. Accepted values are string or integer code.')
        print('\nRheology structure can be defined as code or name:')
        print('Code : Name')
        print('0 : "fluid"')
        print('1 : "elastic"')
        print('2 : "maxwell"')
        print('3 : "newton"')
        print('4 : "kelvin"')
        print('5 : "burgers"')
        print('6 : "andrade"')
        raise ValueError()

    if layer_radius is None and structure is not None:

        # Get layer transition points
        layers = where(model['model'][:, model['columns'].index('phase')][:-1]
                       != model['model'][:, model['columns'].index('phase')][1:])[0]

        if len(layers) == 0:
            print('  >> Could not find pre-defined layers from phases.')
            print('     Inferring layers from density profile.')

            # Get gradient to infer layers
            rhograd = where(round(abs(gradient(model['model'][:, model['columns'].index('rho')],
                                               model['model'][:, model['columns'].index('r')])),
                                  0) > 0.)[0]

            if len(rhograd) == 0:
                print('  >> ERROR: Could not infer layers from density profile.')
                print(
                    '     This model requires layer information which can be passed through parameter "layer_radius" (from core to surface).')
                raise ValueError()

            rhogradidx = abs(rhograd[:-1] - rhograd[1:])
            rhogradidx = where(rhogradidx == 1.)[0]

            # Add 1 to get index of rhograd
            if len(rhogradidx) != 0:
                rhogradidx += 1

            # Remove these values
            mask = ones(len(rhograd), bool)
            mask[rhogradidx] = 0
            rhograd = rhograd[mask]

            # Assign rheology
            rheo = []

            if len(structure) != len(rhograd) + 1:
                print(
                    f'  >> ERROR: number of inferred layers={len(rhograd) + 1} is not equal to number of layered rheology structures={len(structure)}.')
                print(
                    '      Error can be mitigated by manually defining boundaries of layers and structure of each layer.')
                raise ValueError()

            for layer in range(len(structure)):
                if layer == 0:
                    rheo = [structure[layer] for i in range(rhograd[layer] + 1)]

                # Intermediate layers
                if layer > 0 and layer < len(structure) - 1:
                    rheo = rheo + [structure[layer] for i in
                                   range((rhograd[layer] + 1) - (rhograd[layer - 1] + 1))]

                # Last layer
                if layer == len(structure) - 1:
                    rheo = rheo + [structure[layer] for i in
                                   range(model['model'].shape[0] - (rhograd[-1] + 1))]

            # Assign parameters
            params = 0. * ones((len(rheo), 2))

        else:  # pre-defined layers exist
            print(f'  >> Found {len(layers) + 1} pre-defined layers in model radius')

            if len(structure) != len(layers) + 1:
                print(
                    f'  >> ERROR: number of pre-defined layers={len(layers) + 1} is not equal to number of layered rheology structures={len(structure)}.')
                raise ValueError()

            for layer in range(len(structure)):
                if layer == 0:
                    rheo = [structure[layer] for i in range(layers[layer] + 1)]

                # Intermediate layers
                if layer > 0 and layer < len(structure) - 1:
                    rheo = rheo + [structure[layer] for i in
                                   range((layers[layer] + 1) - (layers[layer - 1] + 1))]

                # Last layer
                if layer == len(structure) - 1:
                    rheo = rheo + [structure[layer] for i in
                                   range(model['model'].shape[0] - (layers[-1] + 1))]

            # Assign parameters
            params = 0. * ones((len(rheo), 2))

    else:  # User defined layers and structure
        if type(layer_radius_index) is not bool:
            print('  >> ERROR: "layer_radius_index" should be boolean')
            raise ValueError()

        if len(layer_radius) + 1 != len(structure):
            print(
                f'  >> ERROR: number of layers + 1={len(layer_radius) + 1} is not equal to number of number of layered rheology structures={len(structure)}')
            print('     "layer_radius" should have values')
            raise ValueError()

        if layer_radius_index:
            print(f'  >> Setting layered rheology at index: {layer_radius}')

            layer_radius = array(layer_radius)
            if min(layer_radius) < 0 or max(layer_radius) > model['model'].shape[0]:
                print(
                    f'  >> ERROR: layer_radius index value: {min(layer_radius)}, {max(layer_radius)} not within range: [0, {model["model"].shape[0]}].')
                raise ValueError()

            if ((layer_radius[1:] - layer_radius[:-1]) < 0).any():
                print('  >> ERROR: layer_radius should be monotonically increasing.')
                raise ValueError()

            for layer in range(len(structure)):
                if layer == 0:
                    rheo = [structure[layer] for i in range(layer_radius[layer] + 1)]

                # Intermediate layers
                if layer > 0 and layer < len(structure) - 1:
                    rheo = rheo + [structure[layer] for i in
                                   range((layer_radius[layer] + 1) - (layer_radius[layer - 1] + 1))]

                # Last layer
                if layer == len(structure) - 1:
                    rheo = rheo + [structure[layer] for i in
                                   range(model['model'].shape[0] - (layer_radius[-1] + 1))]

            # Assign parameters
            params = 0. * ones((len(rheo), 2))

        else:
            print(f'  >> Setting layered rheology at radius: {layer_radius}')

            # sanity check
            layer_radius = array(layer_radius)
            if ((layer_radius[1:] - layer_radius[:-1]) < 0).any():
                print('ERROR: layer_radius should be monotonically increasing.')
                raise ValueError()

            # Get closest index to defined layer_radius
            layers = []
            rads = model['model'][:, model['columns'].index('r')]
            for layer in layer_radius:
                if layer < min(rads) or layer > max(rads):
                    print(
                        f'ERROR: layer_radius: {layer} is beyond model range: {min(rads)}, {max(rads)}')
                    raise ValueError()

                vals = abs(rads - layer)

                layers.append(where(vals == min(vals))[0][0])

            for layer in range(len(structure)):
                if layer == 0:
                    rheo = [structure[layer] for i in range(layers[layer] + 1)]

                # Intermediate layers
                if layer > 0 and layer < len(structure) - 1:
                    rheo = rheo + [structure[layer] for i in
                                   range((layers[layer] + 1) - (layers[layer - 1] + 1))]

                # Last layer
                if layer == len(structure) - 1:
                    rheo = rheo + [structure[layer] for i in
                                   range(model['model'].shape[0] - (layers[-1] + 1))]

            # Assign parameters
            params = 0. * ones((len(rheo), 2))

    return rheo, params


# Main file
# need to complete this
def run_main_pp():
    params = 'paramsPyALMA3.toml'
    if not os.path.isfile(params):
        raise FileNotFoundError(f'Config file {params} was not found in {os.getcwd()}. Copy over ' +
                                'the default file using the following command:\n' +
                                'python -m alma')
    alma_params = tomlload(params)

    # Check none for parallel
    if type(alma_params['parallel']) is not bool:
        alma_params['parallel'] = None

    # Read PP model
    model = read_model_pp(alma_params['filename'])

    # Build model for alma

    rheology, params = infer_rheology_pp(model,
                                         structure=alma_params['rheology_structure'],
                                         layer_radius=alma_params['layer_radius'],
                                         layer_radius_index=alma_params['layer_radius_index'])

    model_params = build_model(model['model'][:, model['columns'].index('r')],
                               model['model'][:, model['columns'].index('rho')],
                               model['mu'],
                               model['vis'],
                               rheology,
                               params,
                               ndigits=alma_params['num_digits'],
                               verbose=alma_params['verbose'])

    # Harmonic degrees
    n = arange(alma_params['dmin'], alma_params['dmax'], alma_params['dstep'])

    # Time step
    if alma_params['time'] == 'lin':
        t = linspace(alma_params['timestart'],
                     alma_params['timeend'],
                     alma_params['tpoints'],
                     endpoint=True)
    elif alma_params['time'] == 'log':
        t = logspace(alma_params['timestart'],
                     alma_params['timeend'],
                     alma_params['tpoints'],
                     endpoint=True)
    elif alma_params['time'] == 'geom':
        t = geomspace(alma_params['timestart'],
                      alma_params['timeend'],
                      alma_params['tpoints'],
                      endpoint=True)
    else:
        print(f'ERROR: Unknown scale for time: {alma_params["time"]}.')
        print('Allowed values are: "log", "geom", "lin".')
        raise ValueError()

    # Compute love numbers
    h, l, k = love_numbers(n, t,
                           alma_params['mode'],
                           alma_params['function'],
                           alma_params['tau'],
                           model_params,
                           alma_params['type'],
                           alma_params['gorder'],
                           verbose=alma_params['verbose'],
                           parallel=alma_params['parallel'])

    return h, l, k


if __name__ == '__main__':
    CopyOnlyIfNeeded(_defaultConfig, _userConfig)
