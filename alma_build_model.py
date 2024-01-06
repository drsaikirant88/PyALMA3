#%% Import Libraries
import numpy as np

#%% Read velocity model in bm
def read_model_bm(fname):
    print('Reading model from BM velocity files...')

    # Get headers and units
    with open(fname, 'r') as f:
        model = f.readlines()

    # Get number of headerlines
    for index, line in enumerate(model):
        if line[0:2].strip() == 'NA':
            NAME = line.strip().split(' ', 1)[1]

        elif line[0:2].strip() == 'UN':
            UNITS = line.strip().split(' ', 1)[1]

        elif line[0:2].strip() == 'CO':
            COLUMNS = line.strip().split(' ', 1)[1]

        else:
            numheaders = index
            break
    
    # Sanity checks for header variable names
    if 'COLUMNS' not in locals() or 'UNITS' not in locals():
        raise ValueError('Something is wrong with the input model. Check if headers exist.')

    COLUMNS = COLUMNS.strip().split(' ')
    UNITS   = UNITS.strip().split(' ')

    # Remove header lines
    del model[0:numheaders]

    # Skip numheader rows and convert read rest to numpy array
    MODEL = np.empty((0, len(COLUMNS)), dtype=np.float64)
    for line in model:
        MODEL = np.vstack((MODEL, np.array(line.strip().split(' '), dtype=np.float64)))

    ## Elastic parameters
    # LAMBDA = rho (Vp^2 - Vs^2)
    LAMBDA = MODEL[COLUMNS.index('rho')]

    # Return output variables
    return COLUMNS, UNITS, MODEL, RAD, RHO, RIG, VIS

#%% Read velocity model from legacy versions of PlanetProfile (pre-2.0)
def read_model_pp_old(fname):
    print('Reading model from PlanetProfile output...')

    # Read file
    with open(fname, 'r') as f:
        model = f.readlines()

    # Get headers
    COLUMNS = model[0].strip().split('\t')
    UNITS   = []
    temp    = []
    for index, line in enumerate(COLUMNS):
        if len(line) == 0:
            temp.append(index)
            continue

        COLUMNS[index] = line.split(' ', 1)[0].lower()
        try:
            UNITS.append(line.split(' ', 1)[1].strip('()'))
        except IndexError:
            UNITS.append(' ')

    # Remove emply stuff
    for index in reversed(temp):
        del COLUMNS[index]

    # Skip numheader rows and convert read rest to numpy array
    del model[0]
    MODEL = np.empty((0, len(COLUMNS)), dtype=np.float64)
    for line in model:
        MODEL = np.vstack((MODEL, np.array(line.strip().split('\t'), dtype=np.float64)))

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

    ## Calculate elastic parameters
    # LAMBDA = rho (Vp^2 - Vs^2)
    LAMBDA = MODEL[:, COLUMNS.index('rho')] * (np.power(MODEL[:, COLUMNS.index('vp')], 2) - np.power(MODEL[:, COLUMNS.index('vs')], 2))

    # MU = rho Vs^2
    MU = MODEL[:, COLUMNS.index('rho')] * np.power(MODEL[:, COLUMNS.index('vs')], 2)

    # Bulk Modulus K = lambda + 2/3 mu
    K = LAMBDA + 2 * MU / 3

    # Poissons ratio sigma = lambda / 2*(lambda + mu)
    SIGMA = LAMBDA / (2 * LAMBDA + 2* MU)

    # Youngs modulus Y = lambda (1 - 2 sigma) + 2 mu
    Y = LAMBDA * (1 - 2 * SIGMA) + 2 * MU

    # Rigidity RIG = 2/3 mu
    RIG = 2 * MU / 3

    # Viscosity VIS = RIG / GRAD(Vs)
    VIS = RIG / np.gradient(MODEL[:, COLUMNS.index('vs')], MODEL[:, COLUMNS.index('r')])

    # Return output variables
    return COLUMNS, UNITS, MODEL, LAMBDA, MU, K, SIGMA, Y, RIG, VIS

#%% Read velocity model from PlanetProfile v2.0+
def read_model_pp(fname):
    print('Reading model from PlanetProfile output...')

    # Read file
    with open(fname, 'r') as f:
        modelLabel = f.readline()
        nHeadLines = int(f.readline().split('=')[-1])

    # Read data
    P_MPa, T_K, r_m, phase, rho_kgm3, Cp_JkgK, alpha_pK, \
    g_ms2, phi_frac, sigma_Sm, kTherm_WmK, VP_kms, VS_kms, \
    QS, KS_GPa, GS_GPa, Ppore_MPa, rhoMatrix_kgm3, \
    rhoPore_kgm3, MLayer_kg, VLayer_m3, Htidal_Wm3 \
        = np.loadtxt(fname, skiprows=nHeadLines, unpack=True)

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
    MODEL = np.vstack([P_MPa, T_K, r_m, phase, rho_kgm3, Cp_JkgK, alpha_pK,
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

    ## Calculate elastic parameters
    # LAMBDA = rho (Vp^2 - Vs^2)
    LAMBDA = MODEL[:, COLUMNS.index('rho')] * (np.power(MODEL[:, COLUMNS.index('VP')], 2) - np.power(MODEL[:, COLUMNS.index('VS')], 2))

    # MU = rho Vs^2
    MU = MODEL[:, COLUMNS.index('rho')] * np.power(MODEL[:, COLUMNS.index('VS')], 2)

    # Bulk Modulus K = lambda + 2/3 mu
    K = LAMBDA + 2 * MU / 3

    # Poissons ratio sigma = lambda / 2*(lambda + mu)
    SIGMA = LAMBDA / (2 * LAMBDA + 2* MU)

    # Youngs modulus Y = lambda (1 - 2 sigma) + 2 mu
    Y = LAMBDA * (1 - 2 * SIGMA) + 2 * MU

    # Rigidity RIG = 2/3 mu
    RIG = 2 * MU / 3

    # Viscosity VIS = RIG / GRAD(Vs)
    VIS = RIG / np.gradient(MODEL[:, COLUMNS.index('VS')], MODEL[:, COLUMNS.index('r')])

    # Return output variables
    return COLUMNS, UNITS, MODEL, LAMBDA, MU, K, SIGMA, Y, RIG, VIS

#%% Normalization
# Similar to Normalization.f90 routine
# Defines and normalizes the model parameters
# TODO: Change inputs to generic format
def normalization(R, RK, RHO, LAMBDA, MU, K, SIGMA, Y, RIG, VIS):

    print('Building the model...')
    
    # Constants
    pi  = np.pi
    gnt = 0.667408e-10 # Newton's gravity constant.  m^3/Kg s^2

    # TODO: Add functionality for PREM layers etc. For now only compatible with PlanetProfile
    '''
    if params['mode']: # PREM layering
        LT = lth        # Thickness of Lithosphere
        r[0] = ec       # Radius of the core mantle boundary
        r[1] = ea - LT  # Radius of the litho-mantle boundary
        r[2] = ea       # Radius of the earth

        # Thickness of each mantle layer (meters)
        delta = (r[1:] - r[0])

    elif params['mode'] == 2: # User model
        r   = np.flip(MODEL[:, 0])   # Radius
        rho = np.flip(MODEL[:, 0])   # density
        rig = np.flip(MODEL[:, 0])   # rigidity
        vis = np.flip(MODEL[:, 0])   # viscosity
        
        LT = r[0:-1] - r[1:]         # Lithospheric thickness (m)
    '''
    # Mass of model; Integrated mass; gravity
    # Gravity = GM(r) / r^2
    # M(r) - mass enclosed within r
    # MASS = rho (layer) * volume (layer) (integrate over depth)
    # MASS = 4/3 pi sigma(rho (r1^3 - r2^3))
    MASS    = np.zeros((len(R), ), dtype=np.float64)
    GRAVITY = np.zeros((len(R), ), dtype=np.float64)
    for index in range(len(R)):
        if not index == len(R) - 1:
            MASS[index] = np.sum(RHO[index:-1] * (np.power(R[index:-1], 3) - np.power(R[index+1:], 3))) * 4 * pi / 3

            GRAVITY[index] = gnt * MASS[index] / np.power(R[index], 2)

        else:
            MASS[index] = RHO[index] * np.power(R[index-1], 3) * 4 * pi / 3

            GRAVITY[index] = gnt * MASS[index] / np.power(R[index-1], 2)
           
    # Normalization
    R   = R / R[0]
    RK  = RK / RK[0]
    RHO = R / RHO[0]
    RIG = RIG / RIG[0]
    
    MU     = MU / MU[0]
    LAMBDA = LAMBDA / LAMBDA[0]

    # Reference time (1 ka)
    t0 = 1000.0 * 365.250 * 24.0 * 3600.0

    # Normalized viscosity
    VIS = VIS / t0 / VIS[0]

    # Normalized MASS
    MASS = MASS / MASS[0]

    # Normalized layer thicness
    DELTA = np.append(R[0:-1] - R[1:], 0.0)

    # Normalized Newton constant
    GNT = gnt * RHO[0] * RHO[0] * R[0] * R[0] / RIG[0]

    # Normalized <<a>> constants
    AC = (4 * pi / 3) * RHO * gnt
  
    return R, RK, RHO, RIG, LAMBDA, MU, VIS, DELTA, GNT, MASS, AC, GRAVITY
