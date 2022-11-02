
using SpecialFunctions
using LinearAlgebra

function build_model(r_in, rho_in, mu_in, eta_in, rheology, params)

    # Define some constants

    Gnwt  = BigFloat("6.674e-11");

    # Check consistency of input argument size

    nla = length(r_in);

    for arg=( r_in, rho_in, mu_in, eta_in, rheology )
        if (length(arg) != nla)
            error("Inconsistent argument sizes");
        end
    end

    if (size(params) != (nla,2)) 
        error("Inconsistent argument sizes");
    end

    # Set model parameters 

    r     = Array{BigFloat}( undef, nla );
    rho   = Array{BigFloat}( undef, nla );
    mu    = Array{BigFloat}( undef, nla );
    eta   = Array{BigFloat}( undef, nla );
    rpar  = Array{BigFloat}( undef, nla, 2 );
    
    r        .= BigFloat.( r_in   );
    rho      .= BigFloat.( rho_in );    
    mu       .= BigFloat.( mu_in  );    
    eta      .= BigFloat.( eta_in );    
    rpar     .= BigFloat.( params );

    # Parse the rheology

    rheol=Array{Int64}(undef,0)
    for rcode=rheology
        if typeof(rcode)==String
            if lowercase(rcode)=="fluid"
                push!(rheol,0)
            elseif lowercase(rcode)=="elastic"
                push!(rheol,1)
            elseif lowercase(rcode)=="maxwell"
                push!(rheol,2)
            elseif lowercase(rcode)=="newton"
                push!(rheol,3)
            elseif lowercase(rcode)=="kelvin"
                push!(rheol,4)
            elseif lowercase(rcode)=="burgers"
                push!(rheol,5)
            elseif lowercase(rcode)=="andrade"
                push!(rheol,6)
            else
                error("Unknown rheology name: $rcode")
            end
        else
            if( (rcode>=0) && (rcode<=6) )
                push!(rheol,rcode)
            else
                error("Invalid rheology code: $rcode")
            end
        end
    end

    # Compute gamma(alpha+1) for the Andrade layers

    for i=1:nla
        if rheol[i]==6
            rpar[i,2]=gamma(rpar[i,1]+1);
        end
    end

    # Build model

    gra    = Array{BigFloat}( undef, nla );
    mlayer = Array{BigFloat}( undef, nla );

    # ----- Compute mass of the planet

    mlayer[1] = rho[1] * r[1]^3
    for i=2:nla
        mlayer[i] = rho[i] * ( r[i]^3 - r[i-1]^3 )
    end
    mlayer = mlayer * BigFloat("4") / BigFloat("3") * pi
    
    mass = sum( mlayer )

    # ----- Compute gravity at the interface boundaries

    for i=1:nla
        gra[i] = Gnwt * sum( mlayer[1:i] ) / r[i]^2
    end

    # Normalization

    # ----- Define reference scales
 
    r0    = r[end]
    rho0  = rho[2]
    mu0   = maximum(mu)
    t0    = BigFloat("1000") * BigFloat("365.25") * BigFloat("24") * BigFloat("3600")
    eta0  = mu0 * t0
    mass0 = rho0 * r0^3

    # ------- Normalize model parameters

    r      = r   / r0
    rho    = rho / rho0
    mu     = mu  / mu0
    eta    = eta / eta0

    # ------- Normalize Newton constant

    G      = Gnwt * rho0^2 * r0^2 / mu0

    # ------- Normalized mass

    mass   = mass / mass0
    mlayer = mlayer / mass0

    # ------- Normalized gravity at internal interfaces

    gra    = gra * r0^2 * (G/Gnwt) / mass0

    # Return the normalized model

    return (r=r, rho=rho, mu=mu, eta=eta, rheol=rheol, rpar=rpar, G=G, gra=gra, mass=mass)

end


function salzer_weights(order)

    m = order
    zeta = zeros( BigFloat, 2*m )

    for k=1:(2*m)
        j1 = floor( Int64, (k+1)/2 )
        j2 = minimum( (k, m) )
        for j in j1:j2
            fattm = factorial(BigFloat(m))
            q1 = binomial(m, j)
            q2 = binomial(2*j, j)
            q3 = binomial(j, k-j)
            zeta[k] = zeta[k] + BigFloat(j)^(m+1) / BigFloat(fattm) * q1 * q2 * q3
        end
        if ((m+k)%2) != 0
            zeta[k] = -zeta[k]
        end

    end

    return zeta

end

function complex_rigidity(s,mu,eta,code,par)

    if code==1                      # Elastic
        mu_s = mu
    elseif code==2                  # Maxwell
        mu_s = mu * s / ( s + mu/eta )
    elseif code==3                  # Newton
        mu_s = eta * s
    elseif code==4                   # Kelvin
        mu_s = mu + eta * s
    elseif code==5                   # Burgers
        mu2  = par[1] * mu
        eta2 = par[2] * eta
        mu_s = mu * s * ( s + mu2/eta2 ) / 
	       ( s^2 + s*(mu/eta + (mu+mu2)/eta2) + (mu*mu2)/(eta*eta2) )   
    elseif code==6                   # Andrade
        alpha = par[1]
        gam   = par[2]
        mu_s  = 1/mu + 1/(eta*s) + gam * (1/mu) * (s*eta/mu)^(-alpha)
        mu_s  = 1/mu_s
    else
        error("Invalid rheology code: $(code)")
    end

    return mu_s

end

function direct_matrix(n,r,rho,mu,gra,G)

    a1 = 2*n + 3             # 2n+3
    a2 = n + 1               # n+1
    a3 = n + 3               # n+3
    a4 = n^2 - n - 3         # n^2-n-3
    a5 = n + 2               # n+2
    a6 = n - 1               # n-1
    a7 = 2 * n + 1           # 2n+1
    a8 = 2 * n - 1           # 2n-1
    a9 = 2 - n               # 2-n
    a10= n^2 + 3 * n - 1     # n^2+3n-1
    a11= n^2 - 1             # n^2-1

    Y = zeros(Complex{BigFloat},6,6)
 
    Y[1,1] = n/(2 * a1) * r^(n+1)
    Y[2,1] = a3 / ( 2 * a1 * a2 ) * r^(n+1) 
    Y[3,1] = ( n * rho * gra * r + 2 * a4 * mu ) / ( 2 * a1 ) * r^n
    Y[4,1] = n * a5 / ( a1 * a2 ) * mu * r^n
    Y[6,1] = 2 * pi * G * rho * n / a1 * r^(n+1)

    Y[1,2] = r^(n-1)
    Y[2,2] = 1 / n * r^(n-1)
    Y[3,2] = ( rho * gra * r + 2 * a6 * mu ) * r^(n-2)
    Y[4,2] = 2 * a6 / n * mu * r^(n-2)
    Y[6,2] = 4 * pi * G * rho * r^(n-1)

    Y[3,3] = rho * r^n
    Y[5,3] = r^n
    Y[6,3] = a7 * r^(n-1)

    Y[1,4] = a2 / ( 2 * a8 ) * r^(-n)
    Y[2,4] = a9 / ( 2 * n * a8 ) * r^(-n)
    Y[3,4] = ( a2 * rho * gra * r - 2 * a10 * mu ) / ( 2 * a8 ) * r^(-n-1)
    Y[4,4] = a11 / ( n * a8 ) * mu * r^(-n-1)
    Y[6,4] = 2 * pi * G * rho * a2 / a8 * r^(-n)

    Y[1,5] = r^(-n-2)
    Y[2,5] = - 1.0 / a2 * r^(-n-2)
    Y[3,5] = ( rho * gra * r - 2 * a5 * mu ) * r^(-n-3)
    Y[4,5] = 2 * a5 / a2 * mu * r^(-n-3)
    Y[6,5] = 4 * pi * G * rho * r^(-n-2)
    
    Y[3,6] = rho * r^(-n-1)
    Y[5,6] = r^(-n-1)

    return Y

end



function inverse_matrix(n,r,rho,mu,gra,G)

    a1 = 2 * n + 1           # 2n+1
    a2 = 2 * n - 1           # 2n-1
    a3 = n + 1               # n+1
    a4 = n - 1               # n-1
    a5 = n + 2               # n+2
    a6 = n + 3               # n+3
    a7 = n^2 + 3 * n - 1     # n^2+3n-1
    a8 = n^2 - n - 3         # n^2-n-3
    a9 = 2 - n               # 2-n
    a10= n^2 - 1             # n^2-1
    a11= 2 * n + 3           # 2n+3

    Yinv = zeros(Complex{BigFloat},6,6)
    D    = zeros(Complex{BigFloat},6,6)

    D[1,1] = a3 * r^(-(n+1))
    D[2,2] = a3 * n / ( 2 * a2 ) * r^(-(n-1))
    D[3,3] = -r^(-(n-1))
    D[4,4] = n * r^n
    D[5,5] = n * a3 / ( 2 * a11 ) * r^(n+2)
    D[6,6] = r^(n+1)

    D = D / a1
 
    Yinv[1,1] =   rho * gra * r / mu - 2 * a5
    Yinv[2,1] = - rho * gra * r / mu + 2 * a7 / a3
    Yinv[3,1] =   4 * pi * G * rho
    Yinv[4,1] =   rho * gra * r / mu + 2 * a4
    Yinv[5,1] = - rho * gra * r / mu - 2 * a8 / n
    Yinv[6,1] =   4 * pi * G * rho * r

    Yinv[1,2] =   2 * n * a5
    Yinv[2,2] = - 2 * a10
    Yinv[4,2] =   2 * a10
    Yinv[5,2] = - 2 * n * a5

    Yinv[1,3] = - r / mu
    Yinv[2,3] =   r / mu
    Yinv[4,3] = - r / mu
    Yinv[5,3] =   r / mu

    Yinv[1,4] =   n * r / mu
    Yinv[2,4] =   a9 * r / mu
    Yinv[4,4] = - a3 * r / mu
    Yinv[5,4] =   a6 * r / mu
    
    Yinv[1,5] =   rho * r / mu
    Yinv[2,5] = - rho * r / mu
    Yinv[4,5] =   rho * r / mu
    Yinv[5,5] = - rho * r / mu
    Yinv[6,5] =   a1

    Yinv[3,6] = - 1
    Yinv[6,6] = - r

    Yinv = D * Yinv

    return Yinv

end

function fluid_core_bc(n,r,rho,gra,G)

    b = zeros( BigFloat, 6,3 )

    b[1,1] = - 1 / gra * (r^n)
    b[1,3] =   1

    b[2,2] =   1

    b[3,3] =   rho * gra

    b[5,1] =   r^n

    b[6,1] =   2 * ( n - 1 ) * r^(n-1)
    b[6,3] =   4 * pi * G * rho

    return b

end

function surface_bc(n,r,gra,iload,G)

    bs = zeros(BigFloat, 3)
    
    kappa = ( 2 * n + 1) / ( 4 * pi * r^2 ) 
    if iload==1
        bs[1] = - gra * kappa
    end
    bs[3] = -4 * pi * G * kappa

    return bs

end

function parse_loadtype(loadtype)
    # Set load type ('tidal' or 'loading')

    if lowercase(loadtype)=="tidal"
        iload=0
    elseif lowercase(loadtype)=="loading"
        iload=1
    else
        error("Unknown load type '$(loadtype)'")
    end

    return iload

end

function parse_outputtype(output)
    # set output LN type ('real' or 'complex')

    if lowercase(output)=="real"
        itype=1
    elseif lowercase(output)=="complex"
        itype=2
    elseif lowercase(output)=="rate"
        itype=3
    else
        error("Unknown LN mode '$(output)'" )
    end

    return itype
end    

function parse_histtype( loadfcn )
     # set load fcn type ('step' or 'ramp')
    
     if lowercase(loadfcn)=="step"
        ihist=1
    elseif lowercase(loadfcn)=="ramp"
        ihist=2
    else
        error("Unknown load time-history: '$(loadfcn)'" )
    end
    return ihist
end

function love_numbers_sampler(n,s,iload,model)

    r     = model.r
    rho   = model.rho
    mu    = model.mu
    eta   = model.eta
    rheol = model.rheol
    rpar  = model.rpar
    G     = model.G
    mass  = model.mass
    gra   = model.gra

    #
    # ---- Build the propagator product
    #
    nla = length(rho)
    lam = BigFloat.( Matrix(I, 6,6) )

    for i=nla:-1:2
        mu_s = complex_rigidity(s,mu[i],eta[i],rheol[i],rpar[i,:])
        Ydir = direct_matrix( n,r[i  ],rho[i],mu_s,gra[i  ],G)
        Yinv = inverse_matrix(n,r[i-1],rho[i],mu_s,gra[i-1],G)
        lam = lam * (Ydir * Yinv)
    end
    
    if rheol[1]==0
        bc = fluid_core_bc(n,r[1],rho[1],gra[1],G)
    else
        mu_s = complex_rigidity(s,mu[1],eta[1],rheol[1],rpar[1,:])
        Ydir = direct_matrix(n,r[1],rho[1],mu_s,gra[1],G) 
        bc   = Ydir[:,1:3]
    end

    bs = surface_bc(n,r[nla],gra[nla],iload,G)

    # ---- Compute the 'R' and 'Q' arrays
    
    prod = lam * bc

    rr = Array{Complex{BigFloat}}(undef,3,3)
    qq = Array{Complex{BigFloat}}(undef,3,3)
    
    rr[1,:] = prod[3,:]
    rr[2,:] = prod[4,:]
    rr[3,:] = prod[6,:]

    qq[1,:] = prod[1,:]
    qq[2,:] = prod[2,:]
    qq[3,:] = prod[5,:]

    bb = rr\bs 
    x  = qq * bb

    hh = x[1] * mass / r[nla]
    ll = x[2] * mass / r[nla]
    kk = ( - 1 - x[3] * mass / ( r[nla] * gra[nla] ) )

    return hh, ll, kk

end

function love_numbers(model,degrees,timesteps,loadtype,loadfcn,tau,output,order; verbose=false)

    iota = Complex{BigFloat}(1im)

    r, rho, mu, eta, rheol, rpar, G, gra, mass = model

    iload = parse_loadtype( loadtype )
    itype = parse_outputtype( output )

    if( itype != 2 )
        ihist = parse_histtype( loadfcn )
    else
        ihist = 0
    end

    nt   = length(timesteps)
    ndeg = length(degrees)

    tstp = BigFloat.( timesteps )

    # ------- Allocate output arrays

    h_love = zeros( Complex{BigFloat}, ndeg, nt )
    l_love = zeros( Complex{BigFloat}, ndeg, nt )
    k_love = zeros( Complex{BigFloat}, ndeg, nt )
 
    t1 = time();

    # ------- Compute LNs

    idx_n = 0
    idx_t = 0

    if ( itype==1 ) || ( itype==3 )
        zeta = salzer_weights(order)
        for idx_n=1:ndeg
            n = degrees[idx_n]
            for idx_t=1:nt
                t = tstp[idx_t]
                f = log(big"2") / t
                for ik in 1:2*order
                    s = f * ik
                    hh,ll,kk = love_numbers_sampler(n,s,iload,model)
                    if ihist==1
                        fh = (big"1")/s
                    elseif ihist==2
                        fh = ( 1 - exp( -s * tau ) ) / (tau * s^2)
                    end
                    if( itype==3 )
                        h_love[ idx_n, idx_t ] += fh * (s * hh) * zeta[ik] * f
                        l_love[ idx_n, idx_t ] += fh * (s * ll) * zeta[ik] * f
                        k_love[ idx_n, idx_t ] += fh * (s * kk) * zeta[ik] * f
                    else
                        h_love[ idx_n, idx_t ] += fh * hh * zeta[ik] * f
                        l_love[ idx_n, idx_t ] += fh * ll * zeta[ik] * f
                        k_love[ idx_n, idx_t ] += fh * kk * zeta[ik] * f
                    end
                end
            end
            if verbose
                t2 = time();
                println( "Harmonic degree n = $(n) ( $(t2-t1) s)" );
                t1 = t2
            end
        end
    elseif  itype==2
        for idx_n=1:ndeg
            for idx_t=1:nt
                n = degrees[idx_n]
                t = tstp[idx_t]
                omega = big"2" * pi / t
                s     = iota * omega
                hh,ll,kk = love_numbers_sampler(n,s,iload,model)
                h_love[ idx_n, idx_t ] = hh
                l_love[ idx_n, idx_t ] = ll
                k_love[ idx_n, idx_t ] = kk
            end
            if verbose
                t2 = time();
                println( "Harmonic degree n = $(n) ( $(t2-t1) s)" );
                t1 = t2
            end
        end
    end

    if( itype!=2 )
        h_love = real.(h_love)
        l_love = real.(l_love)
        k_love = real.(k_love)
    end

    return h_love, l_love, k_love

end
