
"""
Created by stevertaylor
Copyright (c) 2014 Stephen R. Taylor

Code contributions by Rutger van Haasteren (piccard) and Justin Ellis (PAL/PAL2).

"""

from __future__ import division
import numpy as np
from numpy import *
import os
import math
from scipy import integrate
from scipy.integrate import odeint
from scipy import optimize
from scipy import constants as sc
from numpy import random
from scipy import special as ss
from scipy import linalg as sl
from scipy.interpolate import interp1d
from pkg_resources import resource_filename, Requirement
import numexpr as ne
import optparse
import ephem
from ephem import *

day = 24 * 3600
year = 365.25 * day

SOLAR2S = sc.G / sc.c**3 * 1.98855e30
KPC2S = sc.parsec / sc.c * 1e3
MPC2S = sc.parsec / sc.c * 1e6


def sumTermCovarianceMatrix_fast(tm, fL, gam):
    """
    Calculate the power series expansion for the Hypergeometric
    function in the standard power-law covariance matrix. This
    version uses the Python package numexpr and is much faster
    than using numpy. For now it is hardcoded to use only the 
    first 3 terms.
    
    @param tm: Matrix of time lags in years
    @param fL: Low frequency cutoff
    @param gam: Power Law spectral index
    """
    
    x = 2*np.pi*fL*tm

    sum = ne.evaluate("1/(1-gam) - x**2/(2*(3-gam)) + x**4/(24*(5-gam))")

    return sum


def makeTimeGrid(psra, psrb):
    """
    Construct time-domain DM-variation
    covariance matrix. 

    @param psra: object for pulsar 'a'
    @param psrb: object for pulsar 'b'

    @return: Cdm: Time-lag grid

    """

    ta, tb = np.meshgrid(psra.toas, psrb.toas)  
    tm = np.abs(ta-tb).astype(np.float64)/365.25
  
    return tm


def makeRedTDcov(Ared, gam_red, tm):
    """
    Construct time-domain red-noise
    covariance matrix. 

    @param Ared: Red-noise spectral amplitude
    @param gam_red: Red-noise spectral slope
    @param tm: time-lag matrix

    @return: Cdm: Red-noise covariance matrix

    """

    Tspan = tm.max()
    fL = 1/(100.0*Tspan)
    xgrid = 2.0*np.pi*fL*tm
    
    Cred = ( (Ared**2.0)*(fL**(1.0-gam_red)) / (12.0*np.pi**2.0) ) * \
      ((ss.gamma(1.0-gam_red)*np.sin(np.pi*gam_red/2.)*ne.evaluate("xgrid**(gam_red-1.0)"))
       - sumTermCovarianceMatrix_fast(tm, fL, gam_red))

    Cred *= ((365.25*86400.0)**2.0)

    return Cred


def makeDmTDcov(psr, Adm, gam_dm, tm):
    """
    Construct time-domain DM-variation
    covariance matrix. 

    @param psr: pulsar object
    @param Adm: DM-variation spectral amplitude
    @param gam_dm: DM-variation spectral slope
    @param tm: time-lag matrix

    @return: Cdm: DM covariance matrix

    """

    Tspan = tm.max()
    fL = 1/(100.0*Tspan)
    xgrid = 2.0*np.pi*fL*tm
    
    K = 2.41*10.0**(-16.0)
    Dm = 1.0/(K*(psr.obs_freqs*1e6)**2.0)
    DmA,DmB = np.meshgrid(Dm,Dm)
    DmGrid = DmA*DmB
   
    Cdm = ( (Adm**2.0)*(fL**(1.0-gam_dm)) / (12.0*np.pi**2.0) ) * \
      ((ss.gamma(1-gam_dm)*np.sin(np.pi*gam_dm/2)*ne.evaluate("xgrid**(gam_dm-1)"))
       - sumTermCovarianceMatrix_fast(tm, fL, gam_dm))

    Cdm *= ((365.25*86400.0)**2.0)
    Cdm = np.multiply(DmGrid,Cdm)

    return Cdm


def createFourierDesignmatrix_red(t, nmodes, freq=False,
                                  pshift=False, Tspan=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    @param t: vector of time series in seconds
    @param nmodes: number of fourier coefficients to use
    @param freq: option to output frequencies
    @param pshift: option to add random phase shift
    @param Tspan: option to some other Tspan

    @return: F: fourier design matrix
    @return: f: Sampling frequencies (if freq=True)

    """

    N = len(t)
    F = np.zeros((N, 2*nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    fqs = np.linspace(1/T, nmodes/T, nmodes)
    
    if pshift:
        ranphase = np.random.uniform(0.0, 2.0*np.pi, nmodes)
    elif not pshift:
        ranphase = np.zeros(nmodes)

    # The sine/cosine modes
    ct = 0
    for ii in range(0, 2*nmodes-1, 2):

        if pshift:
            F[:,ii] = np.cos(2*np.pi*fqs[ct]*t + ranphase[ct])
            F[:,ii+1] = np.sin(2*np.pi*fqs[ct]*t + ranphase[ct])
        elif not pshift:
            F[:,ii] = np.cos(2*np.pi*fqs[ct]*t)
            F[:,ii+1] = np.sin(2*np.pi*fqs[ct]*t)
            
        ct += 1

    if freq:
        return F, fqs, ranphase
    else:
        return F, ranphase

def createFourierDesignmatrix_dm(t, nmodes, obs_freqs, freq=False, Tspan=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    @param t: vector of time series in seconds
    @param nmodes: number of fourier coefficients to use
    @param freq: option to output frequencies
    @param Tspan: option to some other Tspan
    @param pbs_freqs: pulsar observing frequencies

    @return: F: fourier design matrix
    @return: f: Sampling frequencies (if freq=True)

    """

    N = len(t)
    F = np.zeros((N, 2*nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    fqs = np.linspace(1/T, nmodes/T, nmodes)

    # compute the DM-variation vectors
    K = 2.41 * 10.0**(-16.0)
    Dm = 1.0/(K * obs_freqs**2.0) # ssbfreqs already in Hz

    # The sine/cosine modes
    ct = 0
    for ii in range(0, 2*nmodes-1, 2):
        
        F[:,ii] = np.multiply(np.cos(2*np.pi*fqs[ct]*t),Dm)
        F[:,ii+1] = np.multiply(np.sin(2*np.pi*fqs[ct]*t),Dm)
        ct += 1
    
    if freq:
        return F, fqs
    else:
        return F

def createFourierDesignmatrix_eph(t, nmodes, psr_locs, freq=False, Tspan=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    @param t: vector of time series in seconds
    @param nmodes: number of fourier coefficients to use
    @param psr_locs: phi and theta coordinates of pulsar
    @param Tspan: option to some other Tspan

    @return: F: fourier design matrix along each positional basis vector
    @return: f: Sampling frequencies (if freq=True)

    """

    N = len(t)
    Fx = np.zeros((N, 2*nmodes))
    Fy = np.zeros((N, 2*nmodes))
    Fz = np.zeros((N, 2*nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    fqs = np.linspace(1/T, nmodes/T, nmodes)

    # define the pulsar position vector
    phi = psr_loc[0]
    theta = np.pi/2. - psr_locs[1]
    x = np.tile(np.sin(theta)*np.cos(phi), 2*nmodes)
    y = np.tile(np.sin(theta)*np.sin(phi), 2*nmodes)
    z = np.tile(np.cos(theta), 2*nmodes)

    # The sine/cosine modes
    ct = 0
    for ii in range(0, 2*nmodes-1, 2):
        
        Fx[:,ii] = np.cos(2*np.pi*fqs[ct]*t)
        Fx[:,ii+1] = np.sin(2*np.pi*fqs[ct]*t)
        Fy[:,ii] = np.cos(2*np.pi*fqs[ct]*t)
        Fy[:,ii+1] = np.sin(2*np.pi*fqs[ct]*t)
        Fz[:,ii] = np.cos(2*np.pi*fqs[ct]*t)
        Fz[:,ii+1] = np.sin(2*np.pi*fqs[ct]*t)
        ct += 1

    Fx = np.dot(Fx,x)
    Fy = np.dot(Fy,y)
    Fz = np.dot(Fz,z)
    
    if freq:
        return Fx, Fy, Fz, fqs
    else:
        return Fx, Fy, Fz

def quantize_fast(times, dt=1.0, calci=False):
    """ Adapted from libstempo: produce the quantisation matrix fast """
    isort = np.argsort(times)
    
    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]
    
    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])
    
    t = np.array([np.mean(times[l]) for l in bucket_ind],'d')
    
    U = np.zeros((len(times),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1
    
    rv = (t, U)

    if calci:
        Ui = ((1.0/np.sum(U, axis=0)) * U).T
        rv = (t, U, Ui)

    return rv


def quantize_split(times, flags, dt=1.0, calci=False):
    """
    As quantize_fast, but now split the blocks per backend. Note: for
    efficiency, this function assumes that the TOAs have been sorted by
    argsortTOAs. This is _NOT_ checked.
    """
    isort = np.arange(len(times))
    
    bucket_ref = [times[isort[0]]]
    bucket_flag = [flags[isort[0]]]
    bucket_ind = [[isort[0]]]
    
    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt and flags[i] == bucket_flag[-1]:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_flag.append(flags[i])
            bucket_ind.append([i])
    
    t = np.array([np.mean(times[l]) for l in bucket_ind],'d')
    
    U = np.zeros((len(times),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1
    
    rv = (t, U)

    if calci:
        Ui = ((1.0/np.sum(U, axis=0)) * U).T
        rv = (t, U, Ui)

    return rv


def argsortTOAs(toas, flags, which=None, dt=1.0):
    """
    Return the sort, and the inverse sort permutations of the TOAs, for the
    requested type of sorting
    NOTE: This one is _not_ optimized for efficiency yet (but is done only once)
    :param toas:    The toas that are to be sorted
    :param flags:   The flags that belong to each TOA (indicates sys/backend)
    :param which:   Which type of sorting we will use (None, 'jitterext', 'time')
    :param dt:      Timescale for which to limit jitter blocks, default [10 secs]
    :return:    perm, perminv       (sorting permutation, and inverse)
    """
    
    if which is None:
        isort = slice(None, None, None)
        iisort = slice(None, None, None)
    elif which == 'time':
        isort = np.argsort(toas, kind='mergesort')
        iisort = np.zeros(len(isort), dtype=np.int)
        for ii, p in enumerate(isort):
            iisort[p] = ii
    elif which == 'jitterext':
        tave, Umat = quantize_fast(toas, dt)

        isort = np.argsort(toas, kind='mergesort')
        uflagvals = list(set(flags))

        for cc, col in enumerate(Umat.T):
            for flagval in uflagvals:
                flagmask = (flags[isort] == flagval)
                if np.sum(col[isort][flagmask]) > 1:
                    # This observing epoch has several TOAs
                    colmask = col[isort].astype(np.bool)
                    epmsk = flagmask[colmask]
                    epinds = np.flatnonzero(epmsk)
                    
                    if len(epinds) == epinds[-1] - epinds[0] + 1:
                        # Keys are exclusively in succession
                        pass
                    else:
                        # Sort the indices of this epoch and backend
                        # We need mergesort here, because it is stable
                        # (A stable sort keeps items with the same key in the
                        # same relative order. )
                        episort = np.argsort(flagmask[colmask], kind='mergesort')
                        isort[colmask] = isort[colmask][episort]
                else:
                    # Only one element, always ok
                    pass

        # Now that we have a correct permutation, also construct the inverse
        iisort = np.zeros(len(isort), dtype=np.int)
        for ii, p in enumerate(isort):
            iisort[p] = ii
    else:
        isort, iisort = np.arange(len(toas)), np.arange(len(toas))

    return isort, iisort


def checkTOAsort(toas, flags, which=None, dt=1.0):
    
    """
    Check whether the TOAs are indeed sorted as they should be according to the
    definition in argsortTOAs
    :param toas:    The toas that are supposed to be already sorted
    :param flags:   The flags that belong to each TOA (indicates sys/backend)
    :param which:   Which type of sorting we will check (None, 'jitterext', 'time')
    :param dt:      Timescale for which to limit jitter blocks, default [10 secs]
    :return:    True/False
    """
    
    rv = True
    if which is None:
        isort = slice(None, None, None)
        iisort = slice(None, None, None)
    elif which == 'time':
        isort = np.argsort(toas, kind='mergesort')
        if not np.all(isort == np.arange(len(isort))):
            rv = False
    elif which == 'jitterext':
        tave, Umat = quantize_fast(toas, dt)

        #isort = np.argsort(toas, kind='mergesort')
        isort = np.arange(len(toas))
        uflagvals = list(set(flags))

        for cc, col in enumerate(Umat.T):
            for flagval in uflagvals:
                flagmask = (flags[isort] == flagval)
                if np.sum(col[isort][flagmask]) > 1:
                    # This observing epoch has several TOAs
                    colmask = col[isort].astype(np.bool)
                    epmsk = flagmask[colmask]
                    epinds = np.flatnonzero(epmsk)
                    
                    if len(epinds) == epinds[-1] - epinds[0] + 1:
                        # Keys are exclusively in succession
                        pass
                    else:
                        # Keys are not sorted for this epoch/flag
                        rv = False
                else:
                    # Only one element, always ok
                    pass
    else:
        pass

    return rv


def checkquant(U, flags, uflagvals=None):
    
    """
    Check the quantization matrix for consistency with the flags
    :param U:           quantization matrix
    :param flags:       the flags of the TOAs
    :param uflagvals:   subset of flags that are not ignored
    
    :return:            True/False, whether or not consistent
    
    The quantization matrix is checked for three kinds of consistency:
    - Every quantization epoch has more than one observation
    - No quantization epoch has no observations
    - Only one flag is allowed per epoch
    """
    
    if uflagvals is None:
        uflagvals = list(set(flags))

    rv = True
    collisioncheck = np.zeros((U.shape[1], len(uflagvals)), dtype=np.int)
    for ii, flagval in enumerate(uflagvals):
        flagmask = (flags == flagval)

        Umat = U[flagmask, :]

        simepoch = np.sum(Umat, axis=0)
        if np.all(simepoch <= 1) and not np.all(simepoch == 0):
            rv = False
            #raise ValueError("quantization matrix contains non-jitter-style data")

        collisioncheck[:, ii] = simepoch

        # Check continuity of the columns
        for cc, col in enumerate(Umat.T):
            if np.sum(col > 2):
                # More than one TOA for this flag/epoch
                epinds = np.flatnonzero(col)
                if len(epinds) != epinds[-1] - epinds[0] + 1:
                    rv = False
                    print("WARNING: checkquant found non-continuous blocks")
                    #raise ValueError("quantization matrix epochs not continuous")
        

    epochflags = np.sum(collisioncheck > 0, axis=1)

    if np.any(epochflags > 1):
        rv = False
        print("WARNING: checkquant found multiple backends for an epoch")
        print epochflags
        #raise ValueError("Some observing epochs include multiple backends")

    if np.any(epochflags < 1):
        rv = False
        print("WARNING: checkquant found epochs without observations (eflags)")
        #raise ValueError("Some observing epochs include no observations... ???")

    obsum = np.sum(U, axis=0)
    if np.any(obsum < 1):
        rv = False
        print("WARNING: checkquant found epochs without observations (all)")
        #raise ValueError("Some observing epochs include no observations... ???")

    return rv


def quant2ind(U):
    
    """
    Convert the quantization matrix to an indices matrix for fast use in the
    jitter likelihoods
    :param U:       quantization matrix
    
    :return:        Index (basic slicing) version of the quantization matrix
    
    This function assumes that the TOAs have been properly sorted according to
    the proper function argsortTOAs above. Checks on the continuity of U are not
    performed
    """
    
    inds = np.zeros((U.shape[1], 2), dtype=np.int)
    
    for cc, col in enumerate(U.T):
        epinds = np.flatnonzero(col)
        inds[cc, 0] = epinds[0]
        inds[cc, 1] = epinds[-1]+1

    return inds


def quantreduce(U, eat, flags, calci=False):
    
    """
    Reduce the quantization matrix by removing the observing epochs that do not
    require any jitter parameters.
    :param U:       quantization matrix
    :param eat:     Epoch-averaged toas
    :param flags:   the flags of the TOAs
    :param calci:   Calculate pseudo-inverse yes/no
    
    :return     newU, jflags (flags that need jitter)
    """
    
    uflagvals = list(set(flags))
    incepoch = np.zeros(U.shape[1], dtype=np.bool)
    jflags = []
    for ii, flagval in enumerate(uflagvals):
        flagmask = (flags == flagval)
        
        Umat = U[flagmask, :]
        ecnt = np.sum(Umat, axis=0)
        incepoch = np.logical_or(incepoch, ecnt>1)

        if np.any(ecnt > 1):
            jflags.append(flagval)

    Un = U[:, incepoch]
    eatn = eat[incepoch]

    if calci:
        Ui = ((1.0/np.sum(Un, axis=0)) * Un).T
        rv = (Un, Ui, eatn, jflags)
    else:
        rv = (Un, eatn, jflags)

    return rv

def dailyAve(times, res, err, ecorr, dt=1, flags=None):
    # Does not work yet in NX01"

    isort = np.argsort(times)
    
    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]
    
    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])
    
    avetoas = np.array([np.mean(times[l]) for l in bucket_ind],'d')
    if flags is not None:
        aveflags = np.array([flags[l[0]] for l in bucket_ind])

    aveerr = np.zeros(len(bucket_ind))
    averes = np.zeros(len(bucket_ind))

   
    for i,l in enumerate(bucket_ind):
        M = np.ones(len(l))
        C = np.diag(err[l]**2) + np.ones((len(l), len(l))) * ecorr[l[0]]

        avr = 1/np.dot(M, np.dot(np.linalg.inv(C), M))
        aveerr[i] = np.sqrt(avr)
        averes[i] = avr * np.dot(M, np.dot(np.linalg.inv(C), res[l]))
 
        
    if flags is not None:
        return avetoas, averes, aveerr, aveflags
    else:
        return avetoas, aveerr, averes

def make_ecc_interpolant():
    
    """
    Make interpolation function from eccentricity file to
    determine number of harmonics to use for a given
    eccentricity.

    :returns: interpolant
    """
    
    pth = resource_filename(Requirement.parse('libstempo'),
                            'libstempo/ecc_vs_nharm.txt')

    fil = np.loadtxt(pth)

    return interp1d(fil[:,0], fil[:,1])

# get interpolant for eccentric binaries
ecc_interp = make_ecc_interpolant()

def get_edot(F, mc, e):
    
    """
    Compute eccentricity derivative from Taylor et al. (2015)

    :param F: Orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param e: Eccentricity of binary

    :returns: de/dt

    """

    # chirp mass
    mc *= SOLAR2S

    dedt = -304/(15*mc) * (2*np.pi*mc*F)**(8/3) * e * \
        (1 + 121/304*e**2) / ((1-e**2)**(5/2))

    return dedt

def get_Fdot(F, mc, e):
    
    """
    Compute frequency derivative from Taylor et al. (2015)

    :param F: Orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param e: Eccentricity of binary

    :returns: dF/dt

    """

    # chirp mass
    mc *= SOLAR2S

    dFdt = 48 / (5*np.pi*mc**2) * (2*np.pi*mc*F)**(11/3) * \
        (1 + 73/24*e**2 + 37/96*e**4) / ((1-e**2)**(7/2))

    return dFdt


def get_gammadot(F, mc, q, e):
    
    """
    Compute gamma dot from Barack and Cutler (2004)

    :param F: Orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param q: Mass ratio of binary
    :param e: Eccentricity of binary

    :returns: dgamma/dt

    """

    # chirp mass
    mc *= SOLAR2S

    #total mass
    m = (((1+q)**2)/q)**(3/5) * mc

    dgdt = 6*np.pi*F * (2*np.pi*F*m)**(2/3) / (1-e**2) * \
        (1 + 0.25*(2*np.pi*F*m)**(2/3)/(1-e**2)*(26-15*e**2))

    return dgdt

def get_coupled_constecc_eqns(y, t, mc, e0):
    
    """
    Computes the coupled system of differential
    equations from Peters (1964) and Barack &
    Cutler (2004). This is a system of three variables:
    
    F: Orbital frequency [Hz]
    phase0: Orbital phase [rad]
    
    :param y: Vector of input parameters [F, e, gamma]
    :param t: Time [s]
    :param mc: Chirp mass of binary [Solar Mass]
    
    :returns: array of derivatives [dF/dt, dphase/dt]
    """
    
    F = y[0]
    phase = y[1]
    
    dFdt = get_Fdot(F, mc, e0)
    dphasedt = 2*np.pi*F
     
    return np.array([dFdt, dphasedt])


def get_coupled_ecc_eqns(y, t, mc, q):
    
    """
    Computes the coupled system of differential
    equations from Peters (1964) and Barack &
    Cutler (2004). This is a system of three variables:
    
    F: Orbital frequency [Hz]
    e: Orbital eccentricity
    gamma: Angle of precession of periastron [rad]
    phase0: Orbital phase [rad]
    
    :param y: Vector of input parameters [F, e, gamma]
    :param t: Time [s]
    :param mc: Chirp mass of binary [Solar Mass]
    :param q: Mass ratio of binary
    
    :returns: array of derivatives [dF/dt, de/dt, dgamma/dt, dphase/dt]
    """
    
    F = y[0]
    e = y[1]
    gamma = y[2]
    phase = y[3]
    
    #total mass
    m = (((1+q)**2)/q)**(3/5) * mc    
    
    dFdt = get_Fdot(F, mc, e)
    dedt = get_edot(F, mc, e)
    dgdt = get_gammadot(F, mc, q, e)
    dphasedt = 2*np.pi*F
     
    return np.array([dFdt, dedt, dgdt, dphasedt])


def solve_coupled_constecc_solution(F0, e0, phase0, mc, t):
    
    """
    Compute the solution to the coupled system of equations
    from from Peters (1964) and Barack & Cutler (2004) at 
    a given time.
    
    :param F0: Initial orbital frequency [Hz]
    :param mc: Chirp mass of binary [Solar Mass]
    :param t: Time at which to evaluate solution [s]
    
    :returns: (F(t), phase(t))
    
    """
    
    y0 = np.array([F0, phase0])

    y, infodict = odeint(get_coupled_constecc_eqns, y0, t, args=(mc,e0), full_output=True)
    
    if infodict['message'] == 'Integration successful.':
        ret = y
    else:
        ret = 0
    
    return ret


def solve_coupled_ecc_solution(F0, e0, gamma0, phase0, mc, q, t):
    
    """
    Compute the solution to the coupled system of equations
    from from Peters (1964) and Barack & Cutler (2004) at 
    a given time.
    
    :param F0: Initial orbital frequency [Hz]
    :param e0: Initial orbital eccentricity
    :param gamma0: Initial angle of precession of periastron [rad]
    :param mc: Chirp mass of binary [Solar Mass]
    :param q: Mass ratio of binary
    :param t: Time at which to evaluate solution [s]
    
    :returns: (F(t), e(t), gamma(t), phase(t))
    
    """
    
    y0 = np.array([F0, e0, gamma0, phase0])

    y, infodict = odeint(get_coupled_ecc_eqns, y0, t, args=(mc,q), full_output=True)
    
    if infodict['message'] == 'Integration successful.':
        ret = y
    else:
        ret = 0
    
    return ret


def get_an(n, mc, dl, h0, F, e):
    
    """
    Compute a_n from Eq. 22 of Taylor et al. (2015).
    
    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    
    :returns: a_n
    
    """
    
    # convert to seconds
    mc *= SOLAR2S
    dl *= MPC2S
    
    omega = 2 * np.pi * F

    if h0 is None:
        amp = n * mc**(5/3) * omega**(2/3) / dl
    elif h0 is not None:
        amp = n * h0 / 2.0
    
    ret = -amp * (ss.jn(n-2,n*e) - 2*e*ss.jn(n-1,n*e) +
                  (2/n)*ss.jn(n,n*e) + 2*e*ss.jn(n+1,n*e) -
                  ss.jn(n+2,n*e))

    return ret

def get_bn(n, mc, dl, h0, F, e):
    """
    Compute b_n from Eq. 22 of Taylor et al. (2015).
    
    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    
    :returns: b_n
    
    """
    
    # convert to seconds
    mc *= SOLAR2S
    dl *= MPC2S 
    
    omega = 2 * np.pi * F

    if h0 is None:
        amp = n * mc**(5/3) * omega**(2/3) / dl
    elif h0 is not None:
        amp = n * h0 / 2.0
        
    ret = -amp * np.sqrt(1-e**2) *(ss.jn(n-2,n*e) - 2*ss.jn(n,n*e) +
                  ss.jn(n+2,n*e)) 

    return ret

def get_cn(n, mc, dl, h0, F, e):
    
    """
    Compute c_n from Eq. 22 of Taylor et al. (2015).
    
    :param n: Harmonic number
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    
    :returns: c_n
    
    """
    
    # convert to seconds
    mc *= SOLAR2S
    dl *= MPC2S
    
    omega = 2 * np.pi * F

    if h0 is None:
        amp = n * mc**(5/3) * omega**(2/3) / dl
    elif h0 is not None:
        amp = n * h0 / 2.0
     
    ret = amp * ss.jn(n,n*e) / (n * omega)

    return ret

def calculate_splus_scross(nmax, mc, dl, h0, F, e, t, l0, gamma, gammadot, inc):
    
    """
    Calculate splus and scross summed over all harmonics. 
    This waveform differs slightly from that in Taylor et al (2015) 
    in that it includes the time dependence of the advance of periastron.
    
    :param nmax: Total number of harmonics to use
    :param mc: Chirp mass of binary [Solar Mass]
    :param dl: Luminosity distance [Mpc]
    :param F: Orbital frequency of binary [Hz]
    :param e: Orbital Eccentricity
    :param t: TOAs [s]
    :param l0: Initial eccentric anomoly [rad]
    :param gamma: Angle of periastron advance [rad]
    :param gammadot: Time derivative of angle of periastron advance [rad/s]
    :param inc: Inclination angle [rad]

    """ 
    '''if e < 0.001:
        n = np.array([2])
        nmax = 2
    else:
        n = np.arange(1, nmax)'''
    n = np.arange(1, nmax)

    # time dependent amplitudes
    an = get_an(n, mc, dl, h0, F, e)
    bn = get_bn(n, mc, dl, h0, F, e)
    cn = get_cn(n, mc, dl, h0, F, e)

    # time dependent terms
    omega = 2*np.pi*F
    gt = gamma + gammadot * t
    lt = l0 + omega * t

    # tiled phase
    phase1 = n * np.tile(lt, (nmax-1,1)).T
    phase2 = np.tile(gt, (nmax-1,1)).T
    #phasep = phase1 + 2*phase2
    #phasem = phase1 - 2*phase2

    # intermediate terms
    #sp = np.sin(phasem)/(n*omega-2*gammadot) + \
    #        np.sin(phasep)/(n*omega+2*gammadot)
    #sm = np.sin(phasem)/(n*omega-2*gammadot) - \
    #        np.sin(phasep)/(n*omega+2*gammadot)
    #cp = np.cos(phasem)/(n*omega-2*gammadot) + \
    #        np.cos(phasep)/(n*omega+2*gammadot)
    #cm = np.cos(phasem)/(n*omega-2*gammadot) - \
    #        np.cos(phasep)/(n*omega+2*gammadot)
    #
    #
    #splus_n = -0.5 * (1+np.cos(inc)**2) * (an*sp - bn*sm) + \
    #        (1-np.cos(inc)**2)*cn * np.sin(phase1)
    #scross_n = np.cos(inc) * (an*cm - bn*cp)

    sinp1 = np.sin(phase1)
    cosp1 = np.cos(phase1)
    sinp2 = np.sin(2*phase2)
    cosp2 = np.cos(2*phase2)

    sinpp = sinp1*cosp2 + cosp1*sinp2
    cospp = cosp1*cosp2 - sinp1*sinp2
    sinpm = sinp1*cosp2 - cosp1*sinp2
    cospm = cosp1*cosp2 + sinp1*sinp2

    # intermediate terms
    sp = sinpm/(n*omega-2*gammadot) + \
      sinpp/(n*omega+2*gammadot)
    sm = sinpm/(n*omega-2*gammadot) - \
      sinpp/(n*omega+2*gammadot)
    cp = cospm/(n*omega-2*gammadot) + \
      cospp/(n*omega+2*gammadot)
    cm = cospm/(n*omega-2*gammadot) - \
      cospp/(n*omega+2*gammadot)
    
    splus_n = -0.5 * (1+np.cos(inc)**2) * (an*sp - bn*sm) + \
            (1-np.cos(inc)**2)*cn * sinp1
    scross_n = np.cos(inc) * (an*cm - bn*cp)
        
    return np.sum(splus_n, axis=1), np.sum(scross_n, axis=1)


def fplus_fcross(psr, gwtheta, gwphi):
    """
    Compute gravitational-wave quadrupolar antenna pattern.

    :param psr: pulsar object
    :param gwtheta: Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]

    :returns: fplus, fcross
    """

    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)

    # unit vectors to GW source
    m = np.array([singwphi, -cosgwphi, 0.0])
    n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
    omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])
    
    # pulsar location
    ptheta = np.pi/2 - psr.psr_locs[1]
    pphi = psr.psr_locs[0]
    
    # use definition from Sesana et al 2010 and Ellis et al 2012
    phat = np.array([np.sin(ptheta)*np.cos(pphi), np.sin(ptheta)*np.sin(pphi),\
            np.cos(ptheta)])

    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))

    return fplus, fcross


def ecc_cgw_signal(psr, gwtheta, gwphi, mc, dist, h0, F, inc, psi, gamma0,
                   e0, l0, q, nmax=100, nset=None, pd=None, gpx=None, lpx=None,
                   periEv=True, psrTerm=False, tref=0, check=False, useFile=True,
                   epochTOAs=False, noEccEvolve=False, dummy_toas=None):
    
    """
    Simulate GW from eccentric SMBHB. Waveform models from
    Taylor et al. (2016) and Barack and Cutler (2004).

    WARNING: This residual waveform is only accurate if the
    GW frequency is not significantly evolving over the 
    observation time of the pulsar.

    :param psr: pulsar object
    :param gwtheta: Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    :param mc: Chirp mass of SMBMB [solar masses]
    :param dist: Luminosity distance to SMBMB [Mpc]
    :param h0: Dimensionless strain amplitude
    :param F: Orbital frequency of SMBHB [Hz]
    :param inc: Inclination of GW source [radians]
    :param psi: Polarization of GW source [radians]
    :param gamma0: Initial angle of periastron [radians]
    :param e0: Initial eccentricity of SMBHB
    :param l0: Initial mean anomaly [radians]
    :param q: Mass ratio of SMBHB
    :param nmax: Number of harmonics to use in waveform decomposition
    :param nset: Fix the number of harmonics to be injected
    :param pd: Pulsar distance [kpc]
    :param gpx: Pulsar-term gamm0 [radians]
    :param lpx: Pulsar-term l0 [radians]
    :param periEv: Evolve the position of periapsis [boolean] 
    :param psrTerm: Option to include pulsar term [boolean] 
    :param tref: Fiducial time at which initial parameters are referenced [s]
    :param check: Check if frequency evolves significantly over obs. time
    :param useFile: Use pre-computed table of number of harmonics vs eccentricity
    :param epochTOAs: Use epoch averaged TOAs to reduce the waveform evaluation time
    :param noEccEvolve: Prevent eccentricity evolving over Earth-pulsar separation
    :param dummy_toas: Instead of actual pulsar TOAs, just use some user-defined timestamps

    :returns: Vector of induced residuals
    """
    
    # define variable for later use
    cosgwtheta, cosgwphi = np.cos(gwtheta), np.cos(gwphi)
    singwtheta, singwphi = np.sin(gwtheta), np.sin(gwphi)
    sin2psi, cos2psi = np.sin(2*psi), np.cos(2*psi)

    # unit vectors to GW source
    m = np.array([singwphi, -cosgwphi, 0.0])
    n = np.array([-cosgwtheta*cosgwphi, -cosgwtheta*singwphi, singwtheta])
    omhat = np.array([-singwtheta*cosgwphi, -singwtheta*singwphi, -cosgwtheta])
    
    # pulsar location
    ptheta = np.pi/2 - psr.psr_locs[1]
    pphi = psr.psr_locs[0]
    
    # use definition from Sesana et al 2010 and Ellis et al 2012
    phat = np.array([np.sin(ptheta)*np.cos(pphi), np.sin(ptheta)*np.sin(pphi),\
            np.cos(ptheta)])

    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
    cosMu = -np.dot(omhat, phat)

    # get values from pulsar object
    if dummy_toas is None:
        if epochTOAs:
            toas = (psr.detsig_avetoas.copy() - tref)*86400.0
        elif not epochTOAs:
            toas = (psr.toas.copy() - tref)*86400.0
    elif dummy_toas is not None:
        toas = (dummy_toas.copy() - tref)*86400.0
    
    if check:
        # check that frequency is not evolving significantly over obs. time
        y = solve_coupled_ecc_solution(F, e0, gamma0, l0, mc, q,
                                       np.array([0.0,toas.max()]))
        
        # initial and final values over observation time
        Fc0, ec0, gc0, phic0 = y[0,:]
        Fc1, ec1, gc1, phic1 = y[-1,:]

        # observation time
        Tobs = 1/(toas.max()-toas.min())

        if np.abs(Fc0-Fc1) > 1/Tobs:
            print('WARNING: Frequency is evolving over more than one frequency bin.')
            print('F0 = {0}, F1 = {1}, delta f = {2}'.format(Fc0, Fc1, 1/Tobs))
    
    # get gammadot for earth term
    if not periEv:
        gammadot = 0.0
    else:
        gammadot = get_gammadot(F, mc, q, e0)

    if nset is not None:
        nharm = nset
    elif useFile:
        if e0 > 0.001 and e0 < 0.999:
            nharm = min(int(ecc_interp(e0)), nmax) + 1
        elif e0 <= 0.001:
            nharm = 3
        else:
            nharm = nmax
    else:
        nharm = nmax
    
    ##### earth term #####
    splus, scross = calculate_splus_scross(nharm, mc, dist, h0, F, e0,
                                           toas, l0, gamma0, gammadot, inc)
    
    ##### pulsar term #####
    if psrTerm:
       
        # convert units
        if pd is None:
            pd = p.h5Obj['pdist'].value
        pd *= KPC2S   # convert from kpc to seconds
    
        # get pulsar time
        tp = toas - pd * (1-cosMu)

        # solve coupled system of equations to get pulsar term values
        if noEccEvolve:
            y = solve_coupled_constecc_solution(F, e0, l0, mc,
                                            np.array([0.0, tp.min()]))
        elif not noEccEvolve:
            y = solve_coupled_ecc_solution(F, e0, gamma0, l0, mc, q,
                                           np.array([0.0, tp.min()]))
        
        # get pulsar term values
        if np.any(y):
            if noEccEvolve:
                Fp, lp = y[-1,:]
                ep = e0
                gp = gamma0
            elif not noEccEvolve:
                Fp, ep, gp, lp = y[-1,:]

            # get gammadot at pulsar term
            if not periEv:
                gammadotp = 0.0
            else:
                gammadotp = get_gammadot(Fp, mc, q, ep)

            if useFile:
                if ep > 0.001 and ep < 0.999:
                    nharm = min(int(ecc_interp(ep)), nmax) + 1
                elif ep <= 0.001:
                    nharm = 3
                else:
                    nharm = nmax
            else:
                nharm = nmax


            if gpx is None:
                gp_tmp = gp
            elif gpx is not None:
                gp_tmp = gpx

            if lpx is None:
                lp_tmp = lp
            elif lpx is not None:
                lp_tmp = lpx
            
            splusp, scrossp = calculate_splus_scross(nharm, mc, dist, h0, Fp, ep,
                                                     toas, lp_tmp, gp_tmp,
                                                     gammadotp, inc)

            rr = (fplus*cos2psi - fcross*sin2psi) * (splusp - splus) + \
                    (fplus*sin2psi + fcross*cos2psi) * (scrossp - scross)

        else:
            rr = np.zeros(len(toas))
            
    else:

        rr = - (fplus*cos2psi - fcross*sin2psi) * splus - \
                (fplus*sin2psi + fcross*cos2psi) * scross
         
    return rr


def BWMantennaPattern(rajp, decjp, raj, decj, pol):
    """Return the antenna pattern for a given source position and
    pulsar position
    :param rajp:    Right ascension pulsar (rad) [0,2pi]
    :param decj:    Declination pulsar (rad) [-pi/2,pi/2]
    :param raj:     Right ascension source (rad) [0,2pi]
    :param dec:     Declination source (rad) [-pi/2,pi/2]
    :param pol:     Polarization angle (rad) [0,pi]
    """

    Omega = np.array([-np.cos(decj)*np.cos(raj), \
                      -np.cos(decj)*np.sin(raj), \
                      -np.sin(decj)]).flatten()

    mhat = np.array([-np.sin(raj), np.cos(raj), 0]).flatten()
    nhat = np.array([-np.cos(raj)*np.sin(decj), \
                     -np.sin(decj)*np.sin(raj), \
                     np.cos(decj)]).flatten()

    p = np.array([np.cos(rajp)*np.cos(decjp), \
                  np.sin(rajp)*np.cos(decjp), \
                  np.sin(decjp)]).flatten()



    Fp = 0.5 * (np.dot(nhat, p)**2 - np.dot(mhat, p)**2) / (1 + np.dot(Omega, p))
    Fc = np.dot(mhat, p) * np.dot(nhat, p) / (1 + np.dot(Omega, p))

    return np.cos(2*pol)*Fp + np.sin(2*pol)*Fc


def bwmsignal(parameters, psr, antennaPattern='quad'):
    """
    Function that calculates the earth-term gravitational-wave burst-with-memory
    signal, as described in:
    Seto et al, van haasteren and Levin, phsirkov et al, Cordes and Jenet.
    This version uses the F+/Fx polarization modes, as verified with the
    Continuous Wave and Anisotropy papers. The rotation matrices were not very
    insightful anyway.
    parameter[0] = TOA time (MJD) the burst hits the earth
    parameter[1] = amplitude of the burst (strain h)
    parameter[2] = azimuthal angle (rad)    [0, 2pi]
    parameter[3] = cosine polar angle (rad) [-1, 1]
    parameter[4] = polarisation angle (rad) [0, pi]
    raj = Right Ascension of the pulsar (rad)
    decj = Declination of the pulsar (rad)
    t = timestamps where the waveform should be returned
    returns the waveform as induced timing residuals (seconds)
    """
    
    gwphi = np.array([parameters[2]])
    gwdec = np.array([np.pi/2-np.arccos(parameters[3])])
    gwpol = np.array([parameters[4]])

    if antennaPattern == 'quad':
        pol = BWMantennaPattern(psr.psr_locs[0].flatten(), psr.psr_locs[1].flatten(),
                                gwphi, gwdec, gwpol)
    elif antennaPattern == 'mono':
        pol = 1.0
    elif antennaPattern == 'absQuad':
        pol = np.absolute(BWMantennaPattern(psr.psr_locs[0].flatten(), psr.psr_locs[1].flatten(),
                                       gwphi, gwdec, gwpol))

    # Define the heaviside function
    heaviside = lambda x: 0.5 * (np.sign(x) + 1)

    # Return the time-series for the pulsar
    bwm = pol * (10**parameters[1]) * heaviside(psr.toas - parameters[0]) * \
            (psr.toas - parameters[0]) * 86400

    return bwm


def bwmsignal_psr(parameters, t):
    """
    Function that calculates the earth-term gravitational-wave burst-with-memory
    signal, as described in:
    Seto et al, van haasteren and Levin, phsirkov et al, Cordes and Jenet.

    This version only has a burst epoch and a strain in order to characterize a
    pulsar-term BWM signal.

    parameter[0] = TOA time (sec) the burst hits the earth
    parameter[1] = amplitude of the burst (strain h)
    parameter[2] = extra multiplier (typically -1 or 1, for sign of signal)

    t = timestamps where the waveform should be returned

    returns the waveform as induced timing residuals (seconds)

    """
    # Define the heaviside function
    heaviside = lambda x: 0.5 * (np.sign(x) + 1)

    s = np.sign(parameters[2])
    amp = 10**parameters[1]
    epoch = (parameters[0] - pic_T0) * pic_spd

    # Return the time-series for the pulsar
    return amp * s * heaviside(t - epoch) * (t - epoch)


def real_sph_harm(ll, mm, phi, theta):
    """
    The real-valued spherical harmonics
    (adapted from van Haasteren's piccard code)
    
    """
    
    if mm>0:
        ans = (1./math.sqrt(2)) * \
                (ss.sph_harm(mm, ll, phi, theta) + \
                ((-1)**mm) * ss.sph_harm(-mm, ll, phi, theta))
    elif mm==0:
        ans = ss.sph_harm(0, ll, phi, theta)
    elif mm<0:
        ans = (1./(math.sqrt(2)*complex(0.,1))) * \
                (ss.sph_harm(-mm, ll, phi, theta) - \
                ((-1)**mm) * ss.sph_harm(mm, ll, phi, theta))

    return ans.real


def SetupPriorSkyGrid(lmax):
    """
    Check whether these anisotropy coefficients correspond to a physical
    angular-distribution of the metric-perturbation quadratic
    expectation-value.
    
    """
    
    ngrid_phi = 40
    ngrid_costheta = 40
    
    phi = np.arange(0.0,2.0*np.pi,2.0*np.pi/ngrid_phi)
    theta = np.arccos(np.arange(-1.0,1.0,2.0/ngrid_costheta))

    xx, yy = np.meshgrid(phi,theta)

    harm_sky_vals = [[0.0]*(2*ll+1) for ll in range(lmax+1)]
    for ll in range(len(harm_sky_vals)):
        for mm in range(len(harm_sky_vals[ll])):
            harm_sky_vals[ll][mm] = real_sph_harm(ll,mm-ll,xx,yy)

    return harm_sky_vals

def PhysPrior(clm,harm_sky_vals):
    """
    Check whether these anisotropy coefficients correspond to a physical
    angular-distribution of the metric-perturbation quadratic
    expectation-value.
    
    """

    Pdist=0.
    for ll in range(len(harm_sky_vals)):
        for mm in range(len(harm_sky_vals[ll])):
            Pdist += clm[ ll**2 + mm ] * harm_sky_vals[ll][mm]

    if np.any(Pdist<0.)==True:
        return 'Unphysical'
    else:
        return 'Physical'


def SetupSkymapPlottingGrid(lmax,skypos):
    """
    Compute the real spherical harmonics
    on a sky-grid defined by healpy for
    plotting purposes.
    
    """
    
    harmvals = [[0.0]*(2*ll+1) for ll in range(lmax+1)]
    for ll in range(len(harmvals)):
        for mm in range(len(harmvals[ll])):
            harmvals[ll][mm] = real_sph_harm(ll,mm-ll,
                                             skypos[:,1],skypos[:,0])

    return harmvals

def GWpower(clm,harmvals):
    """
    Construct the GW power flowing into each pixel
    
    """

    Pdist=0.
    for ll in range(len(harmvals)):
        for mm in range(len(harmvals[ll])):
            Pdist += clm[ ll**2 + mm ] * harmvals[ll][mm]
    
    return Pdist

'''
def singlePsrLL(psr, Amp=5e-14, gam_gwb=13./3.):
    ######################################
    # Calculate the GW covariance matrix
    ######################################
    ta, tb = np.meshgrid(psr.toas, psr.toas)  
    tgrid = np.abs(ta-tb).astype(np.float64)/365.25
    Tspan = tgrid.max()
    fL = 1./(100.0*Tspan)
    xgrid = 2.0*np.pi*fL*tgrid
    
    C = ((Amp**2.0)*(fL**(1.0-gam_gwb))/(12.0*np.pi**2.0)) *\
    ( (ss.gamma(1.0-gam_gwb)*np.sin(np.pi*gam_gwb/2.0)*ne.evaluate("xgrid**(gam_gwb-1.)"))
       - sumTermCovarianceMatrix_fast(tgrid, fL, gam_gwb) )
    C *= ((365.25*86400.0)**2.0)
    #############################################
    # Add other white, red or DM-variation noise
    #############################################
    white = psr.toaerrs
    C += np.diag(white**2.0)

    GCG = np.dot(psr.G.T, np.dot(C, psr.G))
    
    try:
        cho = sl.cho_factor(GCG)
        like = -0.5 * np.dot(psr.Gres, sl.cho_solve(cho, psr.Gres))\
           - 0.5 * len(psr.Gres) * np.log((2.0*np.pi))\
            - 0.5 * np.sum(np.log(np.diag(cho[0])**2.0))
    except np.linalg.LinAlgError:
        print "Problem inverting matrix at A = %s, alpha = %s:" % (Amp,alpha)

        raise

    return like

def sigma_gwRMS(psr):
    """
     Formula by van Haasteren & Levin (2013, equation 24)
     sigma_gwb = 1.37e-9 * (Ah / 1e-15) * (T / yr) ^ (5/3)
     At the minute only applicable for gamma=13/3
    """
    
    gwbvar = np.absolute(np.var(psr.res)-psr.toaerrs[0]*psr.toaerrs[0])
    gwbstd = np.sqrt(gwbvar)

    Tspan = np.max((psr.toas).max() - (psr.toas).min()) * day
    
    return (gwbstd / 1.37e-9) * 1e-15 / ((Tspan / year) ** (5.0/3.0))

def optStat(psr, GCGnoiseInv, ORF, gam_gwb=4.33333):
    """
    Computes the Optimal statistic as defined in Chamberlin et al. (2014)

    @param psr: List of pulsar object instances
    @param GCGnoiseInv: List of (G * Cnoise * G)^{-1} for all pulsars
    @param ORF: Vector of pairwise overlap reduction values
    @param gam_gwb: Power Spectral index of GBW (default = 13/3, ie SMBMBs)

    @return: Opt: Optimal statistic value (A_gw^2)
    @return: sigma: 1-sigma uncertanty on Optimal statistic
    @return: snr: signal-to-noise ratio of cross correlations

    """
    
    top = 0
    bot = 0
    all_top = []
    all_bot = []
    for ll in range(len(psr)):
        for kk in range(ll+1, len(psr)):
            # form matrix of toa residuals and compute SigmaIJ
            tgrid = makeTimeGrid(psr[kk], psr[ll])
            
            # create cross covariance matrix without overall amplitude A^2
            SIJ = ORF[ll][kk] * makeRedTDcov(1.0, gam_gwb, tgrid)    
            G_SIJ_G = np.dot(psr[ll].G.T, np.dot(SIJ, psr[kk].G))

            # construct numerator and denominator of optimal statistic
            tmp_bot = np.trace(np.dot(GCGnoiseInv[ll],
                                      np.dot(G_SIJ_G, np.dot(GCGnoiseInv[kk],
                                                             G_SIJ_G.T))))
            tmp_top = np.dot(psr[ll].Gres, np.dot(GCGnoiseInv[ll],
                                                  np.dot(G_SIJ_G, np.dot(GCGnoiseInv[kk],
                                                                         psr[kk].Gres))))
            
            bot += tmp_bot
            top += tmp_top

            all_top.append(tmp_top / tmp_bot)
            all_bot.append(1./np.sqrt(tmp_bot))
            
    # compute optimal statistic
    Opt = top/bot
    
    # compute uncertainty
    sigma = 1/np.sqrt(bot)

    # compute SNR
    snr = top/np.sqrt(bot)

    # return optimal statistic and snr
    return Opt, sigma, snr, np.array(all_top), np.array(all_bot)


def TFoptStat(psr, fqs, Tspan, F, GCGnoiseInv, ORF, gam_gwb=4.33333):
    """
    Computes the Optimal statistic as defined in Chamberlin et al. (2014)

    @param psr: List of pulsar object instances
    @param GCGnoiseInv: List of (G * Cnoise * G)^{-1} for all pulsars
    @param ORF: Vector of pairwise overlap reduction values
    @param gam_gwb: Power Spectral index of GBW (default = 13/3, ie SMBMBs)

    @return: Opt: Optimal statistic value (A_gw^2)
    @return: sigma: 1-sigma uncertanty on Optimal statistic
    @return: snr: signal-to-noise ratio of cross correlations

    """
    f1yr = 1/3.16e7
    nmodes = len(fqs)
    
    top = 0
    bot = 0
    all_top = []
    all_bot = []
    for ll in range(len(psr)):
        for kk in range(ll+1, len(psr)):

            phi = np.append( 1.0/12/np.pi**2 * f1yr**(gam_gwb-3) * \
                             (fqs/86400.0)**(-gam_gwb)/Tspan, np.zeros(len(fqs)) )

            phi_signal = np.zeros(4*nmodes)
            phi_signal[0::2] = phi
            phi_signal[1::2] = phi
            
            # create cross covariance matrix without overall amplitude A^2
            SIJ = ORF[ll][kk] * np.dot(F[ll], np.dot(np.diag(phi_signal), F[kk].T)) 
            G_SIJ_G = np.dot(psr[ll].G.T, np.dot(SIJ, psr[kk].G))

            # construct numerator and denominator of optimal statistic
            tmp_bot = np.trace(np.dot(GCGnoiseInv[ll],
                                      np.dot(G_SIJ_G, np.dot(GCGnoiseInv[kk],
                                                             G_SIJ_G.T))))
            tmp_top = np.dot(psr[ll].Gres,
                             np.dot(GCGnoiseInv[ll],
                                    np.dot(G_SIJ_G, np.dot(GCGnoiseInv[kk],
                                                           psr[kk].Gres))))
            
            bot += tmp_bot
            top += tmp_top

            all_top.append(tmp_top / tmp_bot)
            all_bot.append(1./np.sqrt(tmp_bot))
            
    # compute optimal statistic
    Opt = top/bot
    
    # compute uncertainty
    sigma = 1/np.sqrt(bot)

    # compute SNR
    snr = top/np.sqrt(bot)

    # return optimal statistic and snr
    return Opt, sigma, snr, np.array(all_top), np.array(all_bot)


def AnisOptStat(psr, GCGnoiseInv, CorrCoeff, lmax, gam_gwb=4.33333):
    """
    Computes the generalised optimal statistic

    @param psr: List of pulsar object instances
    @param GCGnoiseInv: List of (G * Cnoise * G)^{-1} for all pulsars
    @param ORF: Vector of pairwise overlap reduction values
    @param gam: Power Spectral index of GBW (default = 13/3, ie SMBMBs)

    """
 
    amp = np.zeros((len(psr),len(psr)))
    weight = np.zeros((len(psr),len(psr)))
    for ll in range(len(psr)):
        for kk in range(ll+1, len(psr)):
            # form matrix of toa residuals and compute SigmaIJ
            tgrid = makeTimeGrid(psr[kk], psr[ll])
            
            # create cross covariance matrix without overall amplitude A^2
            SIJ = makeRedTDcov(1.0, gam_gwb, tgrid)    
            G_SIJ_G = np.dot(psr[ll].G.T, np.dot(SIJ, psr[kk].G))

            # construct numerator and denominator of optimal statistic
            weight[ll,kk] = np.trace(np.dot(GCGnoiseInv[ll],
                                            np.dot(G_SIJ_G, np.dot(GCGnoiseInv[kk],
                                                                   G_SIJ_G.T)))) 
            amp[ll,kk] = np.dot(psr[ll].Gres,
                                np.dot(GCGnoiseInv[ll], np.dot(G_SIJ_G,
                                                               np.dot(GCGnoiseInv[kk], psr[kk].Gres))))   

    
    X = np.array([np.multiply(CorrCoeff[ii],amp) for ii in range(len(CorrCoeff))])
    X = np.array([np.sum(X[ii]) for ii in range(len(X))])

    fisher = np.zeros((len(CorrCoeff),len(CorrCoeff)))
    for alpha in range(len(CorrCoeff)):
        for beta in range(len(CorrCoeff)):
            orf_grid = np.array(CorrCoeff)[alpha,:,:]*np.array(CorrCoeff)[beta,:,:]
            fisher[alpha,beta] = np.sum(np.triu(orf_grid*weight, 1))

    invFisher = sl.pinv(fisher)
    
    #u,s,v = sl.svd(fisher)
    #print s
    #sold = s
    #try:
    #    max_ind = np.where(s < 1e-4*s[0])[0][0] #will need to perform injections and recovery to justify this eigenvalue cutoff
    #except IndexError:
    #    max_ind = len(s)
    #s = s[:max_ind]
    #u = u[:,:max_ind]
    #v = v[:max_ind,:]
    #invFisher = np.dot(v.T, np.dot(np.diag(1./s), u.T))
    
    P = np.dot(invFisher, X)

    return P, invFisher, np.linalg.slogdet(fisher) #, sold

'''

   


