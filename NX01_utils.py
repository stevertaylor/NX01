
"""
Created by stevertaylor
Copyright (c) 2014 Stephen R. Taylor

Code contributions by Rutger van Haasteren (piccard) and Justin Ellis (PAL/PAL2).

"""

import numpy as np
from numpy import *
import os
import math
from scipy import integrate
from scipy import optimize
from scipy import constants
from numpy import random
from scipy import special as ss
from scipy import linalg as sl
import numexpr as ne
import optparse
import ephem
from ephem import *


def createfourierdesignmatrix_RED(t, nmodes, freq=False, Tspan=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    @param t: vector of time series in seconds
    @param nmodes: number of fourier coefficients to use
    @param freq: option to output frequencies
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

    # The sine/cosine modes
    ct = 0
    for ii in range(0, 2*nmodes-1, 2):
        
        F[:,ii] = np.cos(2*np.pi*fqs[ct]*t)
        F[:,ii+1] = np.sin(2*np.pi*fqs[ct]*t)
        ct += 1
    
    if freq:
        return F, fqs
    else:
        return F

def createfourierdesignmatrix_DM(t, nmodes, obs_freqs, freq=False, Tspan=None):
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
    K = 2.41*10.0**(-16.0)
    Dm = 1.0/(K*(obs_freqs*1e6)**2.0)

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


def real_sph_harm(ll, mm, phi, theta):
    """
    The real-valued spherical harmonics
    ADAPTED FROM vH piccard CODE
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
    """ngrid_phi = 20
    ngrid_costheta = 20
    
    phi = np.arange(0.0,2.0*np.pi,2.0*np.pi/ngrid_phi)
    theta = np.arccos(np.arange(-1.0,1.0,2.0/ngrid_costheta))

    xx, yy = np.meshgrid(phi,theta)

    harm_sky_vals = [[0.0]*(2*ll+1) for ll in range(lmax+1)]
    for ll in range(len(harm_sky_vals)):
        for mm in range(len(harm_sky_vals[ll])):
            harm_sky_vals[ll][mm] = real_sph_harm(ll,mm-ll,xx,yy)
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
            harmvals[ll][mm] = real_sph_harm(ll,mm-ll,skypos[:,1],skypos[:,0])

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
