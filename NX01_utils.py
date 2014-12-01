
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

day = 24 * 3600
year = 365.25 * day

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
    ########
    x = 2*np.pi*fL*tm
    ########
    sum = ne.evaluate("1/(1-gam) - x**2/(2*(3-gam)) + x**4/(24*(5-gam))")
    ########
    return sum

def makeTimeGrid(psra, psrb):

    ta, tb = np.meshgrid(psra.toas, psrb.toas)  
    tm = np.abs(ta-tb).astype(np.float64)/365.25
  
    return tm

def makeRedTDcov(Ared, gam_red, tm):

    Tspan = tm.max()
    fL = 1/(100.0*Tspan)
    xgrid = 2.0*np.pi*fL*tm
    
    Cred = ( (Ared**2.0)*(fL**(1.0-gam_red)) / (12.0*np.pi**2.0) ) * ((ss.gamma(1.0-gam_red)*np.sin(np.pi*gam_red/2.)*ne.evaluate("xgrid**(gam_red-1.0)"))\
                                                                      - sumTermCovarianceMatrix_fast(tm, fL, gam_red))
    Cred *= ((365.25*86400.0)**2.0)

    return Cred

def makeDmTDcov(psr, Adm, gam_dm, tm):

    Tspan = tm.max()
    fL = 1/(100.0*Tspan)
    xgrid = 2.0*np.pi*fL*tm
    
    K = 2.41*10.0**(-16.0)
    Dm = 1.0/(K*(psr.obs_freqs*1e6)**2.0)
    DmA,DmB = np.meshgrid(Dm,Dm)
    DmGrid = DmA*DmB
   
    Cdm = ( (Adm**2.0)*(fL**(1.0-gam_dm)) / (12.0*np.pi**2.0) ) * ((ss.gamma(1-gam_dm)*np.sin(np.pi*gam_dm/2)*ne.evaluate("xgrid**(gam_dm-1)")) \
                                                                   - sumTermCovarianceMatrix_fast(tm, fL, gam_dm))
    Cdm *= ((365.25*86400.0)**2.0)
    Cdm = np.multiply(DmGrid,Cdm)

    return Cdm

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
    ( (ss.gamma(1.0-gam_gwb)*np.sin(np.pi*gam_gwb/2.0)*ne.evaluate("xgrid**(gam_gwb-1.)")) - sumTermCovarianceMatrix_fast(tgrid, fL, gam_gwb) )
    C *= ((365.25*86400.0)**2.0)
    #############################################
    # Add other white, red or DM-variation noise
    #############################################
    white = psr.toaerrs
    C += np.diag(white**2.0)

    GCG = np.dot(psr.G.T, np.dot(C, psr.G))
    
    try:
        cho = sl.cho_factor(GCG)
        like = -0.5 * np.dot(psr.Gres, sl.cho_solve(cho, psr.Gres)) - 0.5 * len(psr.Gres) * np.log((2.0*np.pi)) - 0.5 * np.sum(np.log(np.diag(cho[0])**2.0))
    except np.linalg.LinAlgError:
        print "Problem inverting matrix at A = %s, alpha = %s:" % (Amp,alpha)

        raise

    return like

def sigma_gwRMS(psr):
    # Formula by van Haasteren & Levin (2013, equation 24)
    # sigma_gwb = 1.37e-9 * (Ah / 1e-15) * (T / yr) ^ (5/3)
    # At the minute only applicable for gamma=13/3
    
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
            tmp_bot = np.trace(np.dot(GCGnoiseInv[ll], np.dot(G_SIJ_G, np.dot(GCGnoiseInv[kk], G_SIJ_G.T))))
            tmp_top = np.dot(psr[ll].Gres, np.dot(GCGnoiseInv[ll], np.dot(G_SIJ_G, np.dot(GCGnoiseInv[kk], psr[kk].Gres))))
            
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
            weight[ll,kk] = np.trace(np.dot(GCGnoiseInv[ll], np.dot(G_SIJ_G, np.dot(GCGnoiseInv[kk], G_SIJ_G.T)))) 
            amp[ll,kk] = np.dot(psr[ll].Gres, np.dot(GCGnoiseInv[ll], np.dot(G_SIJ_G, np.dot(GCGnoiseInv[kk], psr[kk].Gres))))   

    
    X = np.array([np.multiply(CorrCoeff[ii],amp) for ii in range(len(CorrCoeff))])
    X = np.array([np.sum(X[ii]) for ii in range(len(X))])

    fisher = np.zeros((len(CorrCoeff),len(CorrCoeff)))
    for alpha in range(len(CorrCoeff)):
        for beta in range(len(CorrCoeff)):
            orf_grid = np.array(CorrCoeff)[alpha,:,:]*np.array(CorrCoeff)[beta,:,:]
            fisher[alpha,beta] = np.sum(np.triu(orf_grid*weight, 1))

    invFisher = sl.inv(fisher)
    P = np.dot(invFisher, X)

    return P, invFisher


  
    

   


