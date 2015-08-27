#!/usr/bin/env python

"""
Created by stevertaylor
Copyright (c) 2014 Stephen R. Taylor

Code contributions by Rutger van Haasteren (piccard) and Justin Ellis (PAL/PAL2).

"""

import numpy as np
from numpy import *
import os
import glob
import math
from collections import OrderedDict
from scipy import integrate as si
from scipy import optimize as sopt
from scipy import constants as sc
from numpy import random
from scipy import special as ss
from scipy import linalg as sl
from scipy import constants as sc
import numexpr as ne
import optparse
import cProfile
import ephem
from ephem import *
import PALInferencePTMCMC as PAL
import libstempo as T2
import time
from time import gmtime, strftime
import NX01_AnisCoefficients as anis
import NX01_utils as utils
import NX01_psr
import pymultinest

Msol = 1.98855*10.0**30.0

parser = optparse.OptionParser(description = 'Eccentric single source code')

############################
############################

parser.add_option('--datapath', dest='datapath', action='store', type=str,
                   help='Which data are you analysing?')
parser.add_option('--eccSearch', dest='eccSearch', action='store_true', default=False,
                   help='Do you want to search for eccentricity?')

(args, x) = parser.parse_args()

if args.eccSearch==True:
    if not os.path.exists(args.datapath+'/chains_eccSearch'):
        os.makedirs(args.datapath+'/chains_eccSearch')
else:
    if not os.path.exists(args.datapath+'/chains_circSearch'):
        os.makedirs(args.datapath+'/chains_circSearch')
        
############################
############################

def sigma(e):
    return ( (e**(12./19.))*((1. + (121./304.)*e**2.)**(870./2299.))/(1.-e**2.) )

def orb_ang_freq(e0, omega0, e):
    return omega0*( sigma(e0)/sigma(e) )**(3./2.)

# Single differential equation in eccentricity
def g(y,t,param):
    mc = param[0]
    estart = param[1]
    wstart = param[2]
    
    ei = y[0]
    g0 = (-304./15.)*((sc.G*mc*Msol)**(5./3.))*(orb_ang_freq(estart,wstart,ei)**(8./3.)) * \
      (ei/( (1.-ei**2.)**(5./2.) )) * ( 1. + (121./304.)*ei**2. )
    g0 /= sc.c**5.
    g0 *= 86400.*365.25
    
    return [g0]

def an(n, mass, dL, omega, ecc, t, l0):  # integrating over time for residual (assuming omega and ecc not changing)
    
    mass *= Msol
    dL *= (10.**6.)*sc.parsec
    
    amp = mass**(5./3.) / ( dL * omega**(1./3.) )
    amp *= sc.G**(5./3.) / sc.c**4.

    return -amp * ( ss.jn(n-2,n*ecc) - 2.*ecc*ss.jn(n-1,n*ecc) + (2./n)*ss.jn(n,n*ecc) + \
                    2.*ecc*ss.jn(n+1,n*ecc) - ss.jn(n+2,n*ecc) ) * np.sin(n*omega*t + n*l0)

def bn(n, mass, dL, omega, ecc, t, l0):  # integrating over time for residual (assuming omega and ecc not changing)
    
    mass *= Msol
    dL *= (10.**6.)*sc.parsec
    
    amp = mass**(5./3.) / ( dL * omega**(1./3.) )
    amp *= sc.G**(5./3.) / sc.c**4.

    return amp * np.sqrt(1.-ecc**2.) * ( ss.jn(n-2,n*ecc) - 2.*ss.jn(n,n*ecc) + \
                                         ss.jn(n+2,n*ecc) ) * np.cos(n*omega*t + n*l0)

def cn(n, mass, dL, omega, ecc, t, l0):  # integrating over time for residual (assuming omega and ecc not changing)
    
    mass *= Msol
    dL *= (10.**6.)*sc.parsec
    
    amp = mass**(5./3.) / ( dL * omega**(1./3.) )
    amp *= sc.G**(5./3.) / sc.c**4.

    return (2./n) * amp * ss.jn(n,n*ecc) * np.sin(n*omega*t + n*l0)

###########################################################

def splus_n(n, mass, dL, omega, ecc, t, l0, gamma, iota):
    return -(1.+np.cos(iota)**2.) * ( an(n, mass, dL, omega, ecc, t, l0)*np.cos(2.*gamma) \
                                      - bn(n, mass, dL, omega, ecc, t, l0)*np.sin(2.*gamma) ) \
                                      + (1.-np.cos(iota)**2.) * cn(n, mass, dL, omega, ecc, t, l0)

def scross_n(n, mass, dL, omega, ecc, t, l0, gamma, iota):
    return 2.*np.cos(iota) * ( bn(n, mass, dL, omega, ecc, t, l0)*np.cos(2.*gamma) \
                               + an(n, mass, dL, omega, ecc, t, l0)*np.sin(2.*gamma) )


############################################################

def createAntennaPatternFuncs_ecc(psr, gwtheta, gwphi, gwpsi):
    """
    Function to create pulsar antenna pattern functions as defined
    in Barack & Cutler (2004).

    @param psr: pulsar object for single pulsar
    @param gwtheta: GW polar angle in radians
    @param gwphi: GW azimuthal angle in radians
    @param gwpsi: GW polarisation angle

    @return: (fpus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the 
             pulsar and the GW source.
    """

    p = np.array([np.cos(gwpsi)*np.cos(gwtheta)*np.cos(gwphi) - np.sin(gwpsi)*np.sin(gwphi),\
                  np.cos(gwpsi)*np.cos(gwtheta)*np.sin(gwphi) + np.sin(gwpsi)*np.cos(gwphi),
                  -np.cos(gwpsi)*np.sin(gwtheta)])
    q = np.array([np.sin(gwpsi)*np.cos(gwtheta)*np.cos(gwphi) + np.cos(gwpsi)*np.sin(gwphi),\
                  np.sin(gwpsi)*np.cos(gwtheta)*np.sin(gwphi) - np.cos(gwpsi)*np.cos(gwphi),\
                  -np.sin(gwpsi)*np.sin(gwtheta)])
                  

    omhat = np.array([-np.sin(gwtheta)*np.cos(gwphi), -np.sin(gwtheta)*np.sin(gwphi),\
                      -np.cos(gwtheta)])

    psrphi = psr.psr_locs[0]
    psrtheta = np.pi/2. - psr.psr_locs[1]
    phat = np.array([np.sin(psrtheta)*np.cos(psrphi), np.sin(psrtheta)*np.sin(psrphi), \
                     np.cos(psrtheta)])

    fplus = 0.5 * (np.dot(p, phat)**2 - np.dot(q, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(p, phat)*np.dot(q, phat)) / (1 + np.dot(omhat, phat))
    
    cosMu = -np.dot(omhat, phat)

    return fplus, fcross, cosMu


def createResiduals_ecc(psr, gwtheta, gwphi, mc, dist, forb, inc, gwpsi, gamma, e0, l0, \
                        Nharm=400, pdist=None, psrTerm=True):
    """

    @param psr: pulsar object for single pulsar
    @param gwtheta: Polar angle of GW source in celestial coords [radians]
    @param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    @param mc: Chirp mass of SMBMB [solar masses]
    @param dist: Luminosity distance to SMBMB [Mpc]
    @param forb: mean orbital frequency of binary [Hz]
    @param inc: Inclination of GW source [radians]
    @param gwpsi: Polarization of GW source [radians]
    @param gamma: An orientation angle.... [radians]
    @param e0: Initial orbital eccentricity of Earth-term
    @param l0: Initial mean anomaly [radians]
    @param Nharm: Number of harmonics included in Bessel series
    @param pdist: Pulsar distance to use other than those in psr [kpc]
    @param psrTerm: Option to include pulsar term [boolean] 

    @return: Vector of induced residuals

    """

    # get antenna pattern funcs and cosMu
    Fp, Fc, cosMu = createAntennaPatternFuncs_ecc(psr, gwtheta, gwphi, gwpsi)

    if psrTerm:
        pdist *= 1e3*sc.parsec

    # get values from pulsar object
    toas = psr.toas  # remember, this is in seconds, and the toas are zeroed to a minimum!

    # Earth-term 
    param = [mc,e0,2.*np.pi*forb]
    omega = 2.*np.pi*forb
    
    if e0 > 0.001:
        res_earth = Fp*np.sum([splus_n(ii, mc, dist, omega, e0, toas, l0, gamma, inc) for ii in range(1,Nharm)], axis=0) \
        + Fc*np.sum([scross_n(ii, mc, dist, omega, e0, toas, l0, gamma, inc)  for ii in range(1,Nharm)], axis=0)
    else:
        res_earth = Fp*np.sum([splus_n(ii, mc, dist, omega, e0, toas, l0, gamma, inc) for ii in range(2,3)], axis=0) \
        + Fc*np.sum([scross_n(ii, mc, dist, omega, e0, toas, l0, gamma, inc)  for ii in range(2,3)], axis=0)

    return res_earth

############################
############################

parfiles = []
timfiles = []
for file in sorted(glob.glob(args.datapath+"/*.par")):
    parfiles.append(file)
    
for file in sorted(glob.glob(args.datapath+"/*.tim")):
    timfiles.append(file)

print parfiles, timfiles
################################################################################################################################
# PASSING THROUGH TEMPO2 VIA libstempo
################################################################################################################################

t2psr=[]
for ii in range(len(parfiles)):
    t2psr.append(T2.tempopulsar(parfile=parfiles[ii], timfile=timfiles[ii]))
    #t2psr[ii].fit(iters=10)
    #if np.any(np.isfinite(t2psr[ii].residuals())==False)==True:
    #    t2psr[ii] = T2.tempopulsar(parfile=parfiles[ii], timfile=timfiles[ii])


# storing red-noise amplitude (nano style),
# gamma (nano style), white-noise level (microseconds)
# [from Arzoumanian et al. (2015a)]
active_psrs = OrderedDict.fromkeys([p.name for p in t2psr])

active_psrs['B1855+09'] = [0.017,-4.9,0.505]
active_psrs['J0030+0451'] = [0.014,-4.8,0.212]
active_psrs['J0613-0200'] = [0.093,-2.9,0.165]
active_psrs['J1012+5307'] = [0.669,-1.0,0.355]
active_psrs['J1024-0719'] = [0.0,0.0,0.280]
active_psrs['J1455-3330'] = [0.0,0.0,0.694]
active_psrs['J1600-3053'] = [0.0,0.0,0.197]
active_psrs['J1614-2230'] = [0.0,0.0,0.189]
active_psrs['J1640+2224'] = [0.0,0.0,0.158]
active_psrs['J1713+0747'] = [0.0,0.0,0.116]
active_psrs['J1741+1351'] = [0.0,0.0,0.103]
active_psrs['J1744-1134'] = [0.0,0.0,0.334]
active_psrs['J1853+1303'] = [0.0,0.0,0.235]
active_psrs['J1909-3744'] = [0.0,0.0,0.080]
active_psrs['J1918-0642'] = [0.0,0.0,0.340]
active_psrs['J2010-1323'] = [0.0,0.0,0.312]
active_psrs['J2145-0750'] = [0.0,0.0,0.370]
active_psrs['J2317+1439'] = [0.0,0.0,0.267]        

################################################################################################################################
# MAKING A PULSAR OBJECT, THEN GRABBING ALL THE VARIABLES, e.g. toas, residuals, error-bars, designmatrices etc.
################################################################################################################################

psr = [NX01_psr.PsrObj(t2psr[ii]) for ii in range(len(t2psr))]

[p.grab_all_vars() for p in psr]

psr_positions = [np.array([p.psr_locs[0], np.pi/2. - p.psr_locs[1]]) for p in psr]
positions = np.array(psr_positions).copy()

nmodes = 50
Tspan = np.min([p.toas.max() - p.toas.min() for p in psr])
fqs = np.linspace(1/Tspan, nmodes/Tspan, nmodes)
[p.makeTe(nmodes=50, Ttot=Tspan, makeDM=False) for p in psr]


logdet_N = []
TtNT = []
logdet_Phi = []
Sigma = []
for ii,p in enumerate(psr):
    errs = p.toaerrs
   
    N = 1./( errs**2.0 )
    right = (N*p.Te.T).T
    TtNT.append( np.dot(p.Te.T, right) )

    logdet_N.append( np.sum(np.log( errs**2.0 )) )

               
for ii,p in enumerate(psr):    
    # parameterize intrinsic red noise as power law
    Tspan = (1/fqs[0])*86400.0
    f1yr = 1/3.16e7
    nmode = len(fqs)

    Ared = 1e-6 * active_psrs[p.name][0] * f1yr * np.sqrt(12.0*np.pi**2.0)
    gam_red = -active_psrs[p.name][1]

    if Ared == 0.0:
        Ared = 1e-25

    # parameterize intrinsic red-noise as power law
    kappa = np.log10( Ared**2/12/np.pi**2 * f1yr**(gam_red-3) * (fqs/86400.0)**(-gam_red)/Tspan )

    # construct elements of sigma array
    diagonal = np.zeros(2*nmode)
    diagonal[0::2] =  10**kappa
    diagonal[1::2] = 10**kappa

    # compute Phi inverse 
    red_phi = np.diag(1./diagonal)
    logdet_Phi.append( np.sum(np.log( diagonal )) )

    # now fill in real covariance matrix
    Phi = np.zeros( TtNT[ii].shape ) 
    for kk in range(0,2*nmode):
        Phi[kk+p.Gc.shape[1],kk+p.Gc.shape[1]] = red_phi[kk,kk]

    # symmeterize Phi
    Phi = Phi + Phi.T - np.diag(np.diag(Phi))
    
    # compute sigma
    Sigma.append( TtNT[ii] + Phi )

 
# make sure all pulsar have same reference time
tt=[] 
for p in psr:
    tt.append(np.min(p.toas))

# find reference time
tref = np.min(tt)

# now scale pulsar time and convert to seconds
for p in psr:
    p.toas -= tref
    p.toas *= 86400.0

######

# reading in number of harmonics required for accurate agreement with numerical solution to Kepler problem
# allows adaptive harmonic summing

keplersum = np.loadtxt('/Users/staylor/Research/PapersInProgress/PTA-eccentric/PTA-single-source/code/eccNumFourierHarmsAdaptDot_07-03-2015.txt')


def my_prior_mnest(cube, ndim, nparams):
    cube[0] = 7.0 + cube[0]*3.0
    cube[1] = cube[1]*4.0
    cube[2] = -9.0 + cube[2]*3.0
    cube[3] = cube[3]*2.0*np.pi
    cube[4] = -1.0 + 2.0*cube[4]
    cube[5] = -1.0 + 2.0*cube[5]
    cube[6] = cube[6]*np.pi
    cube[7] = cube[7]*np.pi
    cube[8] = cube[8]*2.0*np.pi
    if args.eccSearch==True:
        cube[9] = 0.001 + 0.899*cube[9]


def lnprob(cube, ndim, nparams):
    
    # Log-likelihood ratio for eccentric single-source detection.

    
    x = np.array([cube[ii] for ii in range(nparams)])
    if len(x)==9:
        logmass, logdist, logorbfreq, gwphi, costheta, cosinc, gwpol, gwgamma, l0 = x
    else:
        logmass, logdist, logorbfreq, gwphi, costheta, cosinc, gwpol, gwgamma, l0, e0 = x
        
    mass = 10.0**logmass
    dist = 10.0**logdist
    orbfreq = 10.0**logorbfreq
    gwtheta = np.arccos(costheta)
    gwinc = np.arccos(cosinc)

    if len(x)==9:
        nharm = 3
    else:
        nharm = int(keplersum[keplersum[:,0]>e0][0][1])+1 #np.arange(1,int(keplersum[keplersum[:,0]>e0][0][1]),1)
    
    gwres = []
    sig_sub = []

    if len(x)==9:
        for ii,p in enumerate(psr):
            gwres.append( createResiduals_ecc(p, gwtheta, gwphi, mass, dist, orbfreq, gwinc, gwpol, gwgamma, 0.001, l0, Nharm=nharm, psrTerm=False) )
            sig_sub.append( p.res - gwres[ii] )
    else:
        for ii,p in enumerate(psr):
            gwres.append( createResiduals_ecc(p, gwtheta, gwphi, mass, dist, orbfreq, gwinc, gwpol, gwgamma, e0, l0, Nharm=nharm, psrTerm=False) )
            sig_sub.append( p.res - gwres[ii] )
    
    d = []
    dtNdt = []
    logLike = 0.0
    for ii,p in enumerate(psr):
        errs = p.toaerrs
        d.append( np.dot(p.Te.T, sig_sub[ii]/( errs**2.0 )) )

        # triple product in likelihood function
        dtNdt.append( np.sum(sig_sub[ii]**2.0/( errs**2.0 )) )

        loglike1 = -0.5 * (logdet_N[ii] + dtNdt[ii])
        # cholesky decomp for second term in exponential
        try:
            cf = sl.cho_factor(Sigma[ii])
            expval2 = sl.cho_solve(cf, d[ii])
            logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

        except np.linalg.LinAlgError:
            print 'Cholesky Decomposition Failed second time!! Using SVD instead'
            u,s,v = sl.svd(Sigma[ii])
            expval2 = np.dot(v.T, 1/s*np.dot(u.T, d[ii]))
            logdet_Sigma = np.sum(np.log(s))
    
        logLike += -0.5 * (logdet_Phi[ii] + logdet_Sigma) + 0.5 * (np.dot(d[ii], expval2)) + loglike1

    return logLike
            

#########################
#########################

if args.eccSearch==True:
    parameters = ["mass", "dist", "orb-freq", "phi", "costheta", "cosiota", "gwpol", "gwgamma", "l0", "e0"]
else:
    parameters = ["mass", "dist", "orb-freq", "phi", "costheta", "cosiota", "gwpol", "gwgamma", "l0"]

print "\n You are searching for the following parameters: {0}\n".format(parameters)
n_params = len(parameters)

print "\n The total number of parameters is {0}\n".format(n_params)

#####################
# Now, we sample.....
#####################

print "\n Now, we sample... \n"

if args.eccSearch==True:
    dirextension = 'eccSearch'
else:
    dirextension = 'circSearch'

master_path = os.getcwd()
os.chdir(args.datapath)
pymultinest.run(lnprob, my_prior_mnest, n_params, 
                importance_nested_sampling = False, 
                resume = False, verbose = True, n_live_points=1000,
                outputfiles_basename=u'chains_{0}/{0}_'.format(dirextension),
                sampling_efficiency='parameter')







