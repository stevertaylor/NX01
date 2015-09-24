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

parser = optparse.OptionParser(description = 'Eccentric Fe statistic code')

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

def xan(n, omega, ecc, t, l0):  # integrating over time for residual (assuming omega and ecc not changing)

    return - np.dot( np.sin(np.outer(n,omega*t+l0)).T ,  ( ss.jn(n-2,n*ecc) - 2.*ecc*ss.jn(n-1,n*ecc) + \
                                                           (2./n)*ss.jn(n,n*ecc) + 2.*ecc*ss.jn(n+1,n*ecc) - ss.jn(n+2,n*ecc) ) )

def xbn(n, omega, ecc, t, l0):  # integrating over time for residual (assuming omega and ecc not changing)

    return np.dot( np.cos(np.outer(n,omega*t+l0)).T , np.sqrt(1.-ecc**2.) * ( ss.jn(n-2,n*ecc) - \
                                                                              2.*ss.jn(n,n*ecc) + ss.jn(n+2,n*ecc) ) )

def xcn(n, omega, ecc, t, l0):  # integrating over time for residual (assuming omega and ecc not changing)

    return np.dot( np.sin(np.outer(n,omega*t+l0)).T , (2./n) * ss.jn(n,n*ecc) )

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



def createAntennaTilde_ecc(psr, gwtheta, gwphi):
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

    p = np.array([np.cos(gwtheta)*np.cos(gwphi),np.cos(gwtheta)*np.sin(gwphi), -np.sin(gwtheta)])
    q = np.array([np.sin(gwphi),-np.cos(gwphi),0.])
                  

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
Nmat = []
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

    errs = p.toaerrs
    N = np.diag(1./( errs**2.0 ))
    try:
        cf = sl.cho_factor(Sigma[ii])
        expval2 = sl.cho_solve(cf, np.dot(p.Te.T,N))
    except np.linalg.LinAlgError:
        print 'Cholesky Decomposition Failed!! Using SVD instead'
        u,s,v = sl.svd(Sigma[ii])
        expval2 = np.dot(v.T, 1/s*np.dot(u.T, np.dot(p.Te.T,N)))

    Nmat.append( N - np.dot(np.dot(p.Te.T,N).T, expval2) )

 
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

####################################
####################################


def my_prior_mnest(cube, ndim, nparams):
    cube[0] = -9.0 + cube[0]*3.0
    cube[1] = cube[1]*2.0*np.pi
    cube[2] = -1.0 + 2.0*cube[2]
    cube[3] = cube[3]*2.0*np.pi
    if args.eccSearch==True:
        cube[4] = 0.001 + 0.899*cube[4]
     

def lnprob(cube, ndim, nparams):
    '''
    Log-likelihood ratio is the eccentric F statistic

    '''

    x = np.array([cube[ii] for ii in range(nparams)])
    if len(x)==4:
        logorbfreq, gwphi, costheta, l0 = x
        e0 = 0.001
    else:
        logorbfreq, gwphi, costheta, l0, e0 = x
    
    orbfreq = 10.0**logorbfreq
    gwtheta = np.arccos(costheta)

    eccfstat=0.
    npsr = len(psr)

    if len(x)==4 or e0<=0.001:
        nharm = np.arange(2,3,1)
    else:
        nharm = np.arange(1,int(keplersum[keplersum[:,0]>e0][0][1])+1,1)
        
    # define N vectors from Ellis et al, 2012 N_i=(x|A_i) for each pulsar
    N = np.zeros((6,len(psr)))
    M = np.zeros((6, 6, len(psr)))
    
    for ii,p in enumerate(psr):
        # Define A vector
        A = np.zeros((6, len(p.toas))) 
        Fp, Fc, dummy = createAntennaTilde_ecc(p, gwtheta, gwphi)

        A[0,:] = Fp * ( (2.0*np.pi*orbfreq)**(-1./3.) ) * xan(nharm, 2.0*np.pi*orbfreq, e0, p.toas, l0)
        A[1,:] = Fp * ( (2.0*np.pi*orbfreq)**(-1./3.) ) * xbn(nharm, 2.0*np.pi*orbfreq, e0, p.toas, l0)
        A[2,:] = Fp * ( (2.0*np.pi*orbfreq)**(-1./3.) ) * xcn(nharm, 2.0*np.pi*orbfreq, e0, p.toas, l0)
        A[3,:] = Fc * ( (2.0*np.pi*orbfreq)**(-1./3.) ) * xan(nharm, 2.0*np.pi*orbfreq, e0, p.toas, l0)
        A[4,:] = Fc * ( (2.0*np.pi*orbfreq)**(-1./3.) ) * xbn(nharm, 2.0*np.pi*orbfreq, e0, p.toas, l0)
        A[5,:] = Fc * ( (2.0*np.pi*orbfreq)**(-1./3.) ) * xcn(nharm, 2.0*np.pi*orbfreq, e0, p.toas, l0)

        N[:,ii] = np.array([np.dot(A[0,:], np.dot(Nmat[ii], p.res)), \
                            np.dot(A[1,:], np.dot(Nmat[ii], p.res)), \
                            np.dot(A[2,:], np.dot(Nmat[ii], p.res)), \
                            np.dot(A[3,:], np.dot(Nmat[ii], p.res)), \
                            np.dot(A[4,:], np.dot(Nmat[ii], p.res)), \
                            np.dot(A[5,:], np.dot(Nmat[ii], p.res))])

        # define M matrix M_ij=(A_i|A_j)
        for jj in range(6):
            for kk in range(6):
                M[jj,kk,ii] = np.dot(A[jj,:], np.dot(Nmat[ii], A[kk,:]))
        
  
    N = np.sum(N,axis=1)
    M = np.sum(M,axis=2)
    # take inverse of M
    Minv = sl.pinv(M)

    eccfstat = 0.5 * np.dot(N, np.dot(Minv, N))
    
    # return eccentric F-statistic
    return eccfstat
            

#########################
#########################

if args.eccSearch==True:
    parameters = ["orb-freq", "phi", "costheta", "l0", "e0"]
else:
    parameters = ["orb-freq", "phi", "costheta", "l0"]

print "\n You are searching for the following parameters: {0}\n".format(parameters)
n_params = len(parameters)

print "\n The total number of parameters is {0}\n".format(n_params)

#####################
# Now, we sample.....
#####################

print "\n Now, we sample... \n"

if args.eccSearch==True:
    dirextension1 = 'eccSearch'
    dirextension2 = 'eccFeSearch'
else:
    dirextension1 = 'circSearch'
    dirextension2 = 'circFeSearch'

master_path = os.getcwd()
os.chdir(args.datapath)
pymultinest.run(lnprob, my_prior_mnest, n_params, 
                importance_nested_sampling = False, 
                resume = False, verbose = True, n_live_points=1000,
                outputfiles_basename=u'chains_{0}/{1}_'.format(dirextension1,dirextension2),
                sampling_efficiency='parameter')

