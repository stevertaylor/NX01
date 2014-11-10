#!/usr/bin/env python

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

parser = optparse.OptionParser(description = 'NX01 - Precursor to the PANTHER Group ENTERPRISE project')

############################
############################

parser.add_option('--nmodes', dest='nmodes', action='store', type=int,
                   help='Number of modes in low-rank time-frequency approximation')
parser.add_option('--cadence', dest='cadence', action='store', type=float, default=14,
                   help='Number days between successive observations (default = 14 days)')
parser.add_option('--parfile', dest='parfile', action='store', type=str,
                   help='Full path to parfile')
parser.add_option('--timfile', dest='timfile', action='store', type=str,
                   help='Full path to timfile')
parser.add_option('--sample-or-maximize', dest='sample_or_maximize', action='store', type=str, default='maximize',
                   help='Do you want sample from the posteror distribution or just find the maximum likelihood noise values? (default=\'maximize\')?')

(args, x) = parser.parse_args()

if args.sample_or_maximize == 'maximize':
    from pyswarm import pso
    
if not os.path.exists('chains_singlePsr'):
    os.makedirs('chains_singlePsr')

if args.nmodes:
    print "\n You've given me the number of frequencies to include in the low-rank time-frequency approximation, got it?\n"
else:
    print "\n You've given me the sampling cadence for the observations, which determines the upper frequency limit and the number of modes, got it?\n"

################################################################################################################################
# PASSING THROUGH TEMPO2 VIA libstempo
################################################################################################################################
t2psr = T2.tempopulsar(parfile=args.parfile, timfile=args.timfile)
t2psr.fit(iters=10)
if np.any(np.isfinite(t2psr.residuals())==False)==True:
    t2psr = T2.tempopulsar(parfile=args.parfile,timfile=args.timfile)

psr = NX01_psr.PsrObj(t2psr)
psr.grab_all_vars() 

################################################################################################################################
# GETTING MAXIMUM TIME, COMPUTING FOURIER DESIGN MATRICES, AND GETTING MODES 
################################################################################################################################

Tmax = psr.toas.max() - psr.toas.min()

if args.nmodes:

    psr.makeTe(args.nmodes, Tmax)
    # get GW frequencies
    fqs = np.linspace(1/Tmax, args.nmodes/Tmax, args.nmodes)
    nmode = args.nmodes

else:

    nmode = int(round(Tmax/args.cadence))
    psr.makeTe(nmode, Tmax)
    # get GW frequencies
    fqs = np.linspace(1/Tmax, nmode/Tmax, nmode)

################################################################################################################################
# FORM A LIST COMPOSED OF NP ARRAYS CONTAINING THE INDEX POSITIONS WHERE EACH UNIQUE 'sys' BACKEND IS APPLIED
################################################################################################################################

psr.get_backends() 
backends = psr.bkends
        
################################################################################################################################
# SETTING UP PRIOR RANGES
################################################################################################################################
pmin = np.array([-20.0,0.0]) # red-noise
pmin = np.append(pmin,np.array([-20.0,0.0])) # DM-variation noise
pmin = np.append(pmin,0.1*np.ones(len(backends))) # EFACs
pmin = np.append(pmin,-10.0*np.ones(len(backends))) #EQUADs

pmax = np.array([-10.0,7.0]) # red-noise
pmax = np.append(pmax,np.array([-10.0,7.0])) # DM-variation noise
pmax = np.append(pmax,11.9*np.ones(len(backends))) # EFACs
pmax = np.append(pmax,-3.0*np.ones(len(backends))) #EQUADs

##################################################################################################################################

def my_prior(x):
    logp = 0.
    
    if np.all(x <= pmax) and np.all(x >= pmin):
        logp = np.sum(np.log(1/(pmax-pmin)))
    else:
        logp = -np.inf
    
    return logp


def ln_prob(x):
            
    Ared = 10.0**x[0]
    gam_red = x[1]

    Adm = 10.0**x[2]
    gam_dm = x[3]

    EFAC=[0.0]*len(backends)
    EQUAD=[0.0]*len(backends)
    for ii in range(len(backends)):
        EFAC[ii] = x[4+ii] 
    for ii in range(len(backends)):
        EQUAD[ii] = 10.0**x[4+len(backends)+ii] 


    loglike1 = 0
    
    ####################################
    ####################################
    scaled_err = (psr.toaerrs).copy()
    for jj in range(len(backends)):
        scaled_err[backends[jj]] *= EFAC[jj] 
    ###
    white_noise = np.ones(len(scaled_err))
    for jj in range(len(backends)):
        white_noise[backends[jj]] *= EQUAD[jj]
    
    new_err = np.sqrt( scaled_err**2.0 + white_noise**2.0 )
    ########
    # compute d    
    d = np.dot(psr.Te.T, psr.res/( new_err**2.0 ))
       
    # compute T.T * N^-1 * T
    N = 1./( new_err**2.0 )
    right = (N*psr.Te.T).T
    TtNT = np.dot(psr.Te.T, right)

    # log determinant of N
    logdet_N = np.sum(np.log( new_err**2.0 ))
        
    # triple product in likelihood function
    dtNdt = np.sum(psr.res**2.0/( new_err**2.0 ))

    loglike1 += -0.5 * (logdet_N + dtNdt)
    ####################################
    ####################################
    
    # parameterize intrinsic red noise as power law
    Tspan = (1/fqs[0])*86400.0
    f1yr = 1/3.16e7

    # parameterize intrinsic red-noise and DM-variations as power law
    kappa = np.log10( np.append( Ared**2/12/np.pi**2 * f1yr**(gam_red-3) * (fqs/86400.0)**(-gam_red)/Tspan, Adm**2/12/np.pi**2 * f1yr**(gam_dm-3) * (fqs/86400.0)**(-gam_dm)/Tspan ) )

    # construct elements of sigma array
    diagonal = np.zeros(4*nmode)
    diagonal[0::2] =  10**kappa
    diagonal[1::2] = 10**kappa

    # compute Phi inverse 
    red_phi = np.diag(1./diagonal)
    logdet_Phi = np.sum(np.log( diagonal ))
  
    # now fill in real covariance matrix
    Phi = np.zeros( TtNT.shape ) 
    for kk in range(0,4*nmode):
        Phi[kk+psr.Gc.shape[1],kk+psr.Gc.shape[1]] = red_phi[kk,kk]

    # symmeterize Phi
    Phi = Phi + Phi.T - np.diag(np.diag(Phi))
    
    # compute sigma
    Sigma = TtNT + Phi
     
    # cholesky decomp for second term in exponential
    try:
        cf = sl.cho_factor(Sigma)
        expval2 = sl.cho_solve(cf, d)
        logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

    except np.linalg.LinAlgError:
        print 'Cholesky Decomposition Failed second time!! Using SVD instead'
        u,s,v = sl.svd(Sigma)
        expval2 = np.dot(v.T, 1/s*np.dot(u.T, d))
        logdet_Sigma = np.sum(np.log(s))


    logLike = -0.5 * (logdet_Phi + logdet_Sigma) + 0.5 * (np.dot(d, expval2)) + loglike1 
    
    return logLike + sum(np.log(EQUAD[ii] * np.log(10.0)) for ii in range(len(backends)))

#########################
#########################

parameters = ["log(A_red)","gam_red","log(A_dm)","gam_dm"]
for ii in range(len(backends)):
    parameters.append('EFAC_'+psr.bkend_names[ii])
for ii in range(len(backends)):
    parameters.append('log(EQUAD_'+psr.bkend_names[ii]+')')

print "\n You are searching for the following single-pulsar parameters: {0}\n".format(parameters)
n_params = len(parameters)

print "\n The total number of parameters is {0}\n".format(n_params)

x0 = np.array([-15.0,3.0,-15.0,3.0])
x0 = np.append(x0,np.random.uniform(0.75,1.25,len(backends)))
x0 = np.append(x0,np.random.uniform(-10.0,-3.0,len(backends)))

print "\n Your initial parameters are {0}\n".format(x0)

cov_diag = np.array([0.5,0.5,0.5,0.5])
cov_diag = np.append(cov_diag,0.2*np.ones(len(backends)))
cov_diag = np.append(cov_diag,0.5*np.ones(len(backends)))

print "\n Running a quick profile on the likelihood to estimate evaluation speed...\n"
cProfile.run('ln_prob(x0)')

##################################
# Now, we sample or maximize.....
##################################

if args.sample_or_maximize == 'maximize':
    def swarm_prob(x):
        return -ln_prob(x)

    xopt, fopt = pso(swarm_prob, pmin, pmax, swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=10000, debug=True)

    print "\n Printing out the ML noise values for {0}...\n".format(psr.name)
    fil = open("{0}_MLnoise.txt".format(psr.name),'w')
    for ii in range(len(xopt)):
        print "{0} = {1}".format(parameters[ii],xopt[ii])
        print >>fil, "{0} {1}".format(parameters[ii],xopt[ii])
    print "\n Values saved in {0}_MLnoise_nmode{1}.txt".format(psr.name,nmode)

else:
    
    print "\n Now, we sample... \n"
    sampler = PAL.PTSampler(ndim=n_params,logl=ln_prob,logp=my_prior,cov=np.diag(cov_diag),\
                            outDir='./chains_singlePsr/{0}_singlePsr_tf_nmode{1}'.format(psr.name,nmode),resume=False)
    sampler.sample(p0=x0,Niter=500000,thin=10)



