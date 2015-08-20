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
import pymultinest
import libstempo as T2
import time
from time import gmtime, strftime
import NX01_AnisCoefficients as anis
import NX01_utils as utils
import NX01_psr

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

import NX01_jitter as jitter

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
parser.add_option('--target-sysflag', dest='systarg', action='store', type=str, default='group',
                   help='Which system flag should the EFACs/EQUADs target? (default = \'group\')')
parser.add_option('--fullN', dest='fullN', action='store_true', default=False,
                   help='Search for EFAC/EQUAD/ECORR over all systems (True), or just apply a GEFAC (False)? (default=False)')
parser.add_option('--sample-or-maximize', dest='sample_or_maximize', action='store', type=str, default='maximize',
                   help='Do you want sample from the posteror distribution or just find the maximum likelihood noise values? (default=\'maximize\')')

(args, x) = parser.parse_args()
print args.fullN
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

if args.sample_or_maximize == 'maximize':
    from pyswarm import pso
else:    
    if not os.path.exists('chains_nano_{0}'.format(psr.name)):
        os.makedirs('chains_nano_{0}'.format(psr.name))

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
# FORM A LIST COMPOSED OF NP ARRAYS CONTAINING THE INDEX POSITIONS WHERE EACH UNIQUE SYSTEM IS APPLIED
################################################################################################################################

if args.fullN==True:
    #systems = psr.systems
    #sysnames = psr.sysnames
    systems = psr.sysflagdict[args.systarg]
else:
    #systems = [np.arange(len(psr.toas))]
    #sysnames = [psr.name]
    systems = OrderedDict.fromkeys([psr.name])
    systems[psr.name] = np.arange(len(psr.toas))

#print len(systems), len(psr.epflags)
#print psr.epflags

################################################################################################################################
# PRIOR AND LIKELIHOOD
################################################################################################################################

def my_prior(cube, ndim, nparams):
    cube[0] = -20.0 + cube[0]*12.0
    cube[1] = cube[1]*6.95
    cube[2] = -20.0 + cube[2]*12.0
    cube[3] = cube[3]*6.95
    for ii in range(4,4+len(systems)):
        cube[ii] = 0.1 + cube[ii]*11.9
    if args.fullN==True:
        for ii in range(4+len(systems),4+2*len(systems)):
            cube[ii] = -10.0 + cube[ii]*7.0
        for ii in range(4+2*len(systems),nparams):
            cube[ii] = -10.0 + cube[ii]*7.0

def ln_prob(cube, ndim, nparams):

    cube = np.array([cube[ii] for ii in range(nparams)])
    xx = cube

    Ared = 10.0**xx[0]
    gam_red = xx[1]

    Adm = 10.0**xx[2]
    gam_dm = xx[3]

    EFAC = xx[4:4+len(systems)]
    if args.fullN==True: 
        EQUAD = 10.0**xx[4+len(systems):4+2*len(systems)]
        ECORR = 10.0**xx[4+2*len(psr.sysflagdict['nano-f'].keys()):]

    loglike1 = 0

    ####################################
    ####################################
    scaled_err = (psr.toaerrs).copy()
    for jj,sysname in enumerate(systems):
        scaled_err[systems[sysname]] *= EFAC[jj] 
    ###
    white_noise = np.zeros(len(scaled_err))
    if args.fullN==True:
        white_noise = np.ones(len(scaled_err))
        for jj,sysname in enumerate(systems):
            white_noise[systems[sysname]] *= EQUAD[jj]
    
    new_err = np.sqrt( scaled_err**2.0 + white_noise**2.0 )
    ########

    # compute ( T.T * N^-1 * T ) & log determinant of N
    if args.fullN==True:

        Jamp = np.ones(len(psr.epflags))
        for jj,nano_sysname in enumerate(psr.sysflagdict['nano-f'].keys()):
            Jamp[np.where(psr.epflags==nano_sysname)] *= ECORR[jj]**2.0

        Nx = jitter.cython_block_shermor_0D(psr.res.astype(np.float64), new_err**2., Jamp, psr.Uinds)
        d = np.dot(psr.Te.T, Nx)
        
        logdet_N, TtNT = jitter.cython_block_shermor_2D(psr.Te, new_err**2., Jamp, psr.Uinds)

        det_dummy, dtNdt = jitter.cython_block_shermor_1D(psr.res.astype(np.float64), new_err**2., Jamp, psr.Uinds)
        
    else:

        d = np.dot(psr.Te.T, psr.res/( new_err**2.0 ))
        
        N = 1./( new_err**2.0 )
        right = (N*psr.Te.T).T
        TtNT = np.dot(psr.Te.T, right)

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
    kappa = np.log10( np.append( Ared**2/12/np.pi**2 * f1yr**(gam_red-3) * (fqs/86400.0)**(-gam_red)/Tspan,
                                 Adm**2/12/np.pi**2 * f1yr**(gam_dm-3) * (fqs/86400.0)**(-gam_dm)/Tspan ) )

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
    
    return logLike 

#########################
#########################

parameters = ["log(A_red)","gam_red","log(A_dm)","gam_dm"]
for ii in range(len(systems)):
    parameters.append('EFAC_'+systems.keys()[ii])
if args.fullN==True:
    for ii in range(len(systems)):
        parameters.append('EQUAD_'+systems.keys()[ii])
    for ii,nano_sysname in enumerate(psr.sysflagdict['nano-f'].keys()):
        parameters.append('ECORR_'+nano_sysname)

print "\n You are searching for the following single-pulsar parameters: {0}\n".format(parameters)
n_params = len(parameters)

print "\n The total number of parameters is {0}\n".format(n_params)

##################################
# Now, we sample or maximize.....
##################################

if args.sample_or_maximize == 'maximize':

    ##################################################################
    # SETTING UP PRIOR RANGES
    ##################################################################
    pmin = np.array([-20.0,0.0]) # red-noise
    pmin = np.append(pmin,np.array([-20.0,0.0])) # DM-variation noise
    pmin = np.append(pmin,0.1*np.ones(len(systems))) # EFACs
    if args.fullN==True:
        pmin = np.append(pmin,-10.0*np.ones(len(systems))) #EQUADs
        pmin = np.append(pmin,-10.0*np.ones(len(systems))) #ECORRs
    
    pmax = np.array([-8.0,7.0]) # red-noise
    pmax = np.append(pmax,np.array([-8.0,7.0])) # DM-variation noise
    pmax = np.append(pmax,11.9*np.ones(len(systems))) # EFACs
    if args.fullN==True:
        pmin = np.append(pmin,-3.0*np.ones(len(systems))) #EQUADs
        pmax = np.append(pmax,-3.0*np.ones(len(systems))) #ECORRs

    ###################################################################

    def swarm_prob(x):
        return -ln_prob(x)

    xopt, fopt = pso(swarm_prob, pmin, pmax, swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=1000, debug=True)

    print "\n Printing out the ML noise values for {0}...\n".format(psr.name)
    fil = open("{0}_MLnoise_nmode{1}.txt".format(psr.name,nmode),'w')
    for ii in range(len(xopt)):
        print "{0} = {1}".format(parameters[ii],xopt[ii])
        print >>fil, "{0} {1}".format(parameters[ii],xopt[ii])
    print "\n Values saved in {0}_MLnoise_nmode{1}.txt".format(psr.name,nmode)

else:

    pymultinest.run(ln_prob, my_prior, n_params, importance_nested_sampling = False, resume = False, verbose = True, 
                    n_live_points=500, outputfiles_basename=u'chains_nano_{0}/{0}_'.format(psr.name), 
                    sampling_efficiency=0.8,const_efficiency_mode=False)

