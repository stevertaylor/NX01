#!/usr/bin/env python

"""
Created by stevertaylor
Copyright (c) 2014 Stephen R. Taylor

Code contributions by Rutger van Haasteren (piccard) and Justin Ellis (PAL/PAL2).

"""

import os, math, optparse, time, cProfile
from time import gmtime, strftime
from collections import OrderedDict

import numpy as np
from numpy import *

from scipy import integrate
from scipy import optimize
from scipy import constants
from numpy import random
from scipy import special as ss
from scipy import linalg as sl

import numexpr as ne
import ephem
from ephem import *

import libstempo as T2

import NX01_AnisCoefficients as anis
import NX01_utils as utils
import NX01_psr

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

import NX01_jitter as jitter

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = optparse.OptionParser(description = 'NX01 - Precursor to the PANTHER Group ENTERPRISE project')

############################
############################

parser.add_option('--nmodes', dest='nmodes', action='store', type=int,
                   help='Number of modes in low-rank time-frequency approximation')
parser.add_option('--cadence', dest='cadence', action='store', type=float, default=14.0,
                   help='Number days between successive observations (default = 14 days)')
parser.add_option('--parfile', dest='parfile', action='store', type=str,
                   help='Full path to parfile')
parser.add_option('--timfile', dest='timfile', action='store', type=str,
                   help='Full path to timfile')
parser.add_option('--efacequad-sysflag', dest='systarg', action='store', type=str, default='group',
                   help='Which system flag should the EFACs/EQUADs target? (default = \'group\')')
parser.add_option('--redamp-prior', dest='redamp_prior', action='store', type=str, default='loguniform',
                   help='What kind of prior to place on the red noise amplitude? (default = \'loguniform\')')
parser.add_option('--dmamp-prior', dest='dmamp_prior', action='store', type=str, default='loguniform',
                   help='What kind of prior to place on the DM variation amplitude? (default = \'loguniform\')')
parser.add_option('--dmVar', dest='dmVar', action='store_true', default=False,
                   help='Search for DM variations in the data (False)? (default=False)')
parser.add_option('--fullN', dest='fullN', action='store_true', default=False,
                   help='Search for EFAC/EQUAD/ECORR over all systems (True), or just apply a GEFAC (False)? (default=False)')
parser.add_option('--jitterbin', dest='jitterbin', action='store', type=float, default=1.0,
                   help='What time duration do you want a jitter bin to be? (default = 1.0)')
parser.add_option('--ptmcmc', dest='ptmcmc', action='store_true', default=False,
                   help='Sample using PALs parallel tempering MCMC (False)? (default=False)')

(args, x) = parser.parse_args()

if args.nmodes:
    print ("\n You've given me the number of frequencies",
           "to include in the low-rank time-frequency approximation, got it?\n")
else:
    print ("\n You've given me the sampling cadence for the observations,",
           "which determines the upper frequency limit and the number of modes, got it?\n")

if args.ptmcmc==True:
    import PALInferencePTMCMC as PAL
else:
    import pymultinest

################################################################################################################################
# PASSING THROUGH TEMPO2 VIA libstempo
################################################################################################################################
t2psr = T2.tempopulsar(parfile=args.parfile, timfile=args.timfile)
t2psr.fit(iters=10)
if np.any(np.isfinite(t2psr.residuals())==False)==True:
    t2psr = T2.tempopulsar(parfile=args.parfile,timfile=args.timfile)

psr = NX01_psr.PsrObj(t2psr)
psr.grab_all_vars(jitterbin=args.jitterbin)

################################################################################################################################
# GETTING MAXIMUM TIME, COMPUTING FOURIER DESIGN MATRICES, AND GETTING MODES 
################################################################################################################################

Tmax = psr.toas.max() - psr.toas.min()

if args.nmodes:

    psr.makeTe(args.nmodes, Tmax, makeDM=args.dmVar)
    # get GW frequencies
    fqs = np.linspace(1/Tmax, args.nmodes/Tmax, args.nmodes)
    nmode = args.nmodes

else:

    nmode = int(round(Tmax/args.cadence))
    psr.makeTe(nmode, Tmax, makeDM=args.dmVar)
    # get GW frequencies
    fqs = np.linspace(1/Tmax, nmode/Tmax, nmode)

################################################################################################################################
# FORM A LIST COMPOSED OF NP ARRAYS CONTAINING THE INDEX POSITIONS WHERE EACH UNIQUE SYSTEM IS APPLIED
################################################################################################################################

if args.fullN:
    systems = psr.sysflagdict[args.systarg]
else:
    systems = OrderedDict.fromkeys([psr.name])
    systems[psr.name] = np.arange(len(psr.toas))

################################################################################################################################
# SETTING UP PRIOR RANGES
################################################################################################################################

pmin = np.array([-20.0,0.0])
pmax = np.array([-11.0,7.0])
if args.dmVar:
    pmin = np.append(pmin,np.array([-20.0,0.0]))
    pmax = np.append(pmax,np.array([-8.0,7.0]))       
pmin = np.append(pmin,0.001*np.ones(len(systems)))
pmax = np.append(pmax,10.0*np.ones(len(systems)))
if args.fullN:
    pmin = np.append(pmin,-10.0*np.ones(len(systems)))
    pmax = np.append(pmax,-5.0*np.ones(len(systems)))
    if len(psr.sysflagdict['nano-f'].keys())>0:
        pmin = np.append(pmin, -8.5*np.ones(len(psr.sysflagdict['nano-f'].keys())))
        pmax = np.append(pmax, -5.0*np.ones(len(psr.sysflagdict['nano-f'].keys())))
            
################################################################################################################################
# PRIOR AND LIKELIHOOD
################################################################################################################################

def my_prior1(x):
    logp = 0.
    
    if np.all(x <= pmax) and np.all(x >= pmin):
        logp = np.sum(np.log(1/(pmax-pmin)))
    else:
        logp = -np.inf
    
    return logp

def my_prior2(cube, ndim, nparams):
    cube[0] = -20.0 + cube[0]*9.0
    cube[1] = cube[1]*7.0

    ct = 2
    if args.dmVar:
        cube[ct] = -20.0 + cube[ct]*12.0
        cube[ct+1] = cube[ct+1]*7.0
        ct = 4

    for ii in range(ct,ct+len(systems)):
        cube[ii] = 0.001 + cube[ii]*9.999
    if args.fullN:
        for ii in range(ct+len(systems),ct+2*len(systems)):
            cube[ii] = -10.0 + cube[ii]*5.0
        for ii in range(ct+2*len(systems),nparams):
            cube[ii] = -8.5 + cube[ii]*3.5

#######################
#######################
            
def ln_prob1(xx):
    
    Ared = 10.0**xx[0]
    gam_red = xx[1]

    ct = 2
    if args.dmVar:
        Adm = 10.0**xx[ct]
        gam_dm = xx[ct+1]

        ct = 4

    EFAC = xx[ct:ct+len(systems)]
    if args.fullN: 
        EQUAD = 10.0**xx[ct+len(systems):ct+2*len(systems)]
        ECORR = 10.0**xx[ct+2*len(systems):]

    loglike1 = 0

    ####################################
    ####################################
    scaled_err = (psr.toaerrs).copy()
    for jj,sysname in enumerate(systems):
        scaled_err[systems[sysname]] *= EFAC[jj] 
    ###
    white_noise = np.zeros(len(scaled_err))
    if args.fullN:
        white_noise = np.ones(len(scaled_err))
        for jj,sysname in enumerate(systems):
            white_noise[systems[sysname]] *= EQUAD[jj]
    
    new_err = np.sqrt( scaled_err**2.0 + white_noise**2.0 )
    ########

    # compute ( T.T * N^-1 * T )
    # & log determinant of N
    if args.fullN:
        if len(ECORR)>0:
            Jamp = np.ones(len(psr.epflags))
            for jj,nano_sysname in enumerate(psr.sysflagdict['nano-f'].keys()):
                Jamp[np.where(psr.epflags==nano_sysname)] *= ECORR[jj]**2.0

            Nx = jitter.cython_block_shermor_0D(psr.res, new_err**2.,
                                                Jamp, psr.Uinds)
            d = np.dot(psr.Te.T, Nx)
            logdet_N, TtNT = \
              jitter.cython_block_shermor_2D(psr.Te, new_err**2.,
                                             Jamp, psr.Uinds)
            det_dummy, dtNdt = \
              jitter.cython_block_shermor_1D(psr.res, new_err**2.,
                                             Jamp, psr.Uinds)
        else:
            d = np.dot(psr.Te.T, psr.res/( new_err**2.0 ))
        
            N = 1./( new_err**2.0 )
            right = (N*psr.Te.T).T
            TtNT = np.dot(psr.Te.T, right)
    
            logdet_N = np.sum(np.log( new_err**2.0 ))
        
            # triple product in likelihood function
            dtNdt = np.sum(psr.res**2.0/( new_err**2.0 ))
        
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
    if args.dmVar:
        kappa = np.log10( np.append( Ared**2/12/np.pi**2 * \
                                     f1yr**(gam_red-3) * \
                                     (fqs/86400.0)**(-gam_red)/Tspan,
                                    Adm**2/12/np.pi**2 * \
                                    f1yr**(gam_dm-3) * \
                                    (fqs/86400.0)**(-gam_dm)/Tspan ) )
    else:
        kappa = np.log10( Ared**2/12/np.pi**2 * \
                          f1yr**(gam_red-3) * \
                          (fqs/86400.0)**(-gam_red)/Tspan )

    # construct elements of sigma array
    if args.dmVar:
        mode_count = 4*nmode
    else:
        mode_count = 2*nmode

    diagonal = np.zeros(mode_count)
    diagonal[0::2] =  10**kappa
    diagonal[1::2] = 10**kappa

    # compute Phi inverse 
    red_phi = np.diag(1./diagonal)
    logdet_Phi = np.sum(np.log( diagonal ))
  
    # now fill in real covariance matrix
    Phi = np.zeros( TtNT.shape )
    for kk in range(0,mode_count):
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


    logLike = -0.5 * (logdet_Phi + logdet_Sigma) + \
      0.5 * (np.dot(d, expval2)) + loglike1

    prior_fac = 0.0
    if args.redamp_prior == 'uniform':
        prior_fac += np.log(Ared * np.log(10.0))
    if (args.dmVar==True) and (args.dmamp_prior == 'uniform'):
        prior_fac += np.log(Adm * np.log(10.0))
    
    return logLike + prior_fac


def ln_prob2(cube, ndim, nparams):

    cube = np.array([cube[ii] for ii in range(nparams)])
    xx = cube

    Ared = 10.0**xx[0]
    gam_red = xx[1]

    ct = 2
    if args.dmVar==True:
        Adm = 10.0**xx[ct]
        gam_dm = xx[ct+1]

        ct = 4

    EFAC = xx[ct:ct+len(systems)]
    if args.fullN: 
        EQUAD = 10.0**xx[ct+len(systems):ct+2*len(systems)]
        ECORR = 10.0**xx[ct+2*len(systems):]

    loglike1 = 0

    ####################################
    ####################################
    scaled_err = (psr.toaerrs).copy()
    for jj,sysname in enumerate(systems):
        scaled_err[systems[sysname]] *= EFAC[jj] 
    ###
    white_noise = np.zeros(len(scaled_err))
    if args.fullN:
        white_noise = np.ones(len(scaled_err))
        for jj,sysname in enumerate(systems):
            white_noise[systems[sysname]] *= EQUAD[jj]
    
    new_err = np.sqrt( scaled_err**2.0 + white_noise**2.0 )
    ########

    # compute ( T.T * N^-1 * T ) & log determinant of N
    if args.fullN:
        if len(ECORR)>0:
            Jamp = np.ones(len(psr.epflags))
            for jj,nano_sysname in enumerate(psr.sysflagdict['nano-f'].keys()):
                Jamp[np.where(psr.epflags==nano_sysname)] *= ECORR[jj]**2.0

            Nx = jitter.cython_block_shermor_0D(psr.res, new_err**2.,
                                                Jamp, psr.Uinds)
            d = np.dot(psr.Te.T, Nx)
            logdet_N, TtNT = \
              jitter.cython_block_shermor_2D(psr.Te, new_err**2.,
                                             Jamp, psr.Uinds)
            det_dummy, dtNdt = \
              jitter.cython_block_shermor_1D(psr.res, new_err**2.,
                                             Jamp, psr.Uinds)
        else:
            d = np.dot(psr.Te.T, psr.res/( new_err**2.0 ))
        
            N = 1./( new_err**2.0 )
            right = (N*psr.Te.T).T
            TtNT = np.dot(psr.Te.T, right)
    
            logdet_N = np.sum(np.log( new_err**2.0 ))
        
            # triple product in likelihood function
            dtNdt = np.sum(psr.res**2.0/( new_err**2.0 ))
        
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
    if args.dmVar:
        kappa = np.log10( np.append( Ared**2/12/np.pi**2 * \
                                     f1yr**(gam_red-3) *\
                                      (fqs/86400.0)**(-gam_red)/Tspan,
                                    Adm**2/12/np.pi**2 * \
                                    f1yr**(gam_dm-3) * \
                                    (fqs/86400.0)**(-gam_dm)/Tspan ) )
    else:
        kappa = np.log10( Ared**2/12/np.pi**2 * \
                          f1yr**(gam_red-3) * \
                          (fqs/86400.0)**(-gam_red)/Tspan )

    # construct elements of sigma array
    if args.dmVar:
        mode_count = 4*nmode
    else:
        mode_count = 2*nmode

    diagonal = np.zeros(mode_count)
    diagonal[0::2] =  10**kappa
    diagonal[1::2] = 10**kappa

    # compute Phi inverse 
    red_phi = np.diag(1./diagonal)
    logdet_Phi = np.sum(np.log( diagonal ))
  
    # now fill in real covariance matrix
    Phi = np.zeros( TtNT.shape )
    for kk in range(0,mode_count):
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


    logLike = -0.5 * (logdet_Phi + logdet_Sigma) + \
      0.5 * (np.dot(d, expval2)) + loglike1

    prior_fac = 0.0
    if args.redamp_prior == 'uniform':
        prior_fac += np.log(Ared * np.log(10.0))
    if (args.dmVar==True) and (args.dmamp_prior == 'uniform'):
        prior_fac += np.log(Adm * np.log(10.0))
    
    return logLike + prior_fac

#########################
#########################

parameters = ["log(A_red)","gam_red"]
if args.dmVar:
    parameters.append("log(A_dm)")
    parameters.append("gam_dm")
for ii in range(len(systems)):
    parameters.append('EFAC_'+systems.keys()[ii])
if args.fullN:
    for ii in range(len(systems)):
        parameters.append('EQUAD_'+systems.keys()[ii])

    if len(psr.sysflagdict['nano-f'].keys())>0:
        if rank == 0:
            print "\n You have some NANOGrav ECORR parameters..."
        for ii,nano_sysname in enumerate(psr.sysflagdict['nano-f'].keys()):
            parameters.append('ECORR_'+nano_sysname)

n_params = len(parameters)
if rank == 0:
    print "\n You are searching for the following single-pulsar parameters: {0}\n".format(parameters)
    print "\n The total number of parameters is {0}\n".format(n_params)

##################################
# Now, we sample or maximize.....
##################################

if rank == 0:
    os.system('say -v Victoria \'Engage N X zero 1!\' ')

if args.ptmcmc:
    
    x0 = np.array([-15.0,2.0])
    cov_diag = np.array([0.5,0.5])
    
    if args.dmVar:
        x0 = np.append(x0,np.array([-15.0,2.0]))
        cov_diag = np.append(cov_diag,np.array([0.5,0.5]))

    x0 = np.append(x0,np.random.uniform(0.75,1.25,len(systems)))
    cov_diag = np.append(cov_diag,0.5*np.ones(len(systems)))

    if args.fullN:
        x0 = np.append(x0,np.random.uniform(-10.0,-5.0,len(systems)))
        cov_diag = np.append(cov_diag,0.5*np.ones(len(systems)))
        if len(psr.sysflagdict['nano-f'].keys())>0:
            x0 = np.append(x0, np.random.uniform(-8.5,-5.0,len(psr.sysflagdict['nano-f'].keys())))
            cov_diag = np.append(cov_diag,0.5*np.ones(len(psr.sysflagdict['nano-f'].keys())))
    
    if rank == 0:
        print "\n Your initial parameters are {0}\n".format(x0)


    sampler = PAL.PTSampler(ndim = n_params, logl = ln_prob1,
                            logp = my_prior1, cov = np.diag(cov_diag),
                            outDir='./chains_{0}_{1}_{2}'.format(psr.name,'red'+args.redamp_prior,'ptmcmc'),
                            resume = False)

    def drawFromRedNoisePrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        # log prior
        if args.redamp_prior == 'loguniform':
            
            q[0] = np.random.uniform(pmin[0], pmax[0])
            qxy += 0

        elif args.redamp_prior == 'uniform':
            
            q[0] = np.random.uniform(pmin[0], pmax[0])
            qxy += 0

            #Ared = np.log10(np.random.uniform(10 ** Ared_ll, 10 ** Ared_ul, len(Ared)))
            #qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])

        q[1] = np.random.uniform(pmin[1], pmax[1])
    
        qxy += 0
        
        return q, qxy

    def drawFromEquadPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        if args.dmVar==True:
            ind = np.arange(4+len(systems),4+2*len(systems))
        else:
            ind = np.arange(2+len(systems),2+2*len(systems))
        equad_jump = np.zeros(len(q))
        equad_jump[ind] = np.random.uniform(pmin[ind[0]], pmax[ind[0]], len(systems))

        q = equad_jump
        
        qxy += 0
        
        return q, qxy

    def drawFromEcorrPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        if args.dmVar==True:
            ind = np.arange(4+2*len(systems),4+3*len(systems))
        else:
            ind = np.arange(2+2*len(systems),2+3*len(systems))
        ecorr_jump = np.zeros(len(q))
        ecorr_jump[ind] = np.random.uniform(pmin[ind[0]], pmax[ind[0]], len(systems))

        q = ecorr_jump
        
        qxy += 0
        
        return q, qxy


    # add jump proposals
    sampler.addProposalToCycle(drawFromRedNoisePrior, 10)
    sampler.addProposalToCycle(drawFromEquadPrior, 10)
    if (args.fullN==True) and (len(psr.sysflagdict['nano-f'].keys())>0):
        sampler.addProposalToCycle(drawFromEcorrPrior, 10)
    #if args.incDMBand and args.dmModel=='powerlaw':
    #    sampler.addProposalToCycle(model.drawFromDMNoiseBandPrior, 5)

    sampler.sample(p0=x0, Niter=1e6, thin=10,
                   covUpdate=1000, AMweight=15,
                   SCAMweight=30, DEweight=50, KDEweight=0)
else:

    if not os.path.exists('chains_{0}_{1}'.format(psr.name,'mnest')):
        os.makedirs('chains_{0}_{1}'.format(psr.name,'mnest'))
    
    pymultinest.run(ln_prob2, my_prior2, n_params,
                    importance_nested_sampling = False,
                    resume = False, verbose = True, 
                    n_live_points=500,
                    outputfiles_basename=u'chains_{0}_{1}/{0}_'.format(psr.name,'Mnest'), 
                    sampling_efficiency=0.8,
                    const_efficiency_mode=False)

