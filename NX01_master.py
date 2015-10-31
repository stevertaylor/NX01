#!/usr/bin/env python

"""
Created by stevertaylor
Copyright (c) 2014 Stephen R. Taylor

Code contributions by Rutger van Haasteren (piccard) and Justin Ellis (PAL/PAL2).

"""

import os, math, optparse, time, cProfile
from time import gmtime, strftime
from collections import OrderedDict
import h5py as h5

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

parser.add_option('--from-h5', dest='from_h5', action='store_true', default = False,
                   help='Do you want to read in pulsars from hdf5 files instead of directly via libstempo? (default = False)')
parser.add_option('--psrlist', dest='psrlist', action='store', type=str, default = None,
                   help='Provide path to file containing list of pulsars and their respective par/tim paths')
parser.add_option('--nmodes', dest='nmodes', action='store', type=int,
                   help='Number of modes in low-rank time-frequency approximation')
parser.add_option('--dmVar', dest='dmVar', action='store_true', default=False,
                   help='Search for DM variations in the data (False)? (default=False)')
parser.add_option('--ptmcmc', dest='ptmcmc', action='store_true', default=True,
                   help='Sample using PALs parallel tempering MCMC (False)? (default=True)')
parser.add_option('--num_gwfreq_wins', dest='num_gwfreq_wins', action='store', type=int, default=1,
                   help='Number windows to split the band into (useful for evolving anisotropic searches (default = 1 windows)')
parser.add_option('--lmax', dest='LMAX', action='store', type=int, default=0,
                   help='Maximum multipole in anisotropic search (default = 0, i.e. isotropic-search)')
parser.add_option('--use-gpu', dest='use_gpu', action='store_true', default=False,
                  help='Do you want to use the GPU for accelerated linear algebra? (default = False)')
parser.add_option('--fix-slope', dest='fix_slope', action='store_true', default=False,
                  help='Do you want to fix the slope of the GWB spectrum? (default = False)')
parser.add_option('--limit-or-detect-gwb', dest='limit_or_detect_gwb', action='store', type=str, default='limit',
                   help='Do you want to use a uniform prior on log_10(Agwb) [detect] or Agwb itself [upper-limit] (default=\'limit\')?')
parser.add_option('--limit-or-detect-red', dest='limit_or_detect_red', action='store', type=str, default='limit',
                   help='Do you want to use a uniform prior on log_10(Ared) [detect] or Ared itself [upper-limit] (default=\'limit\')?')
parser.add_option('--anis-modefile', dest='anis_modefile', action='store', type=str, default = None,
                   help='Do you want to provide an anisotropy modefile to split band into frequency windows?')
parser.add_option('--fullN', dest='fullN', action='store_true', default=True,
                  help='Do you want to perform a full noise search? (default = True)')

(args, x) = parser.parse_args()

# Do you want to use GPU acceleration?
if args.use_gpu:
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import pycuda.elementwise as el
    import pycuda.tools as tools
    import scikits.cuda.linalg as culinalg
    import scikits.cuda.misc as cumisc

    culinalg.init()

if args.nmodes:
    print "\n You've given me the number of frequencies to include in the low-rank time-frequency approximation, got it?\n"
else:
    print "\n You've given me the sampling cadence for the observations, which determines the upper frequency limit and the number of modes, got it?\n"

if args.ptmcmc:
    import PALInferencePTMCMC as PAL
else:
    import pymultinest

parser = optparse.OptionParser(description = 'NX01 - Precursor to the PANTHER Group ENTERPRISE project')

################################################################################################################################
# PASSING THROUGH TEMPO2 VIA libstempo
################################################################################################################################

psr_pathinfo = np.genfromtxt(args.psrlist, dtype=str, skip_header=2) # name, hdf5-path, par-path, tim-path

if args.from_h5:

    tmp_psr = []
    for ii,tmp_name in enumerate(psr_pathinfo[:18,0]):
        tmp_psr.append(h5.File(psr_pathinfo[ii,1], 'r')[tmp_name])

    psr = [NX01_psr.PsrObjFromH5(p) for p in tmp_psr]
    
else:
    
    t2psr=[]
    for ii in range(len(psr_pathinfo)):
        t2psr.append( T2.tempopulsar( parfile=psr_pathinfo[ii,2], timfile=psr_pathinfo[ii,3] ) )
        t2psr[ii].fit(iters=3)
        if np.any(np.isfinite(t2psr.residuals())==False)==True:
            t2psr = T2.tempopulsar( parfile=psr_pathinfo[ii,2], timfile=psr_pathinfo[ii,3] )

    psr = [NX01_psr.PsrObj(p) for p in t2psr]


# Grab all the pulsar quantities
[p.grab_all_vars() for p in psr]

# Now, grab the positions and compute the ORF basis functions
psr_positions = [np.array([psr[ii].psr_locs[0],
                           np.pi/2. - psr[ii].psr_locs[1]])
                           for ii in range(len(psr))]
positions = np.array(psr_positions).copy()

CorrCoeff = np.array(anis.CorrBasis(positions,args.LMAX))       # Computing all the correlation basis-functions for the array.
harm_sky_vals = utils.SetupPriorSkyGrid(args.LMAX)              # Computing the values of the spherical-harmonics up to order
                                                                # LMAX on a pre-specified grid

                                                            
if args.anis_modefile is None:
    gwfreqs_per_win = int(1.*args.nmodes/(1.*args.num_gwfreq_wins)) # getting the number of GW frequencies per window
    anis_modefreqs = np.arange(1,args.nmodes+1)
    anis_modefreqs = np.reshape(anis_modefreqs, (args.num_gwfreq_wins,gwfreqs_per_win))

    tmp_num_gwfreq_wins = args.num_gwfreq_wins
else:
    tmp_modefreqs = np.loadtxt(args.anis_modefile)
    tmp_num_gwfreq_wins = tmp_modefreqs.shape[0]
    anis_modefreqs = []
    for ii in range(tmp_num_gwfreq_wins):
        anis_modefreqs.append(np.arange(tmp_modefreqs[ii,0],tmp_modefreqs[ii,1]+1))

# Create a tag for evolving anisotropy searches
if (args.LMAX!=0) and (tmp_num_gwfreq_wins > 1):
    evol_anis_tag = 'EvAnis'
else:
    evol_anis_tag = ''
              
#############################################################################
# GETTING MAXIMUM TIME, COMPUTING FOURIER DESIGN MATRICES, AND GETTING MODES 
#############################################################################

Tmax = np.max([p.toas.max() - p.toas.min() for p in psr])

if args.nmodes:

    [p.makeTe(args.nmodes, Tmax, makeDM=args.dmVar) for p in psr]
    # get GW frequencies
    fqs = np.linspace(1/Tmax, args.nmodes/Tmax, args.nmodes)
    nmode = args.nmodes

else:

    nmode = int(round(Tmax/args.cadence))
    [p.makeTe(nmode, Tmax, makeDM=args.dmVar) for p in psr]
    # get GW frequencies
    fqs = np.linspace(1/Tmax, nmode/Tmax, nmode)


#######################################
# PRE-COMPUTING WHITE NOISE PROPERTIES 
#######################################

loglike1 = 0
TtNT = []
d = []
for ii,p in enumerate(psr):

    # compute ( T.T * N^-1 * T ) & log determinant of N
    new_err = (p.toaerrs).copy()
    if args.fullN==True:
        
        if len(p.ecorrs)>0:

            Jamp = np.ones(len(p.epflags))
            for jj,nano_sysname in enumerate(p.sysflagdict['nano-f'].keys()):
                Jamp[np.where(p.epflags==nano_sysname)] *= p.ecorrs[nano_sysname]**2.0

            Nx = jitter.cython_block_shermor_0D(p.res, new_err**2., Jamp, p.Uinds)
            d.append(np.dot(p.Te.T, Nx))
            logdet_N, TtNT_dummy = jitter.cython_block_shermor_2D(p.Te, new_err**2., Jamp, p.Uinds)
            TtNT.append(TtNT_dummy)
            det_dummy, dtNdt = jitter.cython_block_shermor_1D(p.res, new_err**2., Jamp, p.Uinds)

        else:
            
            d.append(np.dot(p.Te.T, p.res/( new_err**2.0 )))
        
            N = 1./( new_err**2.0 )
            right = (N*p.Te.T).T
            TtNT.append(np.dot(p.Te.T, right))
    
            logdet_N = np.sum(np.log( new_err**2.0 ))
        
            # triple product in likelihood function
            dtNdt = np.sum(p.res**2.0/( new_err**2.0 ))
        
    else:

        d.append(np.dot(p.Te.T, p.res/( new_err**2.0 )))
            
        N = 1./( new_err**2.0 )
        right = (N*p.Te.T).T
        TtNT.append(np.dot(p.Te.T, right))

        logdet_N = np.sum(np.log( new_err**2.0 ))
        
        # triple product in likelihood function
        dtNdt = np.sum(p.res**2.0/( new_err**2.0 ))

    loglike1 += -0.5 * (logdet_N + dtNdt)

d = np.concatenate(d)


##########################
# SETTING UP PRIOR RANGES
##########################

pmin = -20.0*np.ones(len(psr))
pmin = np.append(pmin,0.0*np.ones(len(psr)))
if args.dmVar==True:
    pmin = np.append(pmin,-20.0*np.ones(len(psr)))
    pmin = np.append(pmin,0.0*np.ones(len(psr)))
pmin = np.append(pmin,-18.0)
if args.fix_slope==False:
    pmin = np.append(pmin,0.0)
pmin = np.append(pmin,-10.0*np.ones( tmp_num_gwfreq_wins*(((args.LMAX+1)**2)-1) ))


pmax = -11.0*np.ones(len(psr))
pmax = np.append(pmax,7.0*np.ones(len(psr)))
if args.dmVar==True:
    pmax = np.append(pmax,-11.0*np.ones(len(psr)))
    pmax = np.append(pmax,7.0*np.ones(len(psr)))
pmax = np.append(pmax,-11.0)
if args.fix_slope==False:
    pmax = np.append(pmax,7.0)
pmax = np.append(pmax,10.0*np.ones( tmp_num_gwfreq_wins*(((args.LMAX+1)**2)-1) ))

##################################################################################

def my_prior(xx):
    logp = 0.
    
    if np.all(xx <= pmax) and np.all(xx >= pmin):
        logp = np.sum(np.log(1/(pmax-pmin)))
    else:
        logp = -np.inf
    
    return logp


def lnprob(xx):

    npsr = len(psr) 

    if args.dmVar==True:
        Ared, gam_red, Adm, gam_dm, Agwb, gam_gwb, orf_coeffs = utils.masterSplitParams(xx, npsr, args.dmVar, args.fix_slope)
        mode_count = 4*nmode
    else:
        Ared, gam_red, Agwb, gam_gwb, orf_coeffs = utils.masterSplitParams(xx, npsr, args.dmVar, args.fix_slope)
        mode_count = 2*nmode


    # Reshaping freq-dependent anis coefficients,
    # and testing for power distribution physicality.

    orf_coeffs = orf_coeffs.reshape((tmp_num_gwfreq_wins,((args.LMAX+1)**2)-1))
    clm = np.array([[0.0]*((args.LMAX+1)**2) for ii in range(tmp_num_gwfreq_wins)])
    clm[:,0] = 2.0*np.sqrt(np.pi)

    physicality = 0.
    if args.LMAX!=0:

        for kk in range(tmp_num_gwfreq_wins):
            for ii in range(1,((args.LMAX+1)**2)):
                clm[kk,ii] = orf_coeffs[kk,ii-1]   

            # Testing for physicality of power distribution.
            if (utils.PhysPrior(clm[kk],harm_sky_vals) == 'Unphysical'):
                physicality += -10.0**7.0
            else:
                physicality += 0.

    
    # Computing frequency dependent overlap reduction functions.
    ORF=[]
    for ii in range(tmp_num_gwfreq_wins): # number of frequency windows
        for jj in range(len(anis_modefreqs[ii])): # number of frequencies in this window
            ORF.append( sum(clm[ii,kk]*CorrCoeff[kk] for kk in range(len(CorrCoeff))) )
    if args.dmVar==True:
        for ii in range(tmp_num_gwfreq_wins): # number of frequency windows
            for jj in range(len(anis_modefreqs[ii])): # number of frequencies in this window
                ORF.append( np.zeros((npsr,npsr)) )

    ORF = np.array(ORF)
    ORFtot = np.zeros((mode_count,npsr,npsr)) # shouldn't be applying ORF to dmfreqs,
                                                 # but the projection of GW spec onto dmfreqs
                                                 # is defined as zero below.
    ORFtot[0::2] = ORF
    ORFtot[1::2] = ORF
    
       
    # parameterize intrinsic red noise as power law
    Tspan = (1/fqs[0])*86400.0
    f1yr = 1/3.16e7
    rho = np.log10(Agwb**2/12/np.pi**2 * f1yr**(gam_gwb-3) * (fqs/86400.0)**(-gam_gwb)/Tspan)

    # parameterize intrinsic red-noise and DM-variations as power law
    kappa = [] 
    if args.dmVar==True:
        for ii in range(npsr):
            kappa.append(np.log10( np.append( Ared[ii]**2/12/np.pi**2 * f1yr**(gam_red[ii]-3) * (fqs/86400.0)**(-gam_red[ii])/Tspan,
                                        Adm[ii]**2/12/np.pi**2 * f1yr**(gam_dm[ii]-3) * (fqs/86400.0)**(-gam_dm[ii])/Tspan ) ))
    else:
        for ii in range(npsr):
            kappa.append(np.log10( Ared[ii]**2/12/np.pi**2 * f1yr**(gam_red[ii]-3) * (fqs/86400.0)**(-gam_red[ii])/Tspan ))
    

    # construct elements of sigma array
    sigdiag = []
    sigoffdiag = []
    
    if args.dmVar==True:
        gwbspec = np.append( 10**rho, np.zeros_like(rho) )
    else:
        gwbspec = 10**rho
        
    for ii in range(npsr):
        tot = np.zeros(mode_count)
        offdiag = np.zeros(mode_count)

        # off diagonal terms
        offdiag[0::2] = gwbspec
        offdiag[1::2] = gwbspec

        # diagonal terms
        tot[0::2] = ORF[:,ii,ii]*gwbspec + 10**kappa[ii]
        tot[1::2] = ORF[:,ii,ii]*gwbspec + 10**kappa[ii] 
                
        # fill in lists of arrays
        sigdiag.append(tot)
        sigoffdiag.append(offdiag)


    # compute Phi matrix
    smallMatrix = np.zeros((mode_count, npsr, npsr))
    for ii in range(npsr):
        for jj in range(ii,npsr):

            if ii == jj:
                smallMatrix[:,ii,jj] = sigdiag[jj] 
            else:
                smallMatrix[:,ii,jj] = ORFtot[:,ii,jj] * sigoffdiag[jj] 
                smallMatrix[:,jj,ii] = smallMatrix[:,ii,jj]

    
    # invert Phi matrix frequency-wise
    logdet_Phi = 0
    non_pos_def = 0
    for ii in range(mode_count):

        try:

            L = sl.cho_factor(smallMatrix[ii,:,:])
            smallMatrix[ii,:,:] = sl.cho_solve(L, np.eye(npsr))
            logdet_Phi += np.sum(2*np.log(np.diag(L[0])))

        except np.linalg.LinAlgError:

            print 'Cholesky Decomposition Failed!! Rejecting...'
            non_pos_def += 1

    # Break if we have non-positive-definiteness of Phi
    if non_pos_def > 0:

        return -np.inf

    else:

        bigTtNT = sl.block_diag(*TtNT)
        Phi = np.zeros_like( bigTtNT )

        # now fill in real covariance matrix
        ind = [0]
        ind = np.append(ind,np.cumsum([TtNT[ii].shape[0] for ii in range(len(psr))]))
        ind = [np.arange(ind[ii]+psr[ii].Gc.shape[1],ind[ii]+psr[ii].Gc.shape[1]+mode_count)
               for ii in range(len(ind)-1)]
        for ii in range(npsr):
            for jj in range(npsr):
                Phi[ind[ii],ind[jj]] = smallMatrix[:,ii,jj]
            
        # compute sigma
        Sigma = bigTtNT + Phi
            
        # cholesky decomp for second term in exponential
        if args.use_gpu:

            try:

                Sigma_gpu = gpuarray.to_gpu( Sigma.astype(np.float64).copy() )
                expval2_gpu = gpuarray.to_gpu( d.astype(np.float64).copy() )
                culinalg.cho_solve( Sigma_gpu, expval2_gpu ) # in-place linear-algebra:
                                                             # Sigma and expval2 overwritten
                logdet_Sigma = np.sum(2.0*np.log(np.diag(Sigma_gpu.get())))

            except cula.culaDataError:

                print 'Cholesky Decomposition Failed (GPU error!!)'
                return -np.inf

            logLike = -0.5 * (logdet_Phi + logdet_Sigma) + 0.5 * (np.dot(d, expval2_gpu.get() )) + loglike1
            
        else:

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


        # Multiplying likelihood to correct log-uniform
        # sampling thus making a uniform prior
        if args.limit_or_detect_gwb == 'limit':
            priorfac_gwb = np.log(Agwb * np.log(10.0))
        else:
            priorfac_gwb = 0.0

        if args.limit_or_detect_red == 'limit':
            priorfac_red = np.sum(np.log(Ared * np.log(10.0)))
        else:
            priorfac_red = 0.0


        return logLike + priorfac_gwb + priorfac_red + physicality


#########################
#########################

# Set up the parameter list

parameters=[]
for ii in range(len(psr)):
    parameters.append('Ared_'+psr[ii].name)
for ii in range(len(psr)):
    parameters.append('gam_red_'+psr[ii].name)
if args.dmVar==True:
    for ii in range(len(psr)):
        parameters.append('Adm_'+psr[ii].name)
    for ii in range(len(psr)):
        parameters.append('gam_dm_'+psr[ii].name)
parameters.append("Agwb")
if args.fix_slope is False:
    parameters.append("gam_gwb")
    gamma_ext = 'GamVary'
else:
    gamma_ext = 'Gam4p33'
for ii in range( tmp_num_gwfreq_wins*(((args.LMAX+1)**2)-1) ):
    parameters.append('clm_{0}'.format(ii+1))


print "\n You are searching for the following parameters: {0}\n".format(parameters)
n_params = len(parameters)

print "\n The total number of parameters is {0}\n".format(n_params)


# Start the sampling off with some reasonable parameter choices
x0 = np.log10(np.array([p.Redamp for p in psr]))
x0 = np.append(x0,np.array([p.Redind for p in psr]))
if args.dmVar==True:
    x0 = np.append(x0,np.log10(np.array([p.Redamp for p in psr])))
    x0 = np.append(x0,np.array([p.Redind for p in psr]))
x0 = np.append(x0,-15.0)
if args.fix_slope is False:
    x0 = np.append(x0,13./3.)
x0 = np.append(x0,np.zeros( tmp_num_gwfreq_wins*(((args.LMAX+1)**2)-1) ))

print "\n Your initial parameters are {0}\n".format(x0)

# Make a reasonable covariance matrix to commence sampling
cov_diag = 0.5*np.ones(len(psr))
cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))
if args.dmVar==True:
    cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))
    cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))
cov_diag = np.append(cov_diag,0.5)
if args.fix_slope is False:
    cov_diag = np.append(cov_diag,0.5)
cov_diag = np.append(cov_diag,0.05*np.ones( tmp_num_gwfreq_wins*(((args.LMAX+1)**2)-1) ))


print "\n Running a quick profile on the likelihood to estimate evaluation speed...\n"
cProfile.run('lnprob(x0)')

#####################
# Now, we sample.....
#####################

print "\n Now, we sample... \n"
sampler = PAL.PTSampler(ndim=n_params,logl=lnprob,logp=my_prior,cov=np.diag(cov_diag),\
                        outDir='./chains_nanoAnalysis/nanograv_gwb{0}_red{1}_nmodes{2}_Lmax{3}_{4}_{5}'.\
                        format(args.limit_or_detect_gwb,args.limit_or_detect_red,args.nmodes,args.LMAX,evol_anis_tag,gamma_ext),resume=False)

# Copy the anisotropy modefile into the results directory
if args.anis_modefile is not None:
    os.system('cp {0} {1}'.format(args.anis_modefile,'./chains_nanoAnalysis/nanograv_gwb{0}_red{1}_nmodes{2}_Lmax{3}_{4}_{5}'.\
                                  format(args.limit_or_detect_gwb,args.limit_or_detect_red,args.nmodes,args.LMAX,evol_anis_tag,gamma_ext)))


#####################################
#####################################

# MCMC jump proposals
# red noise draws (from Justin Ellis' PAL2)

#####################################
#####################################

def drawFromRedNoisePrior(parameters, iter, beta):

    # post-jump parameters
    q = parameters.copy()

    # transition probability
    qxy = 0

    npsr = len(psr)

    if args.dmVar==True:
        
        if args.fix_slope==False:
        
            (Ared_samp, gam_red_samp, 
             Adm_samp, gam_dm_samp, 
             Agwb_samp, gam_gwb_samp, orf_coeffs_samp) = utils.masterSplitParams(q, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ul, gam_red_ul, 
             Adm_ul, gam_dm_ul, 
             Agwb_ul, gam_gwb_ul, orf_coeffs_ul) = utils.masterSplitParams(pmax, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ll, gam_red_ll, 
             Adm_ll, gam_dm_ll, 
             Agwb_ll, gam_gwb_ll, orf_coeffs_ll) = utils.masterSplitParams(pmin, npsr, args.dmVar, args.fix_slope, propose=True)
            
        else:
            
            (Ared_samp, gam_red_samp, 
             Adm_samp, gam_dm_samp, 
             Agwb_samp, orf_coeffs_samp) = utils.masterSplitParams(q, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ul, gam_red_ul, 
             Adm_ul, gam_dm_ul, 
             Agwb_ul, orf_coeffs_ul) = utils.masterSplitParams(pmax, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ll, gam_red_ll, 
             Adm_ll, gam_dm_ll, 
             Agwb_ll, orf_coeffs_ll) = utils.masterSplitParams(pmin, npsr, args.dmVar, args.fix_slope, propose=True)
        
    else:
        
        if args.fix_slope==False:
        
            (Ared_samp, gam_red_samp, 
             Agwb_samp, gam_gwb_samp, orf_coeffs_samp) = utils.masterSplitParams(q, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ul, gam_red_ul, 
             Agwb_ul, gam_gwb_ul, orf_coeffs_ul) = utils.masterSplitParams(pmax, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ll, gam_red_ll, 
             Agwb_ll, gam_gwb_ll, orf_coeffs_ll) = utils.masterSplitParams(pmin, npsr, args.dmVar, args.fix_slope, propose=True)
            
        else:
            
            (Ared_samp, gam_red_samp, 
             Agwb_samp, orf_coeffs_samp) = utils.masterSplitParams(q, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ul, gam_red_ul, 
             Agwb_ul, orf_coeffs_ul) = utils.masterSplitParams(pmax, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ll, gam_red_ll, 
             Agwb_ll, orf_coeffs_ll) = utils.masterSplitParams(pmin, npsr, args.dmVar, args.fix_slope, propose=True)

    
    # log prior
    if args.limit_or_detect_red == 'detect':
        
        Ared_samp = np.random.uniform(Ared_ll, Ared_ul, len(Ared_samp))
        qxy += 0
        
    elif args.limit_or_detect_red == 'limit':
        
        Ared_samp = np.random.uniform(Ared_ll, Ared_ul, len(Ared_samp))
        qxy += 0

        #Ared = np.log10(np.random.uniform(10 ** Ared_ll, 10 ** Ared_ul, len(Ared)))
        #qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])

    gam_red_samp = np.random.uniform(gam_red_ll, gam_red_ul, len(gam_red_samp))
    qxy += 0
    
    
    if args.dmVar==True:
    
        if args.fix_slope==False:
        
            q = np.concatenate([Ared_samp, gam_red_samp,
                                Adm_samp, gam_dm_samp,
                                np.array([Agwb_samp]), np.array([gam_gwb_samp]), orf_coeffs_samp])
        else:
        
            q = np.concatenate([Ared_samp, gam_red_samp,
                                Adm_samp, gam_dm_samp,
                                np.array([Agwb_samp]), orf_coeffs_samp])
        
    else:
    
        if args.fix_slope==False:
    
            q = np.concatenate([Ared_samp, gam_red_samp,
                                np.array([Agwb_samp]), np.array([gam_gwb_samp]), orf_coeffs_samp])
        
        else:
            
            q = np.concatenate([Ared_samp, gam_red_samp,
                            np.array([Agwb_samp]), orf_coeffs_samp])
    

    return q, qxy

# gwb draws (from Justin Ellis' PAL2)

def drawFromGWBPrior(parameters, iter, beta):

    # post-jump parameters
    q = parameters.copy()

    # transition probability
    qxy = 0

    npsr = len(psr)
    
    if args.dmVar==True:
        
        if args.fix_slope==False:
        
            (Ared_samp, gam_red_samp, 
             Adm_samp, gam_dm_samp, 
             Agwb_samp, gam_gwb_samp, orf_coeffs_samp) = utils.masterSplitParams(q, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ul, gam_red_ul, 
             Adm_ul, gam_dm_ul, 
             Agwb_ul, gam_gwb_ul, orf_coeffs_ul) = utils.masterSplitParams(pmax, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ll, gam_red_ll, 
             Adm_ll, gam_dm_ll, 
             Agwb_ll, gam_gwb_ll, orf_coeffs_ll) = utils.masterSplitParams(pmin, npsr, args.dmVar, args.fix_slope, propose=True)
            
        else:
            
            (Ared_samp, gam_red_samp, 
             Adm_samp, gam_dm_samp, 
             Agwb_samp, orf_coeffs_samp) = utils.masterSplitParams(q, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ul, gam_red_ul, 
             Adm_ul, gam_dm_ul, 
             Agwb_ul, orf_coeffs_ul) = utils.masterSplitParams(pmax, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ll, gam_red_ll, 
             Adm_ll, gam_dm_ll, 
             Agwb_ll, orf_coeffs_ll) = utils.masterSplitParams(pmin, npsr, args.dmVar, args.fix_slope, propose=True)
        
    else:
        
        if args.fix_slope==False:
        
            (Ared_samp, gam_red_samp, 
             Agwb_samp, gam_gwb_samp, orf_coeffs_samp) = utils.masterSplitParams(q, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ul, gam_red_ul, 
             Agwb_ul, gam_gwb_ul, orf_coeffs_ul) = utils.masterSplitParams(pmax, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ll, gam_red_ll, 
             Agwb_ll, gam_gwb_ll, orf_coeffs_ll) = utils.masterSplitParams(pmin, npsr, args.dmVar, args.fix_slope, propose=True)
            
        else:
            
            (Ared_samp, gam_red_samp, 
             Agwb_samp, orf_coeffs_samp) = utils.masterSplitParams(q, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ul, gam_red_ul, 
             Agwb_ul, orf_coeffs_ul) = utils.masterSplitParams(pmax, npsr, args.dmVar, args.fix_slope, propose=True)
        
            (Ared_ll, gam_red_ll, 
             Agwb_ll, orf_coeffs_ll) = utils.masterSplitParams(pmin, npsr, args.dmVar, args.fix_slope, propose=True)

    
    # log prior
    if args.limit_or_detect_gwb == 'detect':
        
        Agwb_samp = np.random.uniform(Agwb_ll, Agwb_ul)
        qxy += 0

    elif args.limit_or_detect_gwb == 'limit':
        
        Agwb_samp = np.random.uniform(Agwb_ll, Agwb_ul)
        qxy += 0

        #Ared = np.log10(np.random.uniform(10 ** Ared_ll, 10 ** Ared_ul, len(Ared)))
        #qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])


    if args.fix_slope==False:
        gam_gwb_samp = np.random.uniform(gam_gwb_ll, gam_gwb_ul)
        qxy += 0
        
        
        
    if args.dmVar==True:
    
        if fix_slope==False:
        
            q = np.concatenate([Ared_samp, gam_red_samp,
                                Adm_samp, gam_dm_samp,
                                np.array([Agwb_samp]), np.array([gam_gwb_samp]), orf_coeffs_samp])
        else:
        
            q = np.concatenate([Ared_samp, gam_red_samp,
                                Adm_samp, gam_dm_samp,
                                np.array([Agwb_samp]), orf_coeffs_samp])
        
    else:
    
        if args.fix_slope==False:
    
            q = np.concatenate([Ared_samp, gam_red_samp,
                                np.array([Agwb_samp]), np.array([gam_gwb_samp]), orf_coeffs_samp])
        
        else:
            
            q = np.concatenate([Ared_samp, gam_red_samp,
                            np.array([Agwb_samp]), orf_coeffs_samp])


    return q, qxy



# add jump proposals
sampler.addProposalToCycle(drawFromGWBPrior, 10)
#if args.incGWBAni and args.gwbModel == 'powerlaw':
#    sampler.addProposalToCycle(model.drawFromaGWBPrior, 10)
sampler.addProposalToCycle(drawFromRedNoisePrior, 10)
#if args.incDMBand and args.dmModel=='powerlaw':
#    sampler.addProposalToCycle(model.drawFromDMNoiseBandPrior, 5)
#if args.incORF:
#    sampler.addProposalToCycle(model.drawFromORFPrior, 10)


sampler.sample(p0=x0,Niter=1e6,thin=10)
