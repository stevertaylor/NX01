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
import glob
import NX01_AnisCoefficients as anis
import NX01_utils as utils
import NX01_psr

parser = optparse.OptionParser(description = 'NX01 - Precursor to the PANTHER Group ENTERPRISE project')

############################
############################

parser.add_option('--nmodes', dest='nmodes', action='store', type=int, default=50,
                   help='Number of modes in low-rank time-frequency approximation (default = 50 modes)')
parser.add_option('--use-gpu', dest='use_gpu', action='store_true', default=False,
                  help='Do you want to use the GPU for accelerated linear algebra? (default = False)')
parser.add_option('--fix-slope', dest='fix_slope', action='store_true', default=False,
                  help='Do you want to fix the slope of the GWB spectrum? (default = False)')
parser.add_option('--limit-or-detect-gwb', dest='limit_or_detect_gwb', action='store', type=str, default='limit',
                   help='Do you want to use a uniform prior on log_10(Agwb) [detect] or Agwb itself [upper-limit] (default=\'limit\')?')
parser.add_option('--limit-or-detect-red', dest='limit_or_detect_red', action='store', type=str, default='detect',
                   help='Do you want to use a uniform prior on log_10(Ared) [detect] or Ared itself [upper-limit] (default=\'detect\')?')
parser.add_option('--sample-or-maximize', dest='sample_or_maximize', action='store', type=str, default='maximize',
                   help='Do you want sample from the posteror distribution or just find the maximum likelihood noise values? (default=\'maximize\')?')
parser.add_option('--path', dest='path', action='store', type=str,
                   help='Provide the path to the directory contaning all simulated par and tim files.')

(args, x) = parser.parse_args()

if args.use_gpu:
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import pycuda.elementwise as el
    import pycuda.tools as tools
    import scikits.cuda.cublas as cublas
    import scikits.cuda.cula as cula
    import scikits.cuda.linalg as culinalg
    import scikits.cuda.misc as cumisc

    culinalg.init()

master_path = os.getcwd()
if args.path:
    path = args.path
else:
    path = os.getcwd()


if args.sample_or_maximize == 'maximize':
    from pyswarm import pso
else:    
    if not os.path.exists('chains_EPTAsims'):
        os.makedirs('chains_EPTAsims')

par_names=[]
tim_names=[]
for file in glob.glob(path+'*.par'):
    par_names.append(file)
for file in glob.glob(path+'*.tim'):
    tim_names.append(file)

################################################################################################################################
# PASSING THROUGH TEMPO2 VIA libstempo
################################################################################################################################

t2psr=[]
for ii in range(len(pulsars)):
    t2psr.append(T2.tempopulsar(parfile=par_names[ii],\
                                timfile=tim_names[ii]))
    t2psr[ii].fit(iters=10)
    if np.any(np.isfinite(t2psr[ii].residuals())==False)==True:
        t2psr[ii] = T2.tempopulsar(parfile=par_names[ii],\
                                   timfile=tim_names[ii])
   
################################################################################################################################
# MAKING A PULSAR OBJECT, THEN GRABBING ALL THE VARIABLES, e.g. toas, residuals, error-bars, designmatrices etc.
################################################################################################################################

psr = [NX01_psr.PsrObj(t2psr[ii]) for ii in range(len(t2psr))]

[psr[ii].grab_all_vars() for ii in range(len(psr))]

psr_positions = [np.array([psr[ii].psr_locs[0], np.pi/2. - psr[ii].psr_locs[1]]) for ii in range(len(psr))]
positions = np.array(psr_positions).copy()
psr_positions = np.array([np.array([np.sin(psr_positions[ii][1])*np.cos(psr_positions[ii][0]),np.sin(psr_positions[ii][1])*np.sin(psr_positions[ii][0]),np.cos(psr_positions[ii][1])]) for ii in range(len(psr_positions))])

AngSepGrid = np.zeros((len(psr),len(psr)))
for ii in range(len(psr)):
    for jj in range(len(psr)):
        AngSepGrid[ii,jj] = np.dot(psr_positions[ii],psr_positions[jj])

xsep = 0.5*(1.0-AngSepGrid)
ORF = 1.5*xsep*np.log(xsep) - 0.25*xsep + 0.5 + 0.5*np.eye(len(psr))

for ii in range(len(psr)):
    for jj in range(len(psr)):
        if ii==jj:
            ORF[ii,jj] = 1.0
        else:
            ORF[jj,ii] = ORF[ii,jj]
   
################################################################################################################################
# GETTING MAXIMUM TIME, COMPUTING FOURIER DESIGN MATRICES, AND GETTING MODES 
################################################################################################################################
Tmax = np.max([psr[p].toas.max() - psr[p].toas.min() for p in range(len(psr))])

# initialize fourier design matrices
[psr[ii].makeFtot(args.nmodes, Tmax) for ii in range(len(psr))]
F = [psr[ii].Ftot for ii in range(len(psr))]

# get GW frequencies
fqs = np.linspace(1/Tmax, args.nmodes/Tmax, args.nmodes)

################################################################################################################################
# MAKE FIXED NOISE MATRICES FROM MAXIMUM-LIKELIHOOD VALUES OF SINGLE-PULSAR ANALYSIS
################################################################################################################################
Diag=[]
res_prime=[]
F_prime=[]
for ii in range(len(psr)):   
    psr[ii].two_comp_noise(MLerrors=psr[ii].toaerrs)
    Diag.append( psr[ii].diag_white )
    res_prime.append( psr[ii].res_prime )
    F_prime.append( psr[ii].Ftot_prime )

################################################################################################################################
# SETTING UP PRIOR RANGES
################################################################################################################################

pmin = np.array([-20.0])
if args.fix_slope is False:
    pmin = np.append(pmin,[0.0])
pmin = np.append(pmin,-20.0*np.ones(len(psr)))
pmin = np.append(pmin,0.0*np.ones(len(psr)))

pmax = np.array([-10.0])
if args.fix_slope is False:
    pmax = np.append(pmax,[7.0])
pmax = np.append(pmax,-10.0*np.ones(len(psr)))
pmax = np.append(pmax,7.0*np.ones(len(psr)))

##################################################################################

loglike1 = 0
FtNF = []
for p in range(len(psr)):
        
    # compute d
    if p == 0:
        d = np.dot(F_prime[p].T, res_prime[p]/( Diag[p] ))
    else:
        d = np.append(d, np.dot(F_prime[p].T, res_prime[p]/( Diag[p] )))

    # compute FT N F
    N = 1./( Diag[p] )
    right = (N*F_prime[p].T).T
    FtNF.append(np.dot(F_prime[p].T, right))

    # log determinant of N
    logdet_N = np.sum(np.log( Diag[p] ))
        
    # triple product in likelihood function
    dtNdt = np.sum(res_prime[p]**2.0/( Diag[p] ))

    loglike1 += -0.5 * (logdet_N + dtNdt)

###################################################################################

def my_prior(x):
    logp = 0.
    
    if np.all(x <= pmax) and np.all(x >= pmin):
        logp = np.sum(np.log(1/(pmax-pmin)))
    else:
        logp = -np.inf
    
    return logp


def modelIndependentFullPTANoisePL(x):
    """
    Model Independent stochastic background likelihood function

    """ 

    Agwb = 10.0**x[0]
    if args.fix_slope:
        gam_gwb = 13./3.
        ct = 1
    else:
        gam_gwb = x[1]
        ct = 2
    #####
    Ared = 10.0**x[ct:ct+len(psr)]
    gam_red = x[ct+len(psr):ct+2*len(psr)]
    ###################

    npsr = len(psr)

    # parameterize intrinsic red noise as power law
    Tspan = (1/fqs[0])*86400.0
    f1yr = 1/3.16e7
    rho = np.log10(Agwb**2/12/np.pi**2 * f1yr**(gam_gwb-3) * (fqs/86400.0)**(-gam_gwb)/Tspan)

    # parameterize intrinsic red-noise and DM-variations as power law
    kappa = [] 
    for ii in range(npsr):
        kappa.append(np.log10( Ared[ii]**2/12/np.pi**2 * f1yr**(gam_red[ii]-3) * (fqs/86400.0)**(-gam_red[ii])/Tspan ))

    # construct elements of sigma array
    sigdiag = []
    sigoffdiag = []
    for ii in range(npsr):
        tot = np.zeros(2*args.nmodes)
        offdiag = np.zeros(2*args.nmodes)

        # off diagonal terms
        offdiag[0::2] = 10**rho
        offdiag[1::2] = 10**rho

        # diagonal terms
        tot[0::2] = ORF[ii,ii]*10**rho + 10**kappa[ii]
        tot[1::2] = ORF[ii,ii]*10**rho + 10**kappa[ii]
                
        # fill in lists of arrays
        sigdiag.append(tot)
        sigoffdiag.append(offdiag)


    # compute Phi inverse from Lindley's code
    smallMatrix = np.zeros((2*args.nmodes, npsr, npsr))
    for ii in range(npsr):
        for jj in range(ii,npsr):

            if ii == jj:
                smallMatrix[:,ii,jj] = sigdiag[jj] 
            else:
                smallMatrix[:,ii,jj] = ORFtot[ii,jj] * sigoffdiag[jj] 
                smallMatrix[:,jj,ii] = smallMatrix[ii,jj]

    
    # invert them
    logdet_Phi = 0
    non_pos_def = 0
    for ii in range(2*args.nmodes):
        try:
            L = sl.cho_factor(smallMatrix[ii,:,:])
            smallMatrix[ii,:,:] = sl.cho_solve(L, np.eye(npsr))
            logdet_Phi += np.sum(2*np.log(np.diag(L[0])))
        except np.linalg.LinAlgError:
            print 'Cholesky Decomposition Failed!! Rejecting...'
            non_pos_def += 1
        
    if non_pos_def > 0:
        return -np.inf
    else:
        nftot = 2*args.nmodes
        Phi = np.zeros((npsr*nftot, npsr*nftot))
        # now fill in real covariance matrix
        ind = [np.arange(kk*nftot, kk*nftot+nftot) for kk in range(npsr)]
        for ii in range(npsr):
            for jj in range(npsr):
                Phi[ind[ii],ind[jj]] = smallMatrix[:,ii,jj]
            
        # compute sigma
        Sigma = sl.block_diag(*FtNF) + Phi

            
        # cholesky decomp for second term in exponential
        if args.use_gpu:
            try:
                Sigma_gpu = gpuarray.to_gpu( Sigma.astype(np.float64).copy() )
                expval2_gpu = gpuarray.to_gpu( d.astype(np.float64).copy() )
                culinalg.cho_solve( Sigma_gpu, expval2_gpu ) # in-place linear-algebra: Sigma and expval2 overwritten
                logdet_Sigma = np.sum(2.0*np.log(np.diag(Sigma_gpu.get())))

            except cula.culaDataError:
                print 'Cholesky Decomposition Failed (GPU error!!!!!!!!!!)'
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


        if args.limit_or_detect_gwb == 'limit':
            prior_factor = np.log(Agwb * np.log(10.0))
        else:
            prior_factor = 0.0

        if args.limit_or_detect_red == 'limit':
            prior_factor += sum( np.log(Ared[ii] * np.log(10.0)) for ii in range(npsr) )
        else:
            prior_factor += 0.0
            
        return logLike + prior_factor 


#########################
#########################

parameters = ["Agwb"]
if args.fix_slope is False:
    parameters.append("gam_gwb")
    gamma_ext = 'GamVary'
else:
    gamma_ext = 'Gam4p33'

for ii in range(len(psr)):
    parameters.append('log(Ared_'+psr[ii].name+')')
for ii in range(len(psr)):
    parameters.append('gam_red_'+psr[ii].name)

print "\n You are searching for the following parameters: {0}\n".format(parameters)
n_params = len(parameters)

print "\n The total number of parameters is {0}\n".format(n_params)


x0 = np.array([-15.0])
if args.fix_slope is False:
    x0 = np.append(x0,[13./3.])

for ii in range(len(psr)):
    x0 = np.append(x0,[-15.0])
for ii in range(len(psr)):
    x0 = np.append(x0,[3.0])
    

print "\n Your initial parameters are {0}\n".format(x0)

print "\n Running a quick profile on the likelihood to estimate evaluation speed...\n"
cProfile.run('modelIndependentFullPTANoisePL(x0)')

#################################
# Now, we sample or maximize.....
#################################

if args.sample_or_maximize == 'maximize':

    def swarm_prob(x):
        return -ln_prob(x)

    xopt, fopt = pso(swarm_prob, pmin, pmax, swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=1000, debug=True)

    print "\n Printing out the ML parameter values for the simulated PTA dataset...\n".format(psr.name)
    fil = open("EPTAsims_{1}GWB_{2}Red_nmodes{2}_{4}.txt".format(args.limit_or_detect_gwb,args.limit_or_detect_red,args.nmodes,gamma_ext),'w')
    for ii in range(len(xopt)):
        print "{0} = {1}".format(parameters[ii],xopt[ii])
        print >>fil, "{0} {1}".format(parameters[ii],xopt[ii])
    print "\n Values saved in EPTAsims_{1}GWB_{2}Red_nmodes{2}_{4}.txt".format(args.limit_or_detect_gwb,args.limit_or_detect_red,args.nmodes,gamma_ext)

else:

    cov_diag = np.array([0.5])
    if args.fix_slope is False:
        cov_diag = np.append(cov_diag,[0.5])

    for ii in range(len(psr)):
        cov_diag = np.append(cov_diag,[0.5])
    for ii in range(len(psr)):
        cov_diag = np.append(cov_diag,[0.5])

    print "\n Now, we sample... \n"
    sampler = PAL.PTSampler(ndim=n_params,logl=modelIndependentFullPTANoisePL,logp=my_prior,cov=np.diag(cov_diag),\
                            outDir='./chains_EPTAsims/EPTAsims_{1}GWB_{2}Red_nmodes{2}_{4}'.format(args.limit_or_detect_gwb,args.limit_or_detect_red,args.nmodes,gamma_ext),resume=False)
    sampler.sample(p0=x0,Niter=500000,thin=10)
