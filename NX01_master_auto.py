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

parser.add_option('--nmodes', dest='nmodes', action='store', type=int, default=50,
                   help='Number of modes in low-rank time-frequency approximation (default = 50 modes)')
parser.add_option('--num_gwfreq_wins', dest='num_gwfreq_wins', action='store', type=int, default=1,
                   help='Number windows to split the band into (useful for evolving anisotropic searches (default = 1 windows)')
parser.add_option('--lmax', dest='LMAX', action='store', type=int, default=0,
                   help='Maximum multipole in anisotropic search (default = 0, i.e. isotropic-search)')
parser.add_option('--use-gpu', dest='use_gpu', action='store_true', default=False,
                  help='Do you want to use the GPU for accelerated linear algebra? (default = False)')
parser.add_option('--fix-slope', dest='fix_slope', action='store_true', default=False,
                  help='Do you want to fix the slope of the GWB spectrum? (default = False)')
parser.add_option('--mean-or-max', dest='mean_or_max', action='store', type=str,
                   help='Do you want to use the .par files with mean or max-likelihood white-noise parameters?')
parser.add_option('--snr-tag', dest='snr_tag', action='store', type=float, default=0.9, 
                   help='Do you want the 90%, 95% or 100% SNR dataset? [6, 11, and 41 pulsars respectively] (default=0.90)')
parser.add_option('--limit-or-detect', dest='limit_or_detect', action='store', type=str, default='limit',
                   help='Do you want to use a uniform prior on log_10(Agwb) [detect] or Agwb itself [upper-limit] (default=\'limit\')?')

(args, x) = parser.parse_args()

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

master_path = os.getcwd()
path = '/Users/staylor/Research/EPTAv2/UniEQ'  

if args.snr_tag == 0.9:
    dir = ['J1909-3744', 'J1713+0747', 'J1744-1134', 'J0613-0200', 'J1600-3053', 'J1012+5307']   #gives 90% of total SNR^2
    snr_tag_ext = '90pct'
elif args.snr_tag == 0.95:
    dir = ['J1909-3744', 'J1713+0747', 'J1744-1134', 'J0613-0200', 'J1600-3053', 'J1012+5307', \
           'J1640+2224', 'J2145-0750', 'J1857+0943', 'J1022+1001', 'J0030+0451'] # gives 95% of total SNR^2
    snr_tag_ext = '95pct'
elif args.snr_tag == 1.0:
    os.chdir(path)
    dir = os.walk('.').next()[1]
    dir.remove('J1939+2134')
    os.chdir(master_path)
    snr_tag_ext = '100pct'

if not os.path.exists('chains_Analysis'):
    os.makedirs('chains_Analysis')

pulsars = [s for s in dir if "J" in s]
pulsars.sort()

print pulsars
################################################################################################################################
# PASSING THROUGH TEMPO2 VIA libstempo
################################################################################################################################
if args.mean_or_max == 'mean':
    par_ext = 'Mean'
elif args.mean_or_max == 'max':
    par_ext = 'ML'

t2psr=[]
for ii in range(len(pulsars)):
    os.chdir(path+'/'+pulsars[ii])
    if os.path.isfile('{0}_NoAFB.par'.format(pulsars[ii])):
        t2psr.append(T2.tempopulsar(parfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_TD.{0}.par'.format(par_ext),\
                                    timfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_NoAFB.tim'))
    else:
        t2psr.append(T2.tempopulsar(parfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_TD.{0}.par'.format(par_ext),\
                                    timfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_all.tim'))
    os.chdir(path)
    t2psr[ii].fit(iters=10)
    if np.any(np.isfinite(t2psr[ii].residuals())==False)==True:
        os.chdir(path+'/'+pulsars[ii])
        if os.path.isfile('{0}_NoAFB.par'.format(pulsars[ii])):
            t2psr[ii] = T2.tempopulsar(parfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_TD.{0}.par'.format(par_ext),\
                                       timfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_NoAFB.tim')
        else:
            t2psr[ii] = T2.tempopulsar(parfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_TD.{0}.par'.format(par_ext),\
                                       timfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_all.tim')
        os.chdir(path)

os.chdir(master_path)

################################################################################################################################
# MAKING A PULSAR OBJECT, THEN GRABBING ALL THE VARIABLES, e.g. toas, residuals, error-bars, designmatrices etc.
################################################################################################################################

psr = [NX01_psr.PsrObj(t2psr[ii]) for ii in range(len(t2psr))]

[psr[ii].grab_all_vars() for ii in range(len(psr))]

psr_positions = [np.array([psr[ii].psr_locs[0], np.pi/2. - psr[ii].psr_locs[1]]) for ii in range(len(psr))]
positions = np.array(psr_positions).copy()

CorrCoeff = np.array(anis.CorrBasis(positions,args.LMAX))       # computing all the correlation basis-functions for the array
harm_sky_vals = utils.SetupPriorSkyGrid(args.LMAX)              # computing the values of the spherical-harmonics up to order
                                                                # LMAX on a pre-specified grid
gwfreqs_per_win = int(1.*args.nmodes/(1.*args.num_gwfreq_wins)) # getting the number of GW frequencies per window
        
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
# FORM A LIST COMPOSED OF NP ARRAYS CONTAINING THE INDEX POSITIONS WHERE EACH UNIQUE 'sys' BACKEND IS APPLIED
################################################################################################################################

backends = []
[psr[ii].get_backends() for ii in range(len(psr))]
for ii in range(len(psr)):
    backends.append(psr[ii].bkends)

################################################################################################################################
# GETTING MAXIMUM-LIKELIHOOD VALUES OF SINGLE-PULSAR ANALYSIS FOR OUR STARTING POINT
################################################################################################################################

Adm_ML=[]
gam_dm_ML=[]
Ared_ML=[]
gam_red_ML=[]
EFAC_ML = [[0.0]*len(backends[jj]) for jj in range(len(backends))]
EQUAD_ML = [[0.0]*len(backends[jj]) for jj in range(len(backends))]
for ii in range(len(pulsars)):
    with open(path+'/{0}/{0}_Taylor_TimeDomain_model1.txt'.format(psr[ii].name), 'r') as f:
        Adm_ML.append(float(f.readline().split()[3]))
        gam_dm_ML.append(float(f.readline().split()[3]))
        Ared_ML.append(float(f.readline().split()[3]))
        gam_red_ML.append(float(f.readline().split()[3]))
        for jj in range(len(backends[ii])):
            EFAC_ML[ii][jj] = float(f.readline().split()[3])
        for jj in range(len(backends[ii])):
            EQUAD_ML[ii][jj] = float(f.readline().split()[3])


################################################################################################################################
# GETTING MEAN AND ERROR-BARS VALUES OF SINGLE-PULSAR ANALYSIS FOR OUR INITIAL PARAMETER COVARIANCE ESTIMATE
################################################################################################################################
Adm_mean=[]
Adm_err=[]
gam_dm_mean=[]
gam_dm_err=[]
Ared_mean=[]
Ared_err=[]
gam_red_mean=[]
gam_red_err=[]
EFAC_mean = [[0.0]*len(backends[jj]) for jj in range(len(backends))]
EFAC_err = [[0.0]*len(backends[jj]) for jj in range(len(backends))]
EQUAD_mean = [[0.0]*len(backends[jj]) for jj in range(len(backends))]
EQUAD_err = [[0.0]*len(backends[jj]) for jj in range(len(backends))]
for ii in range(len(pulsars)):
    with open(path+'/{0}/{0}_Taylor_TimeDomain_model1.txt'.format(psr[ii].name), 'r') as f:
        line = f.readline().split()
        Adm_mean.append( 0.5 * (np.log10(float(line[5])) + np.log10(float(line[4]))) ) # the means and error bars will be in log10
        Adm_err.append( 0.5 * (np.log10(float(line[5])) - np.log10(float(line[4]))) )

        line = f.readline().split()
        gam_dm_mean.append( 0.5 * (np.log10(float(line[5])) + np.log10(float(line[4]))) )
        gam_dm_err.append( 0.5 * (np.log10(float(line[5])) - np.log10(float(line[4]))) )
        
        line = f.readline().split()
        Ared_mean.append( 0.5 * (np.log10(float(line[5])) + np.log10(float(line[4]))) )
        Ared_err.append( 0.5 * (np.log10(float(line[5])) - np.log10(float(line[4]))) )
        
        line = f.readline().split()
        gam_red_mean.append( 0.5 * (np.log10(float(line[5])) + np.log10(float(line[4]))) )
        gam_red_err.append( 0.5 * (np.log10(float(line[5])) - np.log10(float(line[4]))) )
        for jj in range(len(backends[ii])):
            line = f.readline().split()
            EFAC_mean[ii][jj] = 0.5 * (np.log10(float(line[5])) + np.log10(float(line[4])))
            EFAC_err[ii][jj] = 0.5 * (np.log10(float(line[5])) - np.log10(float(line[4])))
        for jj in range(len(backends[ii])):
            line = f.readline().split()
            EQUAD_mean[ii][jj] = 0.5 * (np.log10(float(line[5])) + np.log10(float(line[4])))
            EQUAD_err[ii][jj] = 0.5 * (np.log10(float(line[5])) - np.log10(float(line[4])))
    f.close()

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
pmin = np.append(pmin,-20.0*np.ones(len(psr)))
pmin = np.append(pmin,0.0*np.ones(len(psr)))
pmin = np.append(pmin,0.1*np.ones(len(psr)))
pmin = np.append(pmin,-20.0)
pmin = np.append(pmin,0.0)
pmin = np.append(pmin,-20.0)
pmin = np.append(pmin,0.0)
pmin = np.append(pmin,-10.0*np.ones( args.num_gwfreq_wins*(((args.LMAX+1)**2)-1) ))

pmax = np.array([-10.0])
if args.fix_slope is False:
    pmax = np.append(pmax,[7.0])
pmax = np.append(pmax,-10.0*np.ones(len(psr)))
pmax = np.append(pmax,7.0*np.ones(len(psr)))
pmax = np.append(pmax,-10.0*np.ones(len(psr)))
pmax = np.append(pmax,7.0*np.ones(len(psr)))
pmax = np.append(pmax,10.0*np.ones(len(psr)))
pmax = np.append(pmax,-10.0)
pmax = np.append(pmax,7.0)
pmax = np.append(pmax,-10.0)
pmax = np.append(pmax,7.0)
pmax = np.append(pmax,10.0*np.ones( args.num_gwfreq_wins*(((args.LMAX+1)**2)-1) ))
##################################################################################################################################

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
    Adm = 10.0**x[ct+2*len(psr):ct+3*len(psr)]
    gam_dm = x[ct+3*len(psr):ct+4*len(psr)]
    EFAC = x[ct+4*len(psr):ct+5*len(psr)]
    Acm = 10.0**x[ct+5*len(psr)]
    gam_cm = x[ct+5*len(psr) + 1]
    Aun = 10.0**x[ct+5*len(psr) + 2]
    gam_un = x[ct+5*len(psr) + 3]
    ###################
    orf_coeffs = x[ct+5*len(psr) + 4:]
    orf_coeffs = orf_coeffs.reshape((args.num_gwfreq_wins,((args.LMAX+1)**2)-1))
    clm = np.array([[0.0]*((args.LMAX+1)**2) for ii in range(args.num_gwfreq_wins)])
    clm[:,0] = 2.0*np.sqrt(np.pi)
    physicality = 0.
    if args.LMAX!=0:
        for kk in range(args.num_gwfreq_wins):
            for ii in range(1,((args.LMAX+1)**2)):
                clm[kk,ii] = orf_coeffs[kk,ii-1]   
    
            if (utils.PhysPrior(clm[kk],harm_sky_vals) == 'Unphysical'):
                physicality += -10.0**7.0
            else:
                physicality += 0.

    npsr = len(psr)

    ORF=[]
    for ii in range(args.num_gwfreq_wins): # number of frequency windows
        for jj in range(gwfreqs_per_win): # number of frequencies in this window
            ORF.append( sum(clm[ii,kk]*CorrCoeff[kk] for kk in range(len(CorrCoeff))) )
    for ii in range(args.num_gwfreq_wins): # number of frequency windows
        for jj in range(gwfreqs_per_win): # number of frequencies in this window
            ORF.append( np.zeros((npsr,npsr)) )

    ORF = np.array(ORF)
    ORFtot = np.zeros((4*args.nmodes,npsr,npsr)) # shouldn't be applying ORF to dmfreqs, but the projection of GW spec onto dmfreqs is defined as zero below
    ORFtot[0::2] = ORF
    ORFtot[1::2] = ORF

    loglike1 = 0
    FtNF = []
    d = []
    for p in range(len(psr)):

        d.append( np.dot(F_prime[p].T, res_prime[p]/( (EFAC[p]**2.0)*Diag[p] )) )

        # compute FT N F
        N = 1./( (EFAC[p]**2.0)*Diag[p] )
        right = (N*F_prime[p].T).T
        FtNF.append(np.dot(F_prime[p].T, right))

        # log determinant of N
        logdet_N = np.sum(np.log( (EFAC[p]**2.0)*Diag[p] ))
        
        # triple product in likelihood function
        dtNdt = np.sum(res_prime[p]**2.0/( (EFAC[p]**2.0)*Diag[p] ))

        loglike1 += -0.5 * (logdet_N + dtNdt)
    
    
    # parameterize intrinsic red noise as power law
    Tspan = (1/fqs[0])*86400.0
    f1yr = 1/3.16e7
    rho = np.log10(Agwb**2/12/np.pi**2 * f1yr**(gam_gwb-3) * (fqs/86400.0)**(-gam_gwb)/Tspan)

    # spectrum of common-mode
    cm = np.log10(Acm**2/12/np.pi**2 * f1yr**(gam_cm-3) * (fqs/86400.0)**(-gam_cm)/Tspan)

    # spectrum of common uncorrelated red-noise
    un = np.log10(Aun**2/12/np.pi**2 * f1yr**(gam_un-3) * (fqs/86400.0)**(-gam_un)/Tspan)

    # parameterize intrinsic red-noise and DM-variations as power law
    kappa = [] 
    for ii in range(npsr):
        kappa.append(np.log10( np.append( Ared[ii]**2/12/np.pi**2 * f1yr**(gam_red[ii]-3) * (fqs/86400.0)**(-gam_red[ii])/Tspan,\
                                          Adm[ii]**2/12/np.pi**2 * f1yr**(gam_dm[ii]-3) * (fqs/86400.0)**(-gam_dm[ii])/Tspan ) ))

    # construct elements of sigma array
    sigdiag = []
    sigoffdiag = []
    sigcm = []
    for ii in range(npsr):
        tot = np.zeros(4*args.nmodes)

        # diagonal terms
        tot[0::2] = ORF[:,ii,ii]*np.append( 10**rho, np.zeros(len(rho)) ) + np.append( 10**cm + 10**un, np.zeros(len(rho)) ) + 10**kappa[ii]
        tot[1::2] = ORF[:,ii,ii]*np.append( 10**rho, np.zeros(len(rho)) ) + np.append( 10**cm + 10**un, np.zeros(len(rho)) ) + 10**kappa[ii]
                
        # fill in lists of arrays
        sigdiag.append(tot)


    # invert them
    logdet_Phi = 0
    Sigma=[]
    for ii in range(npsr):
        logdet_Phi += np.sum(np.log(sigdiag[ii]))
        Sigma.append( FtNF[ii] + np.diag(1./sigdiag[ii]) )
      

    # cholesky decomp for second term in exponential
    if args.use_gpu:
        logdet_Sigma = 0.
        dSigd = 0.
        for ii in range(npsr):
            try:
                Sigma_gpu = gpuarray.to_gpu( Sigma[ii].astype(np.float64).copy() )
                expval2_gpu = gpuarray.to_gpu( d[ii].astype(np.float64).copy() )
                culinalg.cho_solve( Sigma_gpu, expval2_gpu ) # in-place linear-algebra: Sigma and expval2 overwritten
                dSigd += np.dot(d[ii], expval2_gpu.get())
                logdet_Sigma += np.sum(2.0*np.log(np.diag(Sigma_gpu.get())))

            except cula.culaDataError:
                print 'Cholesky Decomposition Failed (GPU error!!!!!!!!!!)'
                return -np.inf

        logLike = -0.5 * (logdet_Phi + logdet_Sigma) + 0.5 * dSigd + loglike1
            
    else:
        logdet_Sigma = 0.
        dSigd = 0.
        for ii in range(npsr):
            try:
                cf = sl.cho_factor(Sigma[ii])
                expval2 = sl.cho_solve(cf, d[ii])
                dSigd += np.dot(d[ii], expval2)
                logdet_Sigma += np.sum(2*np.log(np.diag(cf[0])))

            except np.linalg.LinAlgError:
                print 'Cholesky Decomposition Failed second time!! Using SVD instead'
                u,s,v = sl.svd(Sigma[ii])
                expval2 = np.dot(v.T, 1/s*np.dot(u.T, d[ii]))
                dSigd += np.dot(d[ii], expval2)
                logdet_Sigma += np.sum(np.log(s))

        logLike = -0.5 * (logdet_Phi + logdet_Sigma) + 0.5 * dSigd + loglike1 


    if args.limit_or_detect == 'limit':
        prior_factor = np.log(Agwb * np.log(10.0))
    else:
        prior_factor = 0.0
    return logLike + prior_factor + physicality


#########################
#########################

parameters = ["Agwb"]
if args.fix_slope is False:
    parameters.append("gam_gwb")
    gamma_ext = 'GamVary'
else:
    gamma_ext = 'Gam4p33'

for ii in range(len(psr)):
    parameters.append('Ared_'+psr[ii].name)
for ii in range(len(psr)):
    parameters.append('gam_red_'+psr[ii].name)
for ii in range(len(psr)):
    parameters.append('Adm_'+psr[ii].name)
for ii in range(len(psr)):
    parameters.append('gam_dm_'+psr[ii].name)
for ii in range(len(psr)):
    parameters.append('EFAC_'+psr[ii].name)
parameters.append('Acm')
parameters.append('gam_cm')
parameters.append('Aun')
parameters.append('gam_un')
for ii in range( args.num_gwfreq_wins*(((args.LMAX+1)**2)-1) ):
    parameters.append('clm_{0}'.format(ii+1))


print "\n You are searching for the following parameters: {0}\n".format(parameters)
n_params = len(parameters)

print "\n The total number of parameters is {0}\n".format(n_params)


x0 = np.array([-18.0])
if args.fix_slope is False:
    x0 = np.append(x0,[13./3.])
x0 = np.append(x0,-13.0*np.ones(len(psr)))            #np.log10(np.array(Ared_ML)))
x0 = np.append(x0,1.0*np.ones(len(psr)))              #np.array(gam_red_ML))
x0 = np.append(x0,-13.0*np.ones(len(psr)))            #np.log10(np.array(Adm_ML)))
x0 = np.append(x0,1.0*np.ones(len(psr)))              #np.array(gam_dm_ML))
x0 = np.append(x0,np.ones(len(psr)))                  #np.random.uniform(0.75,1.25,len(psr)))
x0 = np.append(x0,np.array([-18.0,13./3]))
x0 = np.append(x0,np.array([-18.0,13./3.]))
x0 = np.append(x0,np.zeros( args.num_gwfreq_wins*(((args.LMAX+1)**2)-1) ))

print "\n Your initial parameters are {0}\n".format(x0)

cov_diag = np.array([0.5])
if args.fix_slope is False:
    cov_diag = np.append(cov_diag,[0.5])
cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))  #np.array(Ared_err)**2.0)
cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))  #np.array(gam_red_err)**2.0)
cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))  #np.array(Adm_err)**2.0)
cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))  #np.array(gam_dm_err)**2.0)
cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))
cov_diag = np.append(cov_diag,np.array([0.5,0.5]))
cov_diag = np.append(cov_diag,np.array([0.5,0.5]))
cov_diag = np.append(cov_diag,0.05*np.ones( args.num_gwfreq_wins*(((args.LMAX+1)**2)-1) ))


print "\n Running a quick profile on the likelihood to estimate evaluation speed...\n"
cProfile.run('modelIndependentFullPTANoisePL(x0)')

#####################
# Now, we sample.....
#####################

print "\n Now, we sample... \n"
sampler = PAL.PTSampler(ndim=n_params,logl=modelIndependentFullPTANoisePL,logp=my_prior,cov=np.diag(cov_diag),\
                        outDir='./chains_Analysis/EPTAv2_{0}_{1}mode_SearchNoise_nmodes{2}_Lmax{3}_{4}_autoBkgrd'.format(snr_tag_ext,args.limit_or_detect,args.nmodes,args.LMAX,gamma_ext),resume=False)
sampler.sample(p0=x0,Niter=1e6,thin=10)
