#!/usr/bin/env python

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

dir = ['J1909-3744', 'J1713+0747', 'J1744-1134', 'J0613-0200', 'J1600-3053', 'J1012+5307']   #gives 90%
      # addition of 1640, 2145, 1857, 1022, 0030 give 95% of total SNR^2

master_path = os.getcwd()
path = '/Users/staylor/Research/EPTAv2/UniEQ'  

if not os.path.exists('chains_Analysis'):
    os.makedirs('chains_Analysis')

pulsars = [s for s in dir if "J" in s]
pulsars.sort()

print pulsars
################################################################################################################################
# PASSING THROUGH TEMPO2 VIA libstempo
################################################################################################################################

par_ext = 'ML'  # Running fixed-noise search with ML parameters. Need ML EFACs and EQUADs to represent covariance of params

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
	    t2psr.append(T2.tempopulsar(parfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_TD.{0}.par'.format(par_ext),\
                                    timfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_all.tim'))
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

pmin = np.array([-20.0,0.0])
pmin = np.append(pmin,-10.0*np.ones( args.num_gwfreq_wins*(((args.LMAX+1)**2)-1) ))

pmax = np.array([-10.0,7.0])
pmax = np.append(pmax,10.0*np.ones( args.num_gwfreq_wins*(((args.LMAX+1)**2)-1) ))
##################################################################################################################################

loglike1 = 0
FtNF = []
for p in range(len(psr)):
        
    # compute d
    if p == 0:
        d = np.dot(F_prime[p].T, res_prime[p]/Diag[p] )
    else:
        d = np.append(d, np.dot(F_prime[p].T, res_prime[p]/Diag[p] ))

    # compute FT N F
    N = 1./Diag[p]
    right = (N*F_prime[p].T).T
    FtNF.append(np.dot(F_prime[p].T, right))

    # log determinant of N
    logdet_N = np.sum(np.log( Diag[p] ))
        
    # triple product in likelihood function
    dtNdt = np.sum(res_prime[p]**2.0/( Diag[p] ))

    loglike1 += -0.5 * (logdet_N + dtNdt)


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
    gam_gwb = x[1]
    #####
    ###################
    orf_coeffs = x[2+5*len(psr) + 4:]
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

    # parameterize intrinsic red noise as power law
    Tspan = (1/fqs[0])*86400.0
    f1yr = 1/3.16e7
    rho = np.log10(Agwb**2/12/np.pi**2 * f1yr**(gam_gwb-3) * (fqs/86400.0)**(-gam_gwb)/Tspan)

    # parameterize intrinsic red-noise and DM-variations as power law
    kappa = [] 
    for ii in range(npsr):
        kappa.append(np.log10( np.append( Ared_ML[ii]**2/12/np.pi**2 * f1yr**(gam_red_ML[ii]-3) * (fqs/86400.0)**(-gam_red_ML[ii])/Tspan,\
                                          Adm_ML[ii]**2/12/np.pi**2 * f1yr**(gam_dm_ML[ii]-3) * (fqs/86400.0)**(-gam_dm_ML[ii])/Tspan ) ))

    # construct elements of sigma array
    sigdiag = []
    sigoffdiag = []
    sigcm = []
    for ii in range(npsr):
        tot = np.zeros(4*args.nmodes)
        offdiag = np.zeros(4*args.nmodes)
        commonmode = np.zeros(4*args.nmodes)

        # off diagonal terms
        offdiag[0::2] = np.append( 10**rho, np.zeros(len(rho)) )
        offdiag[1::2] = np.append( 10**rho, np.zeros(len(rho)) )

        # diagonal terms
        tot[0::2] = ORF[:,ii,ii]*np.append( 10**rho, np.zeros(len(rho)) ) + 10**kappa[ii]
        tot[1::2] = ORF[:,ii,ii]*np.append( 10**rho, np.zeros(len(rho)) ) + 10**kappa[ii]
                
        # fill in lists of arrays
        sigdiag.append(tot)
        sigoffdiag.append(offdiag)

    # compute Phi inverse from Lindley's code
    smallMatrix = np.zeros((4*args.nmodes, npsr, npsr))
    for ii in range(npsr):
        for jj in range(ii,npsr):

            if ii == jj:
                smallMatrix[:,ii,jj] = sigdiag[jj] 
            else:
                smallMatrix[:,ii,jj] = ORFtot[:,ii,jj] * sigoffdiag[jj] 
                smallMatrix[:,jj,ii] = smallMatrix[:,ii,jj]

    
    # invert them
    logdet_Phi = 0
    non_pos_def = 0
    for ii in range(4*args.nmodes):
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
        nftot = 4*args.nmodes
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

        return logLike + np.log(Agwb * np.log(10.0)) + physicality



#########################
#########################


parameters = ["Agwb","gam_gwb"]
for ii in range( args.num_gwfreq_wins*(((args.LMAX+1)**2)-1) ):
    parameters.append('clm_{0}'.format(ii+1))


print "\n You are searching for the following parameters: {0}\n".format(parameters)
n_params = len(parameters)

print "\n The total number of parameters is {0}\n".format(n_params)

x0 = np.array([-15.0,13./3.])
x0 = np.append(x0,np.zeros( args.num_gwfreq_wins*(((args.LMAX+1)**2)-1) ))

print "\n Your initial parameters are {0}\n".format(x0)

cov_diag = np.array([0.5,0.5])
cov_diag = np.append(cov_diag,0.05*np.ones( args.num_gwfreq_wins*(((args.LMAX+1)**2)-1) ))


print "\n Running a quick profile on the likelihood to estimate evaluation speed...\n"
cProfile.run('modelIndependentFullPTANoisePL(x0)')

#####################
# Now, we sample.....
#####################

print "\n Now, we sample... \n"
sampler = PAL.PTSampler(ndim=n_params,logl=modelIndependentFullPTANoisePL,logp=my_prior,cov=np.diag(cov_diag),outDir='./chains_Analysis/EPTAv2_90pct_FixedNoiseIsoGam',resume=False)
sampler.sample(p0=x0,Niter=500000,thin=10)
