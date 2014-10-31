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
import ephem
from ephem import *
import PALInferencePTMCMC as PAL
import libstempo as T2
import time
from time import gmtime, strftime
import AnisCoefficients as anis
import NX01_utils as utils

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

path = '/Users/staylor/Research/EPTAv2/UniEQ'   #os.getcwd()

if not os.path.exists('chains_Analysis'):
    os.makedirs('chains_Analysis')

pulsars = [s for s in dir if "J" in s]
pulsars.sort()

print pulsars
################################################################################################################################
# PASSING THROUGH TEMPO2 VIA libstempo
################################################################################################################################

psr=[]
for ii in range(len(pulsars)):
    os.chdir(path+'/'+pulsars[ii])
    if os.path.isfile('{0}_NoAFB.par'.format(pulsars[ii])):
        psr.append(T2.tempopulsar(parfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_TD.Mean.par',timfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_NoAFB.tim'))
    else:
        psr.append(T2.tempopulsar(parfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_TD.Mean.par',timfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_all.tim'))
    os.chdir(path)
    psr[ii].fit(iters=10)
    if np.any(np.isfinite(psr[ii].residuals())==False)==True:
        os.chdir(path+'/'+pulsars[ii])
	if os.path.isfile('{0}_NoAFB.par'.format(pulsars[ii])):
	    psr[ii] = T2.tempopulsar(parfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_TD.Mean.par',timfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_NoAFB.tim')
	else:
	    psr.append(T2.tempopulsar(parfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_TD.Mean.par',timfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_all.tim'))
        os.chdir(path)

################################################################################################################################
# GETTING THE DESIGN MATRICES AND COMPUTING ALL THE 'G' MATRICES
################################################################################################################################

d=[]
G=[]
for ii in range(len(psr)):
    d.append(psr[ii].designmatrix())
    U,S,V = np.linalg.svd(d[ii])
    G.append(U[:,len(S):len(U)])

################################################################################################################################
# GRABBING ALL THE TIME-STAMPS, RESIDUALS, AND ERROR-BARS
################################################################################################################################

obs=[]
res=[]
err=[]
psr_locs=[]
for ii in range(len(psr)):
    obs.append(np.array(psr[ii].toas()))
    res.append(np.array(psr[ii].residuals()))
    err.append(np.array(1e-6*psr[ii].toaerrs))
    if 'RAJ' and 'DECJ' in psr[ii].pars:
        psr_locs.append([psr[ii]['RAJ'].val,psr[ii]['DECJ'].val])
    elif 'ELONG' and 'ELAT' in psr[ii].pars:
        fac = 180./np.pi
        coords = Equatorial(Ecliptic(str(psr[ii]['ELONG'].val*fac), str(psr[ii]['ELAT'].val*fac)))
        psr_locs.append([float(repr(coords.ra)),float(repr(coords.dec))])

psr_locs = [np.array([psr_locs[ii][0], np.pi/2. - psr_locs[ii][1]]) for ii in range(len(psr_locs))]
positions = np.array(psr_locs).copy()

CorrCoeff = np.array(anis.CorrBasis(positions,args.LMAX))

harm_sky_vals = utils.SetupPriorSkyGrid(args.LMAX)

gwfreqs_per_win = int(1.*args.nmodes/(1.*args.num_gwfreq_wins))
        
################################################################################################################################
# GETTING MAXIMUM TIME, COMPUTING FOURIER DESIGN MATRICES, AND GETTING MODES 
################################################################################################################################
Tmax = np.max([obs[p].max() - obs[p].min() for p in range(len(psr))])

# initialize fourier design matrices
Fred = [utils.createfourierdesignmatrix_RED(obs[p], args.nmodes, Tspan=Tmax) for p in range(len(psr))]
Fdm = [utils.createfourierdesignmatrix_DM(obs[p], args.nmodes, psr[p].freqs, Tspan=Tmax) for p in range(len(psr))]

F = [np.append(Fred[p], Fdm[p], axis=1) for p in range(len(psr))]

# get frequency
tmp, fqs = utils.createfourierdesignmatrix_RED(obs[0], args.nmodes, Tspan=Tmax, freq=True)

#print Tmax, len(fqs), len(F), F[0].shape

################################################################################################################################
# FORM A LIST COMPOSED OF NP ARRAYS CONTAINING THE INDEX POSITIONS WHERE EACH UNIQUE 'sys' BACKEND IS APPLIED
################################################################################################################################

backends = [0.0]*len(pulsars)
for ii in range(len(pulsars)):
    if 'sys' in psr[ii].flags:
        backends[ii] = [0.0]*len(np.unique(psr[ii].flags['sys']))
        for k in range(len(np.unique(psr[ii].flags['sys']))):
            backends[ii][k] = np.where(psr[ii].flags['sys'] == np.unique(psr[ii].flags['sys'])[k])[0]
    else:
        print "No 'sys' flags found :("
        print "Using one overall EFAC for {0}\n".format(psr[ii].name)

################################################################################################################################
# GETTING MAXIMUM-LIKELIHOOD VALUES OF SINGLE-PULSAR ANALYSIS
################################################################################################################################

Adm_ML=[]
gam_dm_ML=[]
Ared_ML=[]
gam_red_ML=[]
EFAC_ML = [[0.0]*len(backends[jj]) for jj in range(len(backends))]
EQUAD_ML = [[0.0]*len(backends[jj]) for jj in range(len(backends))]
for ii in range(len(pulsars)):
    with open('{0}/{0}_Taylor_TimeDomain_model1.txt'.format(pulsars[ii]), 'r') as f:
        Adm_ML.append(float(f.readline().split()[3]))
        gam_dm_ML.append(float(f.readline().split()[3]))
        Ared_ML.append(float(f.readline().split()[3]))
        gam_red_ML.append(float(f.readline().split()[3]))
        for jj in range(len(backends[ii])):
            EFAC_ML[ii][jj] = float(f.readline().split()[3])
        for jj in range(len(backends[ii])):
            EQUAD_ML[ii][jj] = float(f.readline().split()[3])


################################################################################################################################
# GETTING MEAN AND ERROR-BARS VALUES OF SINGLE-PULSAR ANALYSIS
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
    with open('{0}/{0}_Taylor_TimeDomain_model1.txt'.format(pulsars[ii]), 'r') as f:
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
MLerrors=[]
Diag=[]
res_prime=[]
F_prime=[]
for ii in range(len(pulsars)):   
    MLerrors.append( err[ii] )
    ########
    efac_bit = np.dot(G[ii].T, np.dot( np.diag(MLerrors[ii]**2.0), G[ii] ) )
    equad_bit = np.dot(G[ii].T,G[ii])
    Lequad = np.linalg.cholesky(equad_bit)
    Lequad_inv = np.linalg.inv(Lequad)
    sand = np.dot(Lequad_inv, np.dot(efac_bit, Lequad_inv.T))
    u,s,v = np.linalg.svd(sand)
    Diag.append(s)
    proj = np.dot(u.T, np.dot(Lequad_inv, G[ii].T))
    ########
    res_prime.append( np.dot(proj, res[ii]) )
    F_prime.append( np.dot(proj, F[ii]) )


################################################################################################################################
# SETTING UP PRIOR RANGES
################################################################################################################################

pmin = np.array([-20.0,0.0])
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

pmax = np.array([-10.0,7.0])
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
    gam_gwb = x[1]
    #####
    Ared = 10.0**x[2:2+len(psr)]
    gam_red = x[2+len(psr):2+2*len(psr)]
    Adm = 10.0**x[2+2*len(psr):2+3*len(psr)]
    gam_dm = x[2+3*len(psr):2+4*len(psr)]
    EFAC = x[2+4*len(psr):2+5*len(psr)]
    Acm = 10.0**x[2+5*len(psr)]
    gam_cm = x[2+5*len(psr) + 1]
    Aun = 10.0**x[2+5*len(psr) + 2]
    gam_un = x[2+5*len(psr) + 3]
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

    # get the number of modes, should be the same for all pulsars
    #nmode = args.num_gwfreq_wins*gwfreqs_per_win
    npsr = len(obs)

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
    for p in range(len(psr)):

        # compute d
        if p == 0:
            d = np.dot(F_prime[p].T, res_prime[p]/( (EFAC[p]**2.0)*Diag[p] ))
        else:
            d = np.append(d, np.dot(F_prime[p].T, res_prime[p]/( (EFAC[p]**2.0)*Diag[p] )))

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
        kappa.append(np.log10( np.append( Ared[ii]**2/12/np.pi**2 * f1yr**(gam_red[ii]-3) * (fqs/86400.0)**(-gam_red[ii])/Tspan, Adm[ii]**2/12/np.pi**2 * f1yr**(gam_dm[ii]-3) * (fqs/86400.0)**(-gam_dm[ii])/Tspan ) ))

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
        tot[0::2] = ORF[:,ii,ii]*np.append( 10**rho, np.zeros(len(rho)) ) + np.append( 10**cm + 10**un, np.zeros(len(rho)) ) + 10**kappa[ii]
        tot[1::2] = ORF[:,ii,ii]*np.append( 10**rho, np.zeros(len(rho)) ) + np.append( 10**cm + 10**un, np.zeros(len(rho)) ) + 10**kappa[ii]

        # common-mode terms
        commonmode[0::2] = np.append( 10**cm, np.zeros(len(rho)) )
        commonmode[1::2] = np.append( 10**cm, np.zeros(len(rho)) )
                
        # fill in lists of arrays
        sigdiag.append(tot)
        sigoffdiag.append(offdiag)
        sigcm.append(commonmode)


    # compute Phi inverse from Lindley's code
    smallMatrix = np.zeros((4*args.nmodes, npsr, npsr))
    for ii in range(npsr):
        for jj in range(ii,npsr):

            if ii == jj:
                smallMatrix[:,ii,jj] = sigdiag[jj] 
            else:
                smallMatrix[:,ii,jj] = ORFtot[:,ii,jj] * sigoffdiag[jj] + sigcm[jj]
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
        # now fill in real covariance matrix
        Phi = np.zeros((4*npsr*args.nmodes, 4*npsr*args.nmodes))
        for ii in range(npsr):
            for jj in range(ii,npsr):
                for kk in range(0,4*args.nmodes):
                    Phi[kk+ii*4*args.nmodes,kk+jj*4*args.nmodes] = smallMatrix[kk,ii,jj]
    
        # symmeterize Phi
        Phi = Phi + Phi.T - np.diag(np.diag(Phi))
            
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

        return logLike + np.log(Agwb) + physicality



#########################
#########################


parameters = ["Agwb","gam_gwb"]
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


x0 = np.array([-15.0,13./3.])
x0 = np.append(x0,np.log10(np.array(Ared_ML)))
x0 = np.append(x0,np.array(gam_red_ML))
x0 = np.append(x0,np.log10(np.array(Adm_ML)))
x0 = np.append(x0,np.array(gam_dm_ML))
x0 = np.append(x0,np.random.uniform(0.75,1.25,len(psr)))
x0 = np.append(x0,np.array([-15.0,2.0]))
x0 = np.append(x0,np.array([-15.0,2.0]))
x0 = np.append(x0,np.zeros( args.num_gwfreq_wins*(((args.LMAX+1)**2)-1) ))

print "\n Your initial parameters are {0}\n".format(x0)

cov_diag = np.array([0.5,0.5])
cov_diag = np.append(cov_diag,np.array(Ared_err)**2.0)
cov_diag = np.append(cov_diag,np.array(gam_red_err)**2.0)
cov_diag = np.append(cov_diag,np.array(Adm_err)**2.0)
cov_diag = np.append(cov_diag,np.array(gam_dm_err)**2.0)
cov_diag = np.append(cov_diag,0.2*np.ones(len(psr)))
cov_diag = np.append(cov_diag,np.array([0.5,0.5]))
cov_diag = np.append(cov_diag,np.array([0.5,0.5]))
cov_diag = np.append(cov_diag,0.05*np.ones( args.num_gwfreq_wins*(((args.LMAX+1)**2)-1) ))


#####################
# Now, we sample.....
#####################

print "\n Now, we sample... \n"
sampler = PAL.PTSampler(ndim=n_params,logl=modelIndependentFullPTANoisePL,logp=my_prior,cov=np.diag(cov_diag),outDir='./chains_Analysis/EPTAv2_90pct_Test',resume=False)
sampler.sample(p0=x0,Niter=500000,thin=1)
