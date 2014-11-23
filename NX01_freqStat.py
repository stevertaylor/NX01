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

parser.add_option('--lmax', dest='LMAX', action='store', type=int, default=0,
                   help='Maximum multipole for generalised OS (default = 0, i.e. isotropic OS)')
parser.add_option('--snr-tag', dest='snr_tag', action='store', type=float, default=0.9, 
                   help='Do you want the 90%, 95% or 100% SNR dataset? [6, 11, and 41 pulsars respectively] (default=0.90)')

(args, x) = parser.parse_args()

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
HnD = 2.0*np.sqrt(np.pi)*CorrCoeff[0]

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
# MAKE FIXED NOISE MATRICES FROM MAXIMUM-LIKELIHOOD VALUES OF SINGLE-PULSAR ANALYSIS
################################################################################################################################

GCGnoiseInv=[]
for ii in range(len(psr)):
    tgrid = utils.makeTimeGrid(psr[ii], psr[ii])

    Cred = utils.makeRedTDcov(Ared_ML[ii], gam_red_ML[ii], tgrid)
    Cdm = utils.makeDmTDcov(psr[ii], Adm_ML[ii], gam_dm_ML[ii], tgrid)
    Cwhite = np.diag(psr[ii].toaerrs**2.0)
    ########
    GCGnoise = np.dot(psr[ii].G.T, np.dot(Cred+Cdm+Cwhite, psr[ii].G))
    GCGnoise = np.nan_to_num(GCGnoise)
    cho = sl.cho_factor(GCGnoise)
    GCGnoiseInv.append(sl.cho_solve(cho, np.eye(len(GCGnoise))))


print utils.optStat(psr, GCGnoiseInv, HnD, gam_gwb=4.33333)


    

'''
def AnisOptStat(gam_gwb=4.33333):
    """
    Computes the Optimal statistic as defined in Chamberlin, Creighton, Demorest et al (2013)

    @param psr: List of pulsar object instances
    @param ORF: Vector of pairwise overlap reduction values
    @param gam: Power Spectral index of GBW (default = 13/3, ie SMBMBs)

    @return: Opt: Optimal statistic value (A_gw^2)
    @return: sigma: 1-sigma uncertanty on Optimal statistic
    @return: snr: signal-to-noise ratio of cross correlations

    """
 
    top = np.zeros((len(psr),len(psr)))
    bot = np.zeros((len(psr),len(psr)))
    for ll in range(len(psr)):
        for kk in range(ll+1, len(psr)):
            # form matrix of toa residuals and compute SigmaIJ
            ta, tb = np.meshgrid(obs[kk], obs[ll])  
            tgrid = np.abs(ta-tb).astype(np.float64)/365.25
            Tspan = tgrid.max()
            fL = 1/(100.0*Tspan)
            xgrid = 2*np.pi*fL*tgrid
            #
            # create cross covariance matrix without overall amplitude A^2
            SIJ = ((fL**(1.0-gam_gwb))/(12.0*np.pi**2.0))*((ss.gamma(1-gam_gwb)*np.sin(np.pi*gam_gwb/2)*ne.evaluate("xgrid**(gam_gwb-1)")) \
                                                                  -sumTermCovarianceMatrix_fast(tgrid, fL, gam_gwb))
            SIJ *= ((365.25*86400.0)**2.0)
            G_SIJ_G = np.dot(G[ll].T,np.dot(SIJ,G[kk]))
            # construct numerator and denominator of optimal statistic
            bot[ll,kk] = np.trace(np.dot(GCGnoise_inv[ll], np.dot(G_SIJ_G, np.dot(GCGnoise_inv[kk], G_SIJ_G.T))))
            top[ll,kk] = np.dot(Gt[ll], np.dot(GCGnoise_inv[ll], np.dot(G_SIJ_G, \
                                                                                np.dot(GCGnoise_inv[kk], Gt[kk]))))
            #
            #
    return top, bot
'''



'''
amp, weight = AnisOptStat(gam_gwb=13./3.)

lmax=2
coeff = np.array(anis.AnisCoeff(positions,lmax))
coeff[0] = coeff[0]/(np.sqrt(4.0*np.pi))

X = np.array([np.multiply(coeff[ii],amp) for ii in range(len(coeff))])
X = np.array([np.sum(X[ii]) for ii in range(len(X))])

gamma = np.zeros((len(coeff),len(coeff)))
for alpha in range(len(coeff)):
            for beta in range(len(coeff)):
                for ll in range(len(psr)):
                    for kk in range(ll+1, len(psr)):
                        #
                        #p,q = np.meshgrid(np.array(coeff)[alpha,ll,kk], np.array(coeff)[beta,ll,kk])
                        #orf_grid = p*q
                        orf_grid = np.array(coeff)[alpha,ll,kk]*np.array(coeff)[beta,ll,kk]
                        gamma[alpha,beta] += orf_grid*weight[ll,kk]

P = np.dot(sl.inv(gamma), X)
'''                          
    

   


