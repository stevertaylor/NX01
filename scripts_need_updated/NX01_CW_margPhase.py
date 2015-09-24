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
import pymultinest
import libstempo as T2
import time
from time import gmtime, strftime
import NX01_AnisCoefficients as anis
import NX01_utils as utils
import NX01_psr

parser = optparse.OptionParser(description = 'NX01 - Precursor to the PANTHER Group ENTERPRISE project')

############################
############################

parser.add_option('--snr-tag', dest='snr_tag', action='store', type=float, default=0.9, 
                   help='Do you want the 90%, 95% or 100% SNR dataset? [6, 11, and 41 pulsars respectively] (default=0.90)')
parser.add_option('--limit-or-detect', dest='limit_or_detect', action='store', type=str, default='limit',
                   help='Do you want to use a uniform prior on strain-amplitudes [detect] or strain itself [upper-limit] (default=\'limit\')?')


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

skyLocs = np.array([np.sin(positions[:,1])*np.cos(positions[:,0]), np.sin(positions[:,1])*np.sin(positions[:,0]), np.cos(positions[:,1])]).T
print skyLocs
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

GGCGG=[]
for ii in range(len(psr)):
    tgrid = utils.makeTimeGrid(psr[ii], psr[ii])

    Cred = utils.makeRedTDcov(Ared_ML[ii], gam_red_ML[ii], tgrid)
    Cdm = utils.makeDmTDcov(psr[ii], Adm_ML[ii], gam_dm_ML[ii], tgrid)
    Cwhite = np.diag(psr[ii].toaerrs**2.0)
    ########
    GCGnoise = np.dot(psr[ii].G.T, np.dot(Cred+Cdm+Cwhite, psr[ii].G))
    GCGnoise = np.nan_to_num(GCGnoise)
    cho = sl.cho_factor(GCGnoise)
    GGCGG.append( np.dot(psr[ii].G, np.dot(sl.cho_solve(cho, np.eye(len(GCGnoise))), psr[ii].G.T)) )


################################################################################################################################
# DEFINING THE PRIOR AND THE LOG-LIKELIHOOD
################################################################################################################################

def my_prior(cube, ndim, nparams):
    cube[0] = -18.0 + cube[0]*12.0
    cube[1] = -9.0 + cube[1]*3.0  # searching from a nanohertz up to sampling cadence of approximately 3 days
    cube[2] = cube[2]*2.0*np.pi
    cube[3] = -1.0 + cube[3]*2.0
    cube[4] = -1.0 + cube[4]*2.0
    cube[5] = cube[5]*np.pi
    cube[6] = cube[6]*2.0*np.pi


def ln_prob(cube, ndim, nparams):

    x = np.array([cube[ii] for ii in range(nparams)])  
    
    orb_freq=0.5*(10.0**x[1])
    zeta = 10.0**x[0]     #((constants.G*10.0**x[0]*1.9891*10.0**30.0)**(5.0/3.0))/((10.0**6.0)*constants.parsec*(constants.c**4.0)*10.0**x[7])

    a1 =  zeta*((1.0+pow(x[4],2.0))*np.cos(2.0*x[6])*np.cos(2.0*x[5]) + 2.0*x[4]*np.sin(2.0*x[6])*np.sin(2.0*x[5]))
    a2 = -zeta*((1.0+pow(x[4],2.0))*np.sin(2.0*x[6])*np.cos(2.0*x[5]) - 2.0*x[4]*np.cos(2.0*x[6])*np.sin(2.0*x[5]))
    a3 = zeta*((1.0+pow(x[4],2.0))*np.cos(2.0*x[6])*np.sin(2.0*x[5]) - 2.0*x[4]*np.sin(2.0*x[6])*np.cos(2.0*x[5]))
    a4 = -zeta*((1.0+pow(x[4],2.0))*np.sin(2.0*x[6])*np.sin(2.0*x[5]) + 2.0*x[4]*np.cos(2.0*x[6])*np.cos(2.0*x[5]))
        
    omega = np.array([-np.sin(np.arccos(x[3]))*np.cos(x[2]),-np.sin(np.arccos(x[3]))*np.sin(x[2]),-np.cos(np.arccos(x[3]))])
    mvec = np.array([-np.sin(x[2]),np.cos(x[2]),0.0])
    nvec = np.array([-np.cos(np.arccos(x[3]))*np.cos(x[2]),-np.cos(np.arccos(x[3]))*np.sin(x[2]),np.sin(np.arccos(x[3]))])
        
    fplus = np.array([0.5*(np.inner(mvec,skyLocs[ii])**2.0 - np.inner(nvec,skyLocs[ii])**2.0)/(1.0+np.inner(omega,skyLocs[ii])) for ii in range(len(psr))])
    fcross = np.array([np.inner(mvec,skyLocs[ii])*np.inner(nvec,skyLocs[ii])/(1.0+np.inner(omega,skyLocs[ii])) for ii in range(len(psr))])

    p = np.array([(fplus[ii]*a1+fcross[ii]*a3) for ii in range(len(psr))])
    q = np.array([(fplus[ii]*a2+fcross[ii]*a4) for ii in range(len(psr))])
        
    b1b1=[]
    b2b2=[]
    b1b2=[]
    rb1=[]
    rb2=[]
    for ii in range(len(psr)):
        b1b1.append( np.dot( np.sin(2.0*np.pi*2.0*orb_freq*psr[ii].toas*86400.0), np.dot(GGCGG[ii], np.sin(2.0*np.pi*2.0*orb_freq*psr[ii].toas*86400.0) ) ) / ((2.0*np.pi*orb_freq)**(2.0/3.0)) )
        b2b2.append( np.dot( np.cos(2.0*np.pi*2.0*orb_freq*psr[ii].toas*86400.0), np.dot(GGCGG[ii], np.cos(2.0*np.pi*2.0*orb_freq*psr[ii].toas*86400.0) ) )/ ((2.0*np.pi*orb_freq)**(2.0/3.0)) )
        b1b2.append( np.dot( np.sin(2.0*np.pi*2.0*orb_freq*psr[ii].toas*86400.0), np.dot(GGCGG[ii], np.cos(2.0*np.pi*2.0*orb_freq*psr[ii].toas*86400.0) ) )/ ((2.0*np.pi*orb_freq)**(2.0/3.0)) )
        rb1.append( np.dot( psr[ii].res, np.dot(GGCGG[ii], np.sin(2.0*np.pi*2.0*orb_freq*psr[ii].toas*86400.0) ) ) / ((2.0*np.pi*orb_freq)**(1.0/3.0)) )
        rb2.append( np.dot( psr[ii].res, np.dot(GGCGG[ii], np.cos(2.0*np.pi*2.0*orb_freq*psr[ii].toas*86400.0) ) ) / ((2.0*np.pi*orb_freq)**(1.0/3.0)) )
        
        
    log_like=0.0
    for ii in range(len(psr)):
        rs = lambda phi: (p[ii]*rb1[ii] + q[ii]*rb2[ii])*(1.0-np.cos(phi)) - (q[ii]*rb1[ii]-p[ii]*rb2[ii])*np.sin(phi)
        ss = lambda phi: ((p[ii]**2)*(1-np.cos(phi))**2 + (q[ii]**2)*np.sin(phi)**2 - 2*p[ii]*q[ii]*(1-np.cos(phi))*np.sin(phi))*b1b1[ii] + ((q[ii]**2)*(1-np.cos(phi))**2 + (p[ii]**2)*np.sin(phi)**2 + 2*p[ii]*q[ii]*(1-np.cos(phi))*np.sin(phi))*b2b2[ii] + 2.0*(p[ii]*q[ii]*(1-np.cos(phi)) + (p[ii]**2)*(1-np.cos(phi))*np.sin(phi) - (q[ii]**2)*(1-np.cos(phi))*np.sin(phi) - p[ii]*q[ii]*np.sin(phi)**2)
        
        sub = -optimize.brute(lambda phi: -(rs(phi)-0.5*ss(phi)),[[0.0,2.0*np.pi]],Ns=100,full_output=True)[1]             

        log_like += -np.log(2.0*np.pi) + sub + np.log(integrate.quad(lambda phi: np.exp(rs(phi)-0.5*ss(phi)-sub),0.0,2.0*np.pi,limit=75)[0])

    return log_like + np.log(zeta*np.log(10.))  #renormalising to produce uniform prior on zeta


#########################
#########################

parameters = ["zeta", "f", "phi", "costheta", "cosiota", "psi", "phase"]
n_params = len(parameters)

print "\n You are searching for the following parameters: {0}\n".format(parameters)
n_params = len(parameters)

print "\n The total number of parameters is {0}\n".format(n_params)

#####################
# Now, we sample.....
#####################

print "\n Now, we sample... \n"

pymultinest.run(ln_prob, my_prior, n_params, 
	importance_nested_sampling = False, 
	resume = False, verbose = True, n_live_points=100, outputfiles_basename=u'chains_Analysis/EPTAv2_{0}_{1}mode_FixedNoise_margPhaseCW/output_'.\
    format(snr_tag_ext,args.limit_or_detect),sampling_efficiency=0.3,const_efficiency_mode=False)


