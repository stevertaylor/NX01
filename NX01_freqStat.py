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
import NX01_bayesutils as bu

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


gam_bkgrd = 4.33333
optimalStat = utils.optStat(psr, GCGnoiseInv, HnD, gam_gwb=gam_bkgrd)
print "\n A^2 = {0}, std = {1}, SNR = {2}\n".format(optimalStat[0],optimalStat[1],optimalStat[2])

far = 0.05
dr_list = [0.95,0.68]
bu.OSupperLimit(psr, GCGnoiseInv, HnD, far, dr_list, optimalStat[1])
