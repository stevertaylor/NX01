#!/usr/bin/env python

"""
Created by stevertaylor
Copyright (c) 2014 Stephen R. Taylor

Code contributions by Rutger van Haasteren (piccard) and Justin Ellis (PAL/PAL2).

"""

# Making realistic pulsar-timing simulations - pipeline by Steve Taylor (22 Oct 2014)

import numpy as np
import scipy
import libstempo as T
import libstempo.plot as LP, libstempo.toasim as LT
import os
import math
import numexpr as ne
from ephem import *
from libstempo import GWB

day = 24 * 3600
year = 365.25 * day
DMk = 4.15e3           # Units MHz^2 cm^3 pc sec

parser = optparse.OptionParser(description = 'NX01 - Precursor to the PANTHER Group ENTERPRISE project')

############################
############################

parser.add_option('--nmodes', dest='nmodes', action='store', type=int, default=50,
                   help='Number of modes in low-rank time-frequency approximation for red and DM noise (default=50).')
parser.add_option('--gwamp', dest='gwamp', action='store', type=float,
                   help='Dimensionless amplitude of simulated GWB strain spectrum.')
parser.add_option('--gwalpha', dest='gwalpha', action='store', type=float, default=-0.666,
                   help='Index of simulated GWB strain spectrum (default=-0.666).')


(args, x) = parser.parse_args()

###############################################################
# Getting the names of all pulsars from the directory names
###############################################################

dir = os.walk('.').next()[1]
dir.remove('J1939+2134')

master_path = os.getcwd()
path = '/Users/staylor/Research/EPTAv2'

pulsars = [s for s in dir if "J" in s]
pulsars.sort()

###############################################################
# Reading in all pulsars with libstempo
###############################################################

psr=[]
for ii in range(len(pulsars)):
    os.chdir(path+'/'+pulsars[ii])
    if os.path.isfile('{0}_NoAFB.par'.format(pulsars[ii])):
        psr.append(T.tempopulsar(parfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_NoAFB.par',\
                                 timfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_NoAFB.tim'))
    else:
        psr.append(T.tempopulsar(parfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'.par',\
                                 timfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_all.tim'))
    os.chdir(path)
    psr[ii].fit(iters=10)
    
    if np.any(np.isfinite(psr[ii].residuals())==False)==True:
        
        os.chdir(path+'/'+pulsars[ii])
        
        if os.path.isfile('{0}_NoAFB.par'.format(pulsars[ii])):
            psr[ii] = T.tempopulsar(parfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_NoAFB.par',\
                                    timfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_NoAFB.tim')
        else:
            psr[ii] = T.tempopulsar(parfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'.par',\
                                    timfile=path+'/'+pulsars[ii]+'/'+pulsars[ii]+'_all.tim')
        
        os.chdir(path)

os.chdir(master_path)

##############################################################################################################################
# Subtracting the computed residuals from the TOAs, obtaining (in effect) a perfect realization
# of the deterministic timing model. The pulsar parameters will have changed somewhat,
# so make_ideal calls fit() on the pulsar object.
##############################################################################################################################

for ii in range(len(psr)):
    LT.make_ideal(psr[ii])

# Find the distinct backends for all pulsars

backends = [0.0]*len(pulsars)
for ii in range(len(pulsars)):
    if 'sys' in psr[ii].flags:
        backends[ii] = [0.0]*len(np.unique(psr[ii].flags['sys']))
        for k in range(len(np.unique(psr[ii].flags['sys']))):
            backends[ii][k] = np.where(psr[ii].flags['sys'] == np.unique(psr[ii].flags['sys'])[k])[0]
    else:
        print "No 'sys' flags found :("
        print "Using one overall EFAC for {0}\n".format(psr[ii].name)

##############################################################################################################################
#  Read in all maximum-likelihood parameters from previous single-pulsar analysis
##############################################################################################################################

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

###############################################################
# Putting back EFAC and EQUAD noise
###############################################################

for ii in range(len(psr)):

    scaled_err = (psr[ii].toaerrs).copy()
    for jj in range(len(backends[ii])):
        scaled_err[backends[ii][jj]] *= EFAC_ML[ii][jj]
        
    psr[ii].stoas[:] += scaled_err * (1e-6 / day) * np.random.randn(psr[ii].nobs)


for ii in range(len(psr)):
        
    white_noise = np.ones(len((psr[ii].toaerrs).copy()))
    for jj in range(len(backends[ii])):
        white_noise[backends[ii][jj]] *= EQUAD_ML[ii][jj]
        
    psr[ii].stoas[:] += (white_noise / day) * np.random.randn(psr[ii].nobs)


##############################################################################################################################
# Now adding back red noise and DM-variation noise (libstempo has the ST format DM amplitude)
##############################################################################################################################

for ii in range(len(psr)):
    LT.add_rednoise(psr[ii],Ared_ML[ii],gam_red_ML[ii],components=args.nmodes)
    LT.add_dm(psr[ii],Adm_ML[ii],gam_dm_ML[ii],args.nmodes)   

###############################################################
# Do we want to add in a GWB signal?
###############################################################

gwb = GWB(ngw=10000,flow=1e-11,fhigh=1e-5,gwAmp=args.gwamp,alpha=args.gwalpha)

psrDist = np.ones(len(psr))  # positions all pulsars at 1kpc as default

for ii in range(len(psr)):
    gwb.add_gwb(psr[ii],psrDist[ii])

###############################################################
# Finally, let's save all the resulting .par and .tim files
###############################################################

sim_dir = 'Sim1000305'  # ...only an example.
# id of the set of simulations (digit 1), the noise realization (digit 2 to 5),
# the injected GWB (digit 6 and 7). For example, Sim1000305 refers to the first
# set of simulation (code 1), the noise realisation 3 (code 0003) and the injection
# 5 (code 05), i.e. log10(GWBamp) = -14.8 .

if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
        
for ii in range(len(psr)):
    psr[ii].savepar(sim_dir+'/{0}.par'.format(psr[ii].name))
    psr[ii].savetim(sim_dir+'/{0}.tim'.format(psr[ii].name))

