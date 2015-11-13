#!/usr/bin/env python

"""
Created by stevertaylor
Copyright (c) 2014 Stephen R. Taylor

Code contributions by Rutger van Haasteren (piccard) and Justin Ellis (PAL/PAL2).

"""

from __future__ import division
import os, math, optparse, time, cProfile
import json
from time import gmtime, strftime
from collections import OrderedDict
import h5py as h5

import numpy as np
from numpy import *

from scipy import integrate
from scipy import optimize
from scipy import constants as sc
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
f1yr = 1.0/(365.25*86400.0)

parser = optparse.OptionParser(description = "NX01 - It's been a long road, getting from there to here...")

############################
############################

parser.add_option('--from-h5', dest='from_h5', action='store_true', default = False,
                   help='Do you want to read in pulsars from hdf5 files instead of directly via libstempo? (default = False)')
parser.add_option('--psrlist', dest='psrlist', action='store', type=str, default = None,
                   help='Provide path to file containing list of pulsars and their respective par/tim paths')
parser.add_option('--nmodes', dest='nmodes', action='store', type=int,
                   help='Number of modes in low-rank time-frequency approximation')
parser.add_option('--cadence', dest='cadence', action='store', type=float,
                   help='Instead of nmodes, provide the observational cadence.')
parser.add_option('--dmVar', dest='dmVar', action='store_true', default=False,
                   help='Search for DM variations in the data (False)? (default=False)')
parser.add_option('--mnest', dest='mnest', action='store_true', default=False,
                   help='Sample using MultiNest? (default=False)')
parser.add_option('--incGWB', dest='incGWB', action='store_true', default=False,
                  help='Do you want to search for a GWB? (default = False)')
parser.add_option('--gwbSpecModel', dest='gwbSpecModel', action='store', type=str, default='powerlaw',
                  help='What kind of spectral model do you want for the GWB: powerlaw, spectrum, broken (default = powerlaw)')
parser.add_option('--incCorr', dest='incCorr', action='store_true', default=False,
                  help='Do you want to include cross-correlations in the GWB model? (default = False)')
parser.add_option('--redSpecModel', dest='redSpecModel', action='store', type=str, default='powerlaw',
                  help='What kind of spectral model do you want for red timing-noise: powerlaw, spectrum, broken (default = powerlaw)')
parser.add_option('--dmSpecModel', dest='dmSpecModel', action='store', type=str, default='powerlaw',
                  help='What kind of spectral model do you want for DM variations: powerlaw, spectrum, broken (default = powerlaw)')
parser.add_option('--num_gwfreq_wins', dest='num_gwfreq_wins', action='store', type=int, default=1,
                   help='Number windows to split the band into (useful for evolving anisotropy searches (default = 1 windows)')
parser.add_option('--lmax', dest='LMAX', action='store', type=int, default=0,
                   help='Maximum multipole in anisotropic search (default = 0, i.e. isotropic-search)')
parser.add_option('--miCorr', dest='miCorr', action='store_true', default=False,
                   help='Do you want to search for the cross-correlation values directly? (default = False)')
parser.add_option('--use-gpu', dest='use_gpu', action='store_true', default=False,
                  help='Do you want to use the GPU for accelerated linear algebra? (default = False)')
parser.add_option('--fix-slope', dest='fix_slope', action='store_true', default=False,
                  help='Do you want to fix the slope of the GWB spectrum? (default = False)')
parser.add_option('--limit-or-detect-gwb', dest='limit_or_detect_gwb', action='store', type=str, default='limit',
                   help='Do you want to use a uniform prior on log_10(Agwb) [detect] or Agwb itself [limit] (default=\'limit\')?')
parser.add_option('--limit-or-detect-red', dest='limit_or_detect_red', action='store', type=str, default='limit',
                   help='Do you want to use a uniform prior on log_10(Ared) [detect] or Ared itself [limit] (default=\'limit\')?')
parser.add_option('--limit-or-detect-dm', dest='limit_or_detect_dm', action='store', type=str, default='limit',
                   help='Do you want to use a uniform prior on log_10(Adm) [detect] or Adm itself [limit] (default=\'limit\')?')
parser.add_option('--anis-modefile', dest='anis_modefile', action='store', type=str, default = None,
                   help='Do you want to provide an anisotropy modefile to split band into frequency windows?')
parser.add_option('--fullN', dest='fullN', action='store_true', default=True,
                  help='Do you want to perform a full noise search? (default = True)')
parser.add_option('--num_psrs', dest='num_psrs', action='store', type=int, default=18,
                  help='How many pulsars do you want to analyse? (default = 18)')
parser.add_option('--det-signal', dest='det_signal', action='store_true', default=False,
                  help='Do you want to search for a deterministic GW signal? (default = False)')
parser.add_option('--bwm-search', dest='bwm_search', action='store_true', default=False,
                  help='Do you want to search for GW burst with memory (BWM)? (default = False)')
parser.add_option('--bwm-model-select', dest='bwm_model_select', action='store_true', default=False,
                  help='Do you want to compute the Bayes Factor for BWM+noise verus noise-only? (default = False)')
parser.add_option('--cgw-search', dest='cgw_search', action='store_true', default=False,
                  help='Do you want to search for a single continuous GW signal? (default = False)')
parser.add_option('--ecc-search', dest='ecc_search', action='store_true', default=False,
                  help='Do you want to search for an eccentric binary? (default = False)')
parser.add_option('--psrTerm', dest='psrTerm', action='store_true', default=False,
                  help='Do you want to include the pulsar term in the continuous wave search? (default = False)')
parser.add_option('--periEv', dest='periEv', action='store_true', default=False,
                  help='Do you want to model the binary periapsis evolution? (default = False)')

(args, x) = parser.parse_args()

header = """\

        
 /$$   /$$ /$$   /$$  /$$$$$$    /$$  
| $$$ | $$| $$  / $$ /$$$_  $$ /$$$$      ________________        _
| $$$$| $$|  $$/ $$/| $$$$\ $$|_  $$      \__(=======/_=_/____.--'-`--.___
| $$ $$ $$ \  $$$$/ | $$ $$ $$  | $$                \ \   `,--,-.___.----'
| $$  $$$$  >$$  $$ | $$\ $$$$  | $$              .--`\\--'../
| $$\  $$$ /$$/\  $$| $$ \ $$$  | $$             '---._____.|]
| $$ \  $$| $$  \ $$|  $$$$$$/ /$$$$$$
|__/  \__/|__/  |__/ \______/ |______/

____    ____  ______    __    __      __    __       ___   ____    ____  _______    
\   \  /   / /  __  \  |  |  |  |    |  |  |  |     /   \  \   \  /   / |   ____|   
 \   \/   / |  |  |  | |  |  |  |    |  |__|  |    /  ^  \  \   \/   /  |  |__      
  \_    _/  |  |  |  | |  |  |  |    |   __   |   /  /_\  \  \      /   |   __|     
    |  |    |  `--'  | |  `--'  |    |  |  |  |  /  _____  \  \    /    |  |____    
    |__|     \______/   \______/     |__|  |__| /__/     \__\  \__/     |_______|   
                                                                                    
.___________. __    __   _______      ______   ______   .__   __. .__   __.         
|           ||  |  |  | |   ____|    /      | /  __  \  |  \ |  | |  \ |  |         
`---|  |----`|  |__|  | |  |__      |  ,----'|  |  |  | |   \|  | |   \|  |         
    |  |     |   __   | |   __|     |  |     |  |  |  | |  . `  | |  . `  |         
    |  |     |  |  |  | |  |____    |  `----.|  `--'  | |  |\   | |  |\   |         
    |__|     |__|  |__| |_______|    \______| \______/  |__| \__| |__| \__|         
                                                                                    
"""
if rank == 0:
    print header                                

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

if rank == 0:
    if args.nmodes:
        print ("\n You've given me the number of frequencies",
            "to include in the low-rank time-frequency approximation, got it?\n")
    else:
        print ("\n You've given me the sampling cadence for the observations,",
            "which determines the upper frequency limit and the number of modes, got it?\n")

if args.mnest:
    import pymultinest
else:
    import PALInferencePTMCMC as PAL

#########################################################################
# PASSING THROUGH TEMPO2 VIA libstempo
#########################################################################

# name, hdf5-path, par-path, tim-path
psr_pathinfo = np.genfromtxt(args.psrlist, dtype=str, skip_header=2) 

if args.from_h5:

    tmp_psr = []
    for ii,tmp_name in enumerate(psr_pathinfo[:args.num_psrs,0]):
        tmp_psr.append(h5.File(psr_pathinfo[ii,1], 'r')[tmp_name])

    psr = [NX01_psr.PsrObjFromH5(p) for p in tmp_psr]
    
else:
    
    print 'Are you sure you do not want to use hdf5 files (recommended)?'
    
    t2psr=[]
    for ii in range(args.num_psrs):
        t2psr.append( T2.tempopulsar( parfile=psr_pathinfo[ii,2],
                                      timfile=psr_pathinfo[ii,3] ) )
        t2psr[ii].fit(iters=3)
        if np.any(np.isfinite(t2psr.residuals())==False)==True:
            t2psr = T2.tempopulsar( parfile=psr_pathinfo[ii,2],
                                    timfile=psr_pathinfo[ii,3] )

    psr = [NX01_psr.PsrObj(p) for p in t2psr]


# Grab all the pulsar quantities
[p.grab_all_vars() for p in psr]

# Now, grab the positions and compute the ORF basis functions
psr_positions = [np.array([psr[ii].psr_locs[0],
                           np.pi/2. - psr[ii].psr_locs[1]])
                           for ii in range(len(psr))]
positions = np.array(psr_positions).copy()

num_corr_params = 0
evol_corr_tag = ''
if args.incGWB and args.incCorr:
    
    if args.miCorr:

        gwfreqs_per_win = int(1.*args.nmodes/(1.*args.num_gwfreq_wins)) 
        corr_modefreqs = np.arange(1,args.nmodes+1)
        corr_modefreqs = np.reshape(corr_modefreqs,
                                    (args.num_gwfreq_wins,gwfreqs_per_win))

        tmp_num_gwfreq_wins = args.num_gwfreq_wins

        num_corr_params = tmp_num_gwfreq_wins*(len(psr)*(len(psr)-1)/2)

    if not args.miCorr:
        
        # Computing all the correlation basis-functions for the array.
        CorrCoeff = np.array(anis.CorrBasis(positions,args.LMAX))
        # Computing the values of the spherical-harmonics up to order
        # LMAX on a pre-specified grid  
        harm_sky_vals = utils.SetupPriorSkyGrid(args.LMAX)              
                                                            
        if args.anis_modefile is None:
        
            # getting the number of GW frequencies per window
            gwfreqs_per_win = int(1.*args.nmodes/(1.*args.num_gwfreq_wins)) 
            corr_modefreqs = np.arange(1,args.nmodes+1)
            corr_modefreqs = np.reshape(corr_modefreqs,
                                        (args.num_gwfreq_wins,gwfreqs_per_win))

            tmp_num_gwfreq_wins = args.num_gwfreq_wins
    
        else:

            tmp_modefreqs = np.loadtxt(args.anis_modefile, skiprows=2)
            tmp_num_gwfreq_wins = tmp_modefreqs.shape[0]
            corr_modefreqs = []
    
            for ii in range(tmp_num_gwfreq_wins):
                corr_modefreqs.append(np.arange(tmp_modefreqs[ii,0],
                                                tmp_modefreqs[ii,1]+1))

        num_corr_params = tmp_num_gwfreq_wins*(((args.LMAX+1)**2)-1)

        # Create a tag for evolving anisotropy searches
        if (args.LMAX!=0) and (tmp_num_gwfreq_wins > 1):
            evol_corr_tag = '_evanis'
        else:
            evol_corr_tag = ''
              
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

    nmode = int(round(0.5*Tmax/args.cadence))
    [p.makeTe(nmode, Tmax, makeDM=args.dmVar) for p in psr]
    # get GW frequencies
    fqs = np.linspace(1/Tmax, nmode/Tmax, nmode)

if args.det_signal:

    # find reference time for all pulsars
    tt = [np.min(p.toas) for p in psr]
    tref = np.min(tt)

#######################################
# PRE-COMPUTING WHITE NOISE PROPERTIES 
#######################################

loglike1 = 0
logdet_N = []
TtNT = []
d = []
Jamp = []
for ii,p in enumerate(psr):

    # compute ( T.T * N^-1 * T )
    # & log determinant of N
    new_err = (p.toaerrs).copy()
    if args.fullN==True:
        
        if len(p.ecorrs)>0:

            Jamp.append(np.ones(len(p.epflags)))
            for jj,nano_sysname in enumerate(p.sysflagdict['nano-f'].keys()):
                Jamp[ii][np.where(p.epflags==nano_sysname)] *= \
                  p.ecorrs[nano_sysname]**2.0

            Nx = jitter.cython_block_shermor_0D(p.res, new_err**2.,
                                                Jamp[ii], p.Uinds)
            d.append(np.dot(p.Te.T, Nx))
            
            logdet_N_dummy, TtNT_dummy = \
              jitter.cython_block_shermor_2D(p.Te, new_err**2.,
                                             Jamp[ii], p.Uinds)
            logdet_N.append(logdet_N_dummy)
            TtNT.append(TtNT_dummy)
            
            det_dummy, dtNdt = \
              jitter.cython_block_shermor_1D(p.res, new_err**2.,
                                             Jamp[ii], p.Uinds)

        else:
            
            d.append(np.dot(p.Te.T, p.res/( new_err**2.0 )))
        
            N = 1./( new_err**2.0 )
            right = (N*p.Te.T).T
            TtNT.append(np.dot(p.Te.T, right))
    
            logdet_N.append(np.sum(np.log( new_err**2.0 )))
        
            # triple product in likelihood function
            dtNdt = np.sum(p.res**2.0/( new_err**2.0 ))
        
    else:

        d.append(np.dot(p.Te.T, p.res/( new_err**2.0 )))
            
        N = 1./( new_err**2.0 )
        right = (N*p.Te.T).T
        TtNT.append(np.dot(p.Te.T, right))

        logdet_N.append(np.sum(np.log( new_err**2.0 )))
        
        # triple product in likelihood function
        dtNdt = np.sum(p.res**2.0/( new_err**2.0 ))

    loglike1 += -0.5 * (logdet_N[ii] + dtNdt)


##########################
# SETTING UP PRIOR RANGES
##########################

if args.redSpecModel == 'powerlaw':
    pmin = -20.0*np.ones(len(psr))
    pmin = np.append(pmin,0.0*np.ones(len(psr)))
elif args.redSpecModel == 'spectrum':
    pmin = -20.0*np.ones(npsr*nmode)
if args.dmVar:
    pmin = np.append(pmin,-20.0*np.ones(len(psr)))
    pmin = np.append(pmin,0.0*np.ones(len(psr)))
if args.incGWB:
    if args.gwbSpecModel == 'powerlaw':
        pmin = np.append(pmin,-18.0)
        if not args.fix_slope:
            pmin = np.append(pmin,0.0)
    elif args.gwbSpecModel == 'spectrum':
        pmin = np.append(pmin,-18.0*np.ones(nmode))
    if args.incCorr:
        if args.miCorr:
            pmin = np.append(pmin,np.zeros(num_corr_params))
        else:
            pmin = np.append(pmin,-10.0*np.ones(num_corr_params))
if args.det_signal:
    if args.cgw_search:
        pmin = np.append(pmin,np.array([6.0,0.1,0.0,-9.3,
                                        0.0,-1.0,-1.0,0.0,0.0,0.0]))
        if args.ecc_search:
            pmin = np.append(pmin,0.001)
    if args.bwm_search:
        pmin = np.append(pmin,[np.min([np.min(p.toas) for p in psr]),
                               -18.0,0.0,-1.0,0.0])
        if args.bwm_model_select:
            pmin = np.append(pmin,-0.5)


if args.redSpecModel == 'powerlaw':
    pmax = -11.0*np.ones(len(psr))
    pmax = np.append(pmax,7.0*np.ones(len(psr)))
elif args.redSpecModel == 'spectrum':
    pmax = -11.0*np.ones(npsr*nmode)
if args.dmVar:
    pmax = np.append(pmax,-11.0*np.ones(len(psr)))
    pmax = np.append(pmax,7.0*np.ones(len(psr)))
if args.incGWB:
    if args.gwbSpecModel == 'powerlaw':
        pmax = np.append(pmax,-11.0)
        if not args.fix_slope:
            pmax = np.append(pmax,7.0)
    elif args.gwbSpecModel == 'spectrum':
        pmax = np.append(pmax,-11.0*np.ones(nmode))
    if args.incCorr:
        if args.miCorr:
            pmax = np.append(pmax,np.pi*np.ones(num_corr_params))
        else:
            pmax = np.append(pmax,10.0*np.ones(num_corr_params))
if args.det_signal:
    if args.cgw_search:
        pmax = np.append(pmax,np.array([10.0,1.0,4.0,-7.0,
                                        2.0*np.pi,1.0,1.0,np.pi,np.pi,2.0*np.pi]))
        if args.ecc_search:
            pmax = np.append(pmax,0.9)
    if args.bwm_search:
        pmax = np.append(pmax,[np.max([np.max(p.toas) for p in psr]),
                               -11.0,2.0*np.pi,1.0,np.pi])
        if args.bwm_model_select:
            pmax = np.append(pmax,1.5)

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

    logLike = 0
    loglike1_tmp = loglike1
    dtmp = list(d)
    
    ###############################
    # Splitting up parameter vector

    param_ct = 0
    
    # Pulsar noise parameters
    if args.redSpecModel == 'powerlaw':
        Ared = 10.0**xx[:npsr]
        gam_red = xx[npsr:2*npsr]
        param_ct += 2*npsr
    elif args.redSpecModel == 'spectrum':
        red_spec = (xx[:nmode*npsr].copy()).reshape((npsr,nmode))
        param_ct += npsr*nmode

    mode_count = 2*nmode
    if args.dmVar==True:
        mode_count = 4*nmode
        Adm = 10.0**xx[2*npsr:3*npsr]
        gam_dm = xx[3*npsr:4*npsr]
        param_ct += 2*npsr

    if args.incGWB:
        # GWB parameters
        if args.gwbSpecModel == 'powerlaw':
            Agwb = 10.0**xx[param_ct]
            param_ct += 1
            if args.fix_slope:
                gam_gwb = 13./3
            else:
                gam_gwb = xx[param_ct]
                param_ct += 1
        elif args.gwbSpecModel == 'spectrum':
            rho_spec = xx[param_ct:param_ct+nmode]
            param_ct += nmode

        if args.incCorr:
            # Anisotropy parameters
            orf_coeffs = xx[param_ct:param_ct+num_corr_params]

    # Remaining parameters are for a deterministic signal
    param_ct += num_corr_params

    ###############################
    # Creating continuous GW signal
    
    if args.det_signal:
        if args.cgw_search:

            cgw_params = xx[param_ct:]
    
            if args.ecc_search:
                logmass, qr, logdist, logorbfreq, gwphi,\
                costheta, cosinc, gwpol, gwgamma0, l0, e0 = cgw_params
            else:
                logmass, qr, logdist, logorbfreq, gwphi,\
                costheta, cosinc, gwpol, gwgamma0, l0 = cgw_params

            mc = 10.0**logmass
            dist = 10.0**logdist
            orbfreq = 10.0**logorbfreq
            gwtheta = np.arccos(costheta)
            gwinc = np.arccos(cosinc)
            
            cgw_res = []
            detres = []
            if args.ecc_search:
                for ii,p in enumerate(psr):
                    cgw_res.append( utils.ecc_cgw_signal(p, gwtheta, gwphi, mc, dist,
                                                        orbfreq, gwinc, gwpol, gwgamma0,
                                                        e0, l0, qr, periEv=args.periEv, tref=tref) )
                    detres.append( p.res - cgw_res[ii] )
            else:
                for ii,p in enumerate(psr):
                    cgw_res.append( utils.ecc_cgw_signal(p, gwtheta, gwphi, mc, dist,
                                                        orbfreq, gwinc, gwpol, gwgamma0,
                                                        0.001, l0, qr, periEv=args.periEv, tref=tref) )
                    detres.append( p.res - cgw_res[ii] )

        if args.bwm_search:
            if args.bwm_model_select:
                bwm_params = xx[param_ct:-1]
                # '0' is noise-only, '1' is BWM
                nmodel = int(np.rint(xx[-1]))
            else:
                bwm_params = xx[param_ct:]

            bwm_res = []
            detres = []
            for ii,p in enumerate(psr):
                if args.bwm_model_select:
                    if nmodel==0:
                        bwm_res.append( np.zeros(len(p.toas)) )
                    elif nmodel==1:
                        bwm_res.append( utils.bwmsignal(bwm_params,p) )
                else:
                    bwm_res.append( utils.bwmsignal(bwm_params,p) )
                detres.append( p.res - bwm_res[ii] )
            

        #############################################################
        # Recomputing some noise quantities involving 'residuals'.
        # Unfortunately necessary when we have a deterministic signal.
        
        loglike1_tmp = 0
        dtNdt = []
        for ii,p in enumerate(psr):

            # compute ( T.T * N^-1 * T )
            # & log determinant of N
            new_err = (p.toaerrs).copy()
            if args.fullN==True:
        
                if len(p.ecorrs)>0:
                    Nx = jitter.cython_block_shermor_0D(detres[ii], new_err**2.,
                                                        Jamp[ii], p.Uinds)
                    dtmp[ii] = np.dot(p.Te.T, Nx)
                    det_dummy, dtNdt_dummy = \
                      jitter.cython_block_shermor_1D(detres[ii], new_err**2.,
                                                     Jamp[ii], p.Uinds)
                    dtNdt.append(dtNdt_dummy)

                else:
            
                    dtmp[ii] = np.dot(p.Te.T, detres[ii]/( new_err**2.0 ))
                    dtNdt.append(np.sum(detres[ii]**2.0/( new_err**2.0 )))
                
            else:
        
                dtmp[ii] = np.dot(p.Te.T, detres[ii]/( new_err**2.0 ))
                dtNdt.append(np.sum(detres[ii]**2.0/( new_err**2.0 )))
        
            loglike1_tmp += -0.5 * (logdet_N[ii] + dtNdt[ii])
        
        

            
    if args.incGWB and args.incCorr:
        
        if args.miCorr:

            npairs = npsr*(npsr-1)/2
            phi_corr = orf_coeffs.reshape((tmp_num_gwfreq_wins,npairs))

            ############################################################
            # Computing frequency-dependent overlap reduction functions.

            ORF=[]
            for ii in range(tmp_num_gwfreq_wins): # number of frequency windows
                for jj in range(len(corr_modefreqs[ii])): # number of frequencies in this window
                    upper_triang = np.zeros((npsr,npsr))
                    phi_els = np.array([[0.0]*ii for ii in range(1,npsr)])
                    ct=0
                    for ii in range(len(phi_els)):
                        for jj in range(len(phi_els[ii])):
                            phi_els[ii,jj] = phi_corr[ct]
                            ct += 1

                    upper_triang[0,0] = 1.
                    for jj in range(1,upper_triang.shape[1]):
                        upper_triang[0,jj] = np.cos(phi_els[jj-1][0])
                    for ii in range(1,upper_triang.shape[1]):
                        upper_triang[ii,ii] = np.prod( np.sin(phi_els[ii-1]) )
                    for ii in range(1,upper_triang.shape[1]):
                        for jj in range(ii+1,upper_triang.shape[1]):
                            upper_triang[ii,jj] = np.cos(phi_els[jj-1][ii]) * \
                            np.prod( np.sin(np.array(phi_els[jj-1])[0:ii]) )   

                    ORF.append(np.dot( upper_triang.T, upper_triang ))
       
            if args.dmVar==True:
                for ii in range(tmp_num_gwfreq_wins): # number of frequency windows
                    for jj in range(len(corr_modefreqs[ii])): # number of frequencies in this window
                        ORF.append( np.zeros((npsr,npsr)) )

            ORF = np.array(ORF)
            ORFtot = np.zeros((mode_count,npsr,npsr)) # shouldn't be applying ORF to dmfreqs,
                                                      # but the projection of GW spec onto dmfreqs
                                                      # is defined as zero below.
            ORFtot[0::2] = ORF
            ORFtot[1::2] = ORF
            
        else:
            
            ################################################
            # Reshaping freq-dependent anis coefficients,
            # and testing for power distribution physicality.
            
            orf_coeffs = orf_coeffs.reshape((tmp_num_gwfreq_wins,
                                            ((args.LMAX+1)**2)-1))
            clm = np.array([[0.0]*((args.LMAX+1)**2)
                            for ii in range(tmp_num_gwfreq_wins)])
            clm[:,0] = 2.0*np.sqrt(np.pi)

            if args.LMAX!=0:

                for kk in range(tmp_num_gwfreq_wins):
                    for ii in range(1,((args.LMAX+1)**2)):
                        clm[kk,ii] = orf_coeffs[kk,ii-1]   

                    # Testing for physicality of power distribution.
                    if (utils.PhysPrior(clm[kk],harm_sky_vals) == 'Unphysical'):
                        return -np.inf

            ############################################################
            # Computing frequency-dependent overlap reduction functions.
        
            ORF=[]
            for ii in range(tmp_num_gwfreq_wins): # number of frequency windows
                for jj in range(len(corr_modefreqs[ii])): # number of frequencies in this window
                    ORF.append( sum(clm[ii,kk]*CorrCoeff[kk]
                                    for kk in range(len(CorrCoeff))) )
            if args.dmVar==True:
                for ii in range(tmp_num_gwfreq_wins): # number of frequency windows
                    for jj in range(len(corr_modefreqs[ii])): # number of frequencies in this window
                        ORF.append( np.zeros((npsr,npsr)) )

            ORF = np.array(ORF)
            ORFtot = np.zeros((mode_count,npsr,npsr)) # shouldn't be applying ORF to dmfreqs,
                                                      # but the projection of GW spec onto dmfreqs
                                                      # is defined as zero below.
            ORFtot[0::2] = ORF
            ORFtot[1::2] = ORF

    ################################################
    # parameterize intrinsic red noise as power law
    
    Tspan = (1/fqs[0])*86400.0

    # parameterize intrinsic red-noise and DM-variations
    kappa = [] 
    if args.dmVar:
        for ii in range(npsr):
            kappa.append(np.log10( np.append( Ared[ii]**2/12/np.pi**2 * \
                                              f1yr**(gam_red[ii]-3) * \
                                              (fqs/86400.0)**(-gam_red[ii])/Tspan,
                                        Adm[ii]**2/12/np.pi**2 * \
                                         f1yr**(gam_dm[ii]-3) * \
                                         (fqs/86400.0)**(-gam_dm[ii])/Tspan ) ))
    else:
        for ii in range(npsr):
            if args.redSpecModel == 'powerlaw':
                kappa.append(np.log10( Ared[ii]**2/12/np.pi**2 * \
                                       f1yr**(gam_red[ii]-3) * \
                                       (fqs/86400.0)**(-gam_red[ii])/Tspan ))
            elif args.redSpecModel == 'spectrum':
                kappa.append(np.log10( 10.0**(2.0*red_spec[ii,:]) /
                                       (12.0 * np.pi**2.0 * (fqs/86400.0)**3.0 * Tspan) ))
    
    ###################################
    # construct elements of sigma array
    
    sigdiag = []

    if args.incGWB:

        if args.gwbSpecModel == 'powerlaw':
            rho = np.log10(Agwb**2/12/np.pi**2 * \
                           f1yr**(gam_gwb-3) * \
                           (fqs/86400.0)**(-gam_gwb)/Tspan)
        elif args.gwbSpecModel == 'spectrum':
            rho = np.log10(10.0**(2.0*rho_spec) /
                           (12.0 * np.pi**2.0 * (fqs/86400.0)**3.0 * Tspan))

        if args.dmVar:
            gwbspec = np.append( 10**rho, np.zeros_like(rho) )
        else:
            gwbspec = 10**rho

        if args.incCorr:
            sigoffdiag = []
        

    for ii in range(npsr):
        tot = np.zeros(mode_count)

        # diagonal terms
        tot[0::2] = 10**kappa[ii]
        tot[1::2] = 10**kappa[ii] 

        if args.incGWB:
            
            if args.incCorr:
                
                offdiag = np.zeros(mode_count)

                # off diagonal terms
                offdiag[0::2] = gwbspec
                offdiag[1::2] = gwbspec

                # diagonal terms
                tot[0::2] += ORF[:,ii,ii]*gwbspec
                tot[1::2] += ORF[:,ii,ii]*gwbspec

                sigoffdiag.append(offdiag)
                
            if not args.incCorr:
                
                # diagonal terms
                tot[0::2] += gwbspec
                tot[1::2] += gwbspec
                
        # fill in lists of arrays
        sigdiag.append(tot)


    ###############################################
    # Computing Phi and Sigma matrices without GWB
    
    if not args.incGWB:

        for ii,p in enumerate(psr):
            
            # compute Phi inverse 
            red_phi = np.diag(1./sigdiag[ii])
            logdet_Phi = np.sum(np.log(sigdiag[ii]))

            # now fill in real covariance matrix
            Phi = np.zeros( TtNT[ii].shape ) 
            for kk in range(0,mode_count):
                Phi[kk+p.Gc.shape[1],kk+p.Gc.shape[1]] = red_phi[kk,kk]

            # symmeterize Phi
            Phi = Phi + Phi.T - np.diag(np.diag(Phi))
    
            # compute sigma
            Sigma = TtNT[ii] + Phi

            # cholesky decomp 
            try:
                
                cf = sl.cho_factor(Sigma)
                expval2 = sl.cho_solve(cf, dtmp[ii])
                logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

            except np.linalg.LinAlgError:
                
                print 'Cholesky Decomposition Failed!!'
                return -np.inf
                
            logLike += -0.5 * (logdet_Phi + logdet_Sigma) + \
              0.5 * (np.dot(dtmp[ii], expval2))

        logLike += loglike1_tmp
        

    if args.incGWB:

        if not args.incCorr:
            
            for ii,p in enumerate(psr):
            
                # compute Phi inverse 
                red_phi = np.diag(1./sigdiag[ii])
                logdet_Phi = np.sum(np.log(sigdiag[ii]))

                # now fill in real covariance matrix
                Phi = np.zeros( TtNT[ii].shape ) 
                for kk in range(0,mode_count):
                    Phi[kk+p.Gc.shape[1],kk+p.Gc.shape[1]] = red_phi[kk,kk]

                # symmeterize Phi
                Phi = Phi + Phi.T - np.diag(np.diag(Phi))
    
                # compute sigma
                Sigma = TtNT[ii] + Phi

                # cholesky decomp 
                try:
                    
                    cf = sl.cho_factor(Sigma)
                    expval2 = sl.cho_solve(cf, dtmp[ii])
                    logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

                except np.linalg.LinAlgError:
                    
                    print 'Cholesky Decomposition Failed!!'
                    return -np.inf
                
                logLike += -0.5 * (logdet_Phi + logdet_Sigma) + \
                  0.5 * (np.dot(dtmp[ii], expval2))

            logLike += loglike1_tmp

        if args.incCorr:
        
            #####################
            # compute Phi matrix

            smallMatrix = np.zeros((mode_count, npsr, npsr))
            for ii in range(npsr):
                for jj in range(ii,npsr):

                    if ii == jj:
                        smallMatrix[:,ii,jj] = sigdiag[jj] 
                    else:
                        smallMatrix[:,ii,jj] = ORFtot[:,ii,jj] * sigoffdiag[jj] 
                        smallMatrix[:,jj,ii] = smallMatrix[:,ii,jj]

            ###################################
            # invert Phi matrix frequency-wise
    
            logdet_Phi = 0
            for ii in range(mode_count):

                try:
    
                    L = sl.cho_factor(smallMatrix[ii,:,:])
                    smallMatrix[ii,:,:] = sl.cho_solve(L, np.eye(npsr))
                    logdet_Phi += np.sum(2*np.log(np.diag(L[0])))

                except np.linalg.LinAlgError:
    
                    ###################################################
                    # Break if we have non-positive-definiteness of Phi
            
                    print 'Cholesky Decomposition Failed!! Rejecting...'
                    return -np.inf


            bigTtNT = sl.block_diag(*TtNT)
            Phi = np.zeros_like( bigTtNT )
    
            # now fill in real covariance matrix
            ind = [0]
            ind = np.append(ind,np.cumsum([TtNT[ii].shape[0]
                                        for ii in range(len(psr))]))
            ind = [np.arange(ind[ii]+psr[ii].Gc.shape[1],
                            ind[ii]+psr[ii].Gc.shape[1]+mode_count)
                            for ii in range(len(ind)-1)]
            for ii in range(npsr):
                for jj in range(npsr):
                    Phi[ind[ii],ind[jj]] = smallMatrix[:,ii,jj]
            
            # compute sigma
            Sigma = bigTtNT + Phi
            
            # cholesky decomp for second term in exponential
            if args.use_gpu:

                try:
                
                    dtmp = np.concatenate(dtmp)
                    Sigma_gpu = gpuarray.to_gpu( Sigma.astype(np.float64).copy() )
                    expval2_gpu = gpuarray.to_gpu( dtmp.astype(np.float64).copy() )
                    culinalg.cho_solve( Sigma_gpu, expval2_gpu ) # in-place linear-algebra:
                                                                 # Sigma and expval2 overwritten
                    logdet_Sigma = np.sum(2.0*np.log(np.diag(Sigma_gpu.get())))

                except cula.culaDataError:
    
                    print 'Cholesky Decomposition Failed (GPU error!!)'
                    return -np.inf

                logLike = -0.5 * (logdet_Phi + logdet_Sigma) + \
                  0.5 * (np.dot(dtmp, expval2_gpu.get() )) + \
                  loglike1_tmp
            
            else:
    
                try:

                    dtmp = np.concatenate(dtmp)
                    cf = sl.cho_factor(Sigma)
                    expval2 = sl.cho_solve(cf, dtmp)
                    logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

                except np.linalg.LinAlgError:
                
                    print 'Cholesky Decomposition Failed second time!! Breaking...'
                    return -np.inf


                logLike = -0.5 * (logdet_Phi + logdet_Sigma) + \
                  0.5 * (np.dot(dtmp, expval2)) + \
                  loglike1_tmp

    

    ################################################
    # Multiplying likelihood to correct log-uniform
    # sampling, thus making a uniform prior
    
    if args.incGWB:
        if args.limit_or_detect_gwb == 'limit':
            if args.gwbSpecModel == 'powerlaw':
                priorfac_gwb = np.log(Agwb * np.log(10.0))
            elif args.gwbSpecModel == 'spectrum':
                priorfac_gwb = np.sum(np.log(10.0**rho_spec * np.log(10.0)))
        else:
            priorfac_gwb = 0.0

    if args.limit_or_detect_red == 'limit':
        if args.redSpecModel == 'powerlaw':
            priorfac_red = np.sum(np.log(Ared * np.log(10.0)))
        elif args.redSpecModel == 'spectrum':
            priorfac_red = np.sum(np.log(10.0**red_spec * np.log(10.0)))
    else:
        priorfac_red = 0.0

    if args.dmVar:
        if args.limit_or_detect_dm == 'limit':
            priorfac_dm = np.sum(np.log(Adm * np.log(10.0)))
        else:
            priorfac_dm = 0.0

    #####################################
    # Finally, return the log-likelihood
    
    if args.incGWB:
        if args.dmVar:
            return logLike + priorfac_gwb + priorfac_red + priorfac_dm
        else:
             return logLike + priorfac_gwb + priorfac_red
    else:
        if args.dmVar:
            return logLike + priorfac_red + priorfac_dm
        else:
            return logLike + priorfac_red


#########################
#########################

# Set up the parameter list

parameters=[]
for ii in range(len(psr)):
    if args.redSpecModel == 'powerlaw':
        parameters.append('Ared_'+psr[ii].name)
        parameters.append('gam_red_'+psr[ii].name)
    elif args.redSpecModel == 'spectrum':
        for jj in range(nmode):
            parameters.append('redSpec'+'{0}'.format(jj+1)+psr[ii].name)
if args.dmVar:
    for ii in range(len(psr)):
        parameters.append('Adm_'+psr[ii].name)
        parameters.append('gam_dm_'+psr[ii].name)
if args.incGWB:
    if args.gwbSpecModel == 'powerlaw':
        parameters.append("Agwb")
        if not args.fix_slope:
            parameters.append("gam_gwb")
    elif args.gwbSpecModel == 'spectrum':
        for ii in range(nmode):
            parameters.append('gwbSpec_{0}'.format(ii+1))
    if args.incCorr:
        if args.miCorr:
            for ii in range(num_corr_params):
                parameters.append('phi_corr_{0}'.format(ii+1))
        else:
            for ii in range(num_corr_params):
                parameters.append('clm_{0}'.format(ii+1))
if args.det_signal:
    if args.cgw_search:
        parameters += ["chirpmass", "qratio", "dist", "orb-freq",
                    "phi", "costheta", "cosiota", "gwpol",
                    "gwgamma", "l0"]
        if args.ecc_search:
            parameters.append("ecc")
    if args.bwm_search:
        parameters += ["burst_mjd", "burst_strain", "phi", "costheta", "gwpol"]
        if args.bwm_model_select:
            parameters.append("nmodel")


n_params = len(parameters)
if rank==0:
    print "\n You are searching for the following parameters: {0}\n".format(parameters)
    print "\n The total number of parameters is {0}\n".format(n_params)


# Define a unique file tag

file_tag = 'nanograv'
if args.incGWB:
    if args.gwbSpecModel == 'powerlaw':
        if args.fix_slope:
            gamma_tag = '_gam4p33'
        else:
            gamma_tag = '_gamVary'
    elif args.gwbSpecModel == 'spectrum':
        gamma_tag = '_gwbSpec'
    if args.incCorr:
        if args.miCorr:
            file_tag += '_gwb{0}_miCorr{1}{2}'.format(args.limit_or_detect_gwb,evol_corr_tag,gamma_tag)
        else:
            file_tag += '_gwb{0}_Lmax{1}{2}{3}'.format(args.limit_or_detect_gwb,
                                                       args.LMAX,evol_corr_tag,gamma_tag)
    else:
        file_tag += '_gwb{0}_noCorr{1}'.format(args.limit_or_detect_gwb,gamma_tag)
if args.det_signal:
    if args.cgw_search:
        if args.ecc_search:
            file_tag += '_ecgw'
        else:
            file_tag += '_ccgw'
    if args.bwm_search:
        file_tag += '_bwm'
        if args.bwm_model_select:
            file_tag += 'ModelSelect'
file_tag += '_red{0}{1}_nmodes{2}'.format(args.limit_or_detect_red,args.redSpecModel,args.nmodes)


if rank == 0:
    print "\n Now, we sample... \n"
    print """\
     _______ .__   __.   _______      ___       _______  _______  __  
    |   ____||  \ |  |  /  _____|    /   \     /  _____||   ____||  | 
    |  |__   |   \|  | |  |  __     /  ^  \   |  |  __  |  |__   |  | 
    |   __|  |  . `  | |  | |_ |   /  /_\  \  |  | |_ | |   __|  |  | 
    |  |____ |  |\   | |  |__| |  /  _____  \ |  |__| | |  |____ |__| 
    |_______||__| \__|  \______| /__/     \__\ \______| |_______|(__) 
    
    """

##########################
# Define function wrappers
##########################

if args.mnest:

    dir_name = './chains_nanoAnalysis/'+file_tag+'_mnest'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if rank == 0:
        if args.incCorr:
            # Copy the anisotropy modefile into the results directory
            if args.anis_modefile is not None:
                os.system('cp {0} {1}'.format(args.anis_modefile,dir_name))

        # Printing out the list of searched parameters
        fil = open(dir_name+'/parameter_list.txt','w')
        for ii,parm in enumerate(parameters):
            print >>fil, ii, parm
        fil.close()

        # Saving command-line arguments to file
        with open(dir_name+'/run_args.json', 'w') as fp:
            json.dump(vars(args), fp)

    def prior_func(xx,ndim,nparams):
        for ii in range(nparams):
            xx[ii] = pmin[ii] + xx[ii]*(pmax[ii]-pmin[ii])
            
    def like_func(xx,ndim,nparams):
        xx = np.array([xx[ii] for ii in range(nparams)])
        return lnprob(xx)        
    
    pymultinest.run(like_func, prior_func, n_params,
                    importance_nested_sampling = False,
                    resume = False, verbose = True, 
                    n_live_points=500,
                    outputfiles_basename=u'{0}/mnest_'.format(dir_name), 
                    sampling_efficiency=0.3,
                    const_efficiency_mode=False)

if not args.mnest:
    
    # Start the sampling off with some reasonable parameter choices
    if args.redSpecModel == 'powerlaw':
        x0 = np.log10(np.array([p.Redamp for p in psr]))
        x0 = np.append(x0,np.array([p.Redind for p in psr]))
    elif args.redSpecModel == 'spectrum':
        x0 = np.random.uniform(-17.0,-13.0,len(psr)*nmode)
    if args.dmVar:
        x0 = np.append(x0,np.log10(np.array([p.Redamp for p in psr])))
        x0 = np.append(x0,np.array([p.Redind for p in psr]))
    if args.incGWB:
        if args.gwbSpecModel == 'powerlaw':
            x0 = np.append(x0,-15.0)
            if not args.fix_slope:
                x0 = np.append(x0,13./3.)
        elif args.gwbSpecModel == 'spectrum':
            x0 = np.append(x0,np.random.uniform(-16.0,-15.0,nmode))
        if args.incCorr:
            if args.miCorr:
                x0 = np.append(x0,np.random.uniform(0.0,np.pi,num_corr_params))
            else:
                x0 = np.append(x0,np.zeros(num_corr_params))
    if args.det_signal:
        if args.cgw_search:
            x0 = np.append(x0,np.array([9.0, 0.5, 1.5, -8.0, 0.5,
                                        0.5, 0.5, 0.5, 0.5, 0.5]))
            if args.ecc_search:
                x0 = np.append(x0,0.1)
        if args.bwm_search:
            x0 = np.append(x0,np.array([55100.0,-14.0,0.3,0.5,0.7]))
            if args.bwm_model_select:
                x0 = np.append(x0,0.4)

    if rank==0:
        print "\n Your initial parameters are {0}\n".format(x0)

    # Make a reasonable covariance matrix to commence sampling
    if args.redSpecModel == 'powerlaw':
        cov_diag = 0.5*np.ones(len(psr))
        cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))
    elif args.redSpecModel == 'spectrum':
        cov_diag = 0.1*np.ones(len(psr)*nmode)
    if args.dmVar:
        cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))
        cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))
    if args.incGWB:
        if args.gwbSpecModel == 'powerlaw':
            cov_diag = np.append(cov_diag,0.5)
            if not args.fix_slope:
                cov_diag = np.append(cov_diag,0.5)
        elif args.gwbSpecModel == 'spectrum':
            cov_diag = np.append(cov_diag,0.05*np.ones(nmode))
        if args.incCorr:
            cov_diag = np.append(cov_diag,0.05*np.ones(num_corr_params))
    if args.det_signal:
        if args.cgw_search:
            cov_diag = np.append(cov_diag,0.2*np.ones(10))
            if args.ecc_search:
                cov_diag = np.append(cov_diag,0.05)
        if args.bwm_search:
            cov_diag = np.append(cov_diag,np.array([10.0,0.1,0.1,0.1,0.1]))
            if args.bwm_model_select:
                cov_diag = np.append(cov_diag,0.1)

    if rank==0:
        print "\n Running a quick profile on the likelihood to estimate evaluation speed...\n"
        cProfile.run('lnprob(x0)')
    
    
    sampler = PAL.PTSampler(ndim=n_params,logl=lnprob,logp=my_prior,
                            cov=np.diag(cov_diag),
                            outDir='./chains_11yrnanoAnalysis/'+file_tag,
                            resume=True)

    if rank == 0:
        if args.incCorr:
            # Copy the anisotropy modefile into the results directory
            if args.anis_modefile is not None:
                os.system('cp {0} {1}'.format(args.anis_modefile,
                                              './chains_11yrnanoAnalysis/'+file_tag))

        # Printing out the list of searched parameters
        fil = open('./chains_11yrnanoAnalysis/'+file_tag+'/parameter_list.txt','w')
        for ii,parm in enumerate(parameters):
            print >>fil, ii, parm
        fil.close()

        # Saving command-line arguments to file
        with open('./chains_11yrnanoAnalysis/'+file_tag+'/run_args.json', 'w') as fp:
            json.dump(vars(args), fp)

    #####################################
    # MCMC jump proposals
    #####################################

    # red noise draws (from Justin Ellis' PAL2)
    def drawFromRedpowerlawNoisePrior(parameters, iter, beta):
    
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)

        #ind = np.unique(np.random.randint(0, npsr, npsr))
        ind = np.unique(np.random.randint(0, npsr, 1))

        for ii in ind:
            # log prior
            if args.limit_or_detect_red == 'detect':
        
                q[ii] = np.random.uniform(pmin[ii], pmax[ii])
                qxy += 0
        
            elif args.limit_or_detect_red == 'limit':
        
                q[ii] = np.random.uniform(pmin[ii], pmax[ii])
                qxy += 0

                #Ared = np.log10(np.random.uniform(10 ** Ared_ll, 10 ** Ared_ul, len(Ared)))
                #qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])
    
            q[npsr+ii] = np.random.uniform(pmin[npsr+ii], pmax[npsr+ii])
            qxy += 0

        return q, qxy

    # red noise draws (from Justin Ellis' PAL2)
    def drawFromRedspectrumNoisePrior(parameters, iter, beta):
    
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)

        ind = np.unique(np.random.randint(0, npsr*nmode, 1))

        for ii in ind:
            # log prior
            if args.limit_or_detect_red == 'detect':
        
                q[ii] = np.random.uniform(pmin[ii], pmax[ii])
                qxy += 0
        
            elif args.limit_or_detect_red == 'limit':
        
                q[ii] = np.random.uniform(pmin[ii], pmax[ii])
                qxy += 0

        return q, qxy

    # dm var draws 
    def drawFromDMNoisePrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)

        #ind = np.unique(np.random.randint(0, npsr, npsr))
        ind = np.unique(np.random.randint(0, npsr, 1))

        for ii in ind:
            # log prior
            if args.limit_or_detect_dm == 'detect':
        
                q[2*npsr+ii] = np.random.uniform(pmin[2*npsr+ii], pmax[2*npsr+ii])
                qxy += 0
        
            elif args.limit_or_detect_dm == 'limit':
            
                q[2*npsr+ii] = np.random.uniform(pmin[2*npsr+ii], pmax[2*npsr+ii])
                qxy += 0

                #Ared = np.log10(np.random.uniform(10 ** Ared_ll, 10 ** Ared_ul, len(Ared)))
                #qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])
    
            q[3*npsr+ii] = np.random.uniform(pmin[3*npsr+ii], pmax[3*npsr+ii])
            qxy += 0

        return q, qxy


    # gwb draws 
    def drawFromGWBpowerlawPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 2*npsr
    
        if args.dmVar:
            pct += 2*npsr

        # log prior
        if args.limit_or_detect_gwb == 'detect':
        
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0

        elif args.limit_or_detect_gwb == 'limit':
            
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0

            #Ared = np.log10(np.random.uniform(10 ** Ared_ll, 10 ** Ared_ul, len(Ared)))
            #qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])


        if not args.fix_slope:
            q[pct+1] = np.random.uniform(pmin[pct+1], pmax[pct+1])
            qxy += 0
        
        return q, qxy

    def drawFromGWBspectrumPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 2*npsr
    
        if args.dmVar:
            pct += 2*npsr

        ind = np.unique(np.random.randint(0, nmode, 1))

        for ii in ind:
            # log prior
            if args.limit_or_detect_gwb == 'detect':

                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0

            elif args.limit_or_detect_gwb == 'limit':
            
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0
        
        return q, qxy

    # cgw draws 
    def drawFromCWPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 2*npsr
    
        if args.dmVar:
            pct += 2*npsr

        if args.incGWB:
            pct += 1
            if not args.fix_slope:
                pct += 1

            if args.incCorr:
                pct += num_corr_params

        # logmass, qr, logdist, logorbfreq, gwphi,
        # costheta, cosinc, gwpol, gwgamma0, l0
        if args.ecc_search:
            ind = np.unique(np.random.randint(0, 11, 1))
        else:
            ind = np.unique(np.random.randint(0, 10, 1))

        for ii in ind:
            q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
            qxy += 0
        
        return q, qxy

    # bwm draws 
    def drawFromBWMPrior(parameters, iter, beta):
    
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 2*npsr
    
        if args.dmVar:
            pct += 2*npsr

        if args.incGWB:
            pct += 1
            if not args.fix_slope:
                pct += 1

            if args.incCorr:
                pct += num_corr_params

        # burst_mjd, burst_amp, phi, costheta, gwpol
        ind = np.unique(np.random.randint(0, 5, 1))

        for ii in ind:
            q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
            qxy += 0
        
        return q, qxy

    # bwm model index draws 
    def drawFromModelIndexPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 2*npsr
    
        if args.dmVar:
            pct += 2*npsr

        if args.incGWB:
            pct += 1
            if not args.fix_slope:
                pct += 1

            if args.incCorr:
                pct += num_corr_params

        if args.det_signal and args.bwm_search:
            pct += 5
            # indexing parameter is at end of list
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0
        
        return q, qxy

  

    # add jump proposals
    if args.redSpecModel == 'powerlaw':
        sampler.addProposalToCycle(drawFromRedpowerlawNoisePrior, 10)
    elif args.redSpecModel == 'spectrum':
        sampler.addProposalToCycle(drawFromRedspectrumNoisePrior, 10)
    if args.dmVar:
        sampler.addProposalToCycle(drawFromDMNoisePrior, 10)
    if args.incGWB:
        if args.gwbSpecModel == 'powerlaw':
            sampler.addProposalToCycle(drawFromGWBpowerlawPrior, 10)
        elif args.gwbSpecModel == 'spectrum':
            sampler.addProposalToCycle(drawFromGWBspectrumPrior, 10)
    if args.det_signal and args.cgw_search:
        sampler.addProposalToCycle(drawFromCWPrior, 10)
    if args.det_signal and args.bwm_search:
        sampler.addProposalToCycle(drawFromBWMPrior, 10)
        if args.bwm_model_select:
            sampler.addProposalToCycle(drawFromModelIndexPrior, 5)

    sampler.sample(p0=x0,Niter=5e6,thin=10,
                covUpdate=1000, AMweight=20,
                SCAMweight=30, DEweight=50, KDEweight=0)
