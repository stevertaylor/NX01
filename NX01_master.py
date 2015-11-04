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
parser.add_option('--ptmcmc', dest='ptmcmc', action='store_true', default=True,
                   help='Sample using PALs parallel tempering MCMC (False)? (default=True)')
parser.add_option('--incGWB', dest='incGWB', action='store_true', default=False,
                  help='Do you want to search for a GWB? (default = False)')
parser.add_option('--incCorr', dest='incCorr', action='store_true', default=False,
                  help='Do you want to include cross-correlations in the GWB model? (default = False)')
parser.add_option('--num_gwfreq_wins', dest='num_gwfreq_wins', action='store', type=int, default=1,
                   help='Number windows to split the band into (useful for evolving anisotropic searches (default = 1 windows)')
parser.add_option('--lmax', dest='LMAX', action='store', type=int, default=0,
                   help='Maximum multipole in anisotropic search (default = 0, i.e. isotropic-search)')
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
parser.add_option('--cgw-search', dest='cgw_search', action='store_true', default=False,
                  help='Do you want to search for a single GW source? (default = False)')
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

if args.ptmcmc:
    import PALInferencePTMCMC as PAL
else:
    import pymultinest

################################################################################################################################
# PASSING THROUGH TEMPO2 VIA libstempo
################################################################################################################################

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

num_anis_params = 0
evol_anis_tag = ''
if args.incGWB and args.incCorr:
    # Computing all the correlation basis-functions for the array.
    CorrCoeff = np.array(anis.CorrBasis(positions,args.LMAX))
    # Computing the values of the spherical-harmonics up to order
    # LMAX on a pre-specified grid  
    harm_sky_vals = utils.SetupPriorSkyGrid(args.LMAX)              

                                                            
    if args.anis_modefile is None:
    
        # getting the number of GW frequencies per window
        gwfreqs_per_win = int(1.*args.nmodes/(1.*args.num_gwfreq_wins)) 
        anis_modefreqs = np.arange(1,args.nmodes+1)
        anis_modefreqs = np.reshape(anis_modefreqs,
                                    (args.num_gwfreq_wins,gwfreqs_per_win))

        tmp_num_gwfreq_wins = args.num_gwfreq_wins
    
    else:

        tmp_modefreqs = np.loadtxt(args.anis_modefile)
        tmp_num_gwfreq_wins = tmp_modefreqs.shape[0]
        anis_modefreqs = []
    
        for ii in range(tmp_num_gwfreq_wins):
            anis_modefreqs.append(np.arange(tmp_modefreqs[ii,0],
                                            tmp_modefreqs[ii,1]+1))

    num_anis_params = tmp_num_gwfreq_wins*(((args.LMAX+1)**2)-1)

    # Create a tag for evolving anisotropy searches
    if (args.LMAX!=0) and (tmp_num_gwfreq_wins > 1):
        evol_anis_tag = '_evanis'
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

    nmode = int(round(0.5*Tmax/args.cadence))
    [p.makeTe(nmode, Tmax, makeDM=args.dmVar) for p in psr]
    # get GW frequencies
    fqs = np.linspace(1/Tmax, nmode/Tmax, nmode)

if args.cgw_search:

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
                Jamp[ii][np.where(p.epflags==nano_sysname)] *= p.ecorrs[nano_sysname]**2.0

            Nx = jitter.cython_block_shermor_0D(p.res, new_err**2., Jamp[ii], p.Uinds)
            d.append(np.dot(p.Te.T, Nx))
            
            logdet_N_dummy, TtNT_dummy = jitter.cython_block_shermor_2D(p.Te, new_err**2., Jamp[ii], p.Uinds)
            logdet_N.append(logdet_N_dummy)
            TtNT.append(TtNT_dummy)
            
            det_dummy, dtNdt = jitter.cython_block_shermor_1D(p.res, new_err**2., Jamp[ii], p.Uinds)

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

pmin = -20.0*np.ones(len(psr))
pmin = np.append(pmin,0.0*np.ones(len(psr)))
if args.dmVar==True:
    pmin = np.append(pmin,-20.0*np.ones(len(psr)))
    pmin = np.append(pmin,0.0*np.ones(len(psr)))
if args.incGWB:
    pmin = np.append(pmin,-18.0)
    if not args.fix_slope:
        pmin = np.append(pmin,0.0)
    if args.incCorr:
        pmin = np.append(pmin,-10.0*np.ones(num_anis_params))
if args.cgw_search:
    pmin = np.append(pmin,np.array([6.0,0.1,0.0,-9.3,
                                    0.0,-1.0,-1.0,0.0,0.0,0.0]))
    if args.ecc_search:
        pmin = np.append(pmin,0.001)


pmax = -11.0*np.ones(len(psr))
pmax = np.append(pmax,7.0*np.ones(len(psr)))
if args.dmVar==True:
    pmax = np.append(pmax,-11.0*np.ones(len(psr)))
    pmax = np.append(pmax,7.0*np.ones(len(psr)))
if args.incGWB:
    pmax = np.append(pmax,-11.0)
    if not args.fix_slope:
        pmax = np.append(pmax,7.0)
    if args.incCorr:
        pmax = np.append(pmax,10.0*np.ones(num_anis_params))
if args.cgw_search:
    pmax = np.append(pmax,np.array([10.0,1.0,4.0,-6.5,
                                    2.0*np.pi,1.0,1.0,np.pi,np.pi,2.0*np.pi]))
    if args.ecc_search:
        pmax = np.append(pmax,0.9)

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

    if args.dmVar:
        mode_count = 4*nmode
        if args.incGWB:
            if args.incCorr:
                Ared, gam_red, Adm, gam_dm, Agwb, gam_gwb, orf_coeffs, param_ct = \
                  utils.masterSplitParams(xx, npsr, args.dmVar, args.fix_slope,
                                          args.incGWB, args.incCorr, num_anis_params )
            else:
                Ared, gam_red, Adm, gam_dm, Agwb, gam_gwb, param_ct = \
                  utils.masterSplitParams(xx, npsr, args.dmVar, args.fix_slope,
                                          args.incGWB, args.incCorr, num_anis_params )
        else:
            Ared, gam_red, Adm, gam_dm, param_ct = \
              utils.masterSplitParams(xx, npsr, args.dmVar, args.fix_slope,
                                      args.incGWB, args.incCorr, num_anis_params )
            
    else:
        mode_count = 2*nmode
        if args.incGWB:
            if args.incCorr:
                Ared, gam_red, Agwb, gam_gwb, orf_coeffs, param_ct = \
                  utils.masterSplitParams(xx, npsr, args.dmVar, args.fix_slope,
                                          args.incGWB, args.incCorr, num_anis_params )
            else:
                Ared, gam_red, Agwb, gam_gwb, param_ct = \
                  utils.masterSplitParams(xx, npsr, args.dmVar, args.fix_slope,
                                          args.incGWB, args.incCorr, num_anis_params )
        else:
            Ared, gam_red, param_ct = \
              utils.masterSplitParams(xx, npsr, args.dmVar, args.fix_slope,
                                      args.incGWB, args.incCorr, num_anis_params )
        

    ###############################
    # Creating continuous GW signal
    
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

        #############################################################
        # Recomputing some noise quantities involving 'residuals'.
        # Unfortunately necessary when we have a deterministic signal.
        
        loglike1_tmp = 0
        #d = []
        dtNdt = []
        for ii,p in enumerate(psr):

            # compute ( T.T * N^-1 * T )
            # & log determinant of N
            new_err = (p.toaerrs).copy()
            if args.fullN==True:
        
                if len(p.ecorrs)>0:
                    Nx = jitter.cython_block_shermor_0D(detres[ii], new_err**2.,
                                                        Jamp[ii], p.Uinds)
                    #d.append(np.dot(p.Te.T, Nx))
                    dtmp[ii] = np.dot(p.Te.T, Nx)
                    det_dummy, dtNdt_dummy = \
                      jitter.cython_block_shermor_1D(detres[ii], new_err**2.,
                                                     Jamp[ii], p.Uinds)
                    dtNdt.append(dtNdt_dummy)

                else:
            
                    #d.append(np.dot(p.Te.T, detres[ii]/( new_err**2.0 )))
                    dtmp[ii] = np.dot(p.Te.T, detres[ii]/( new_err**2.0 ))
                    dtNdt.append(np.sum(detres[ii]**2.0/( new_err**2.0 )))
                
            else:
        
                #d.append(np.dot(p.Te.T, detres[ii]/( new_err**2.0 )))
                dtmp[ii] = np.dot(p.Te.T, detres[ii]/( new_err**2.0 ))
                dtNdt.append(np.sum(detres[ii]**2.0/( new_err**2.0 )))
        
            loglike1_tmp += -0.5 * (logdet_N[ii] + dtNdt[ii])
        
    
    if args.incGWB and args.incCorr:
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
        # Computing frequency dependent overlap reduction functions.
        
        ORF=[]
        for ii in range(tmp_num_gwfreq_wins): # number of frequency windows
            for jj in range(len(anis_modefreqs[ii])): # number of frequencies in this window
                ORF.append( sum(clm[ii,kk]*CorrCoeff[kk]
                                for kk in range(len(CorrCoeff))) )
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
    
    ################################################
    # parameterize intrinsic red noise as power law
    
    Tspan = (1/fqs[0])*86400.0

    # parameterize intrinsic red-noise and DM-variations as power law
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
            kappa.append(np.log10( Ared[ii]**2/12/np.pi**2 * \
                                   f1yr**(gam_red[ii]-3) * \
                                   (fqs/86400.0)**(-gam_red[ii])/Tspan ))
    
    ###################################
    # construct elements of sigma array
    
    sigdiag = []

    if args.incGWB:

        rho = np.log10(Agwb**2/12/np.pi**2 * \
                       f1yr**(gam_gwb-3) * \
                       (fqs/86400.0)**(-gam_gwb)/Tspan)

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
            priorfac_gwb = np.log(Agwb * np.log(10.0))
        else:
            priorfac_gwb = 0.0

    if args.limit_or_detect_red == 'limit':
        priorfac_red = np.sum(np.log(Ared * np.log(10.0)))
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
    parameters.append('Ared_'+psr[ii].name)
for ii in range(len(psr)):
    parameters.append('gam_red_'+psr[ii].name)
if args.dmVar:
    for ii in range(len(psr)):
        parameters.append('Adm_'+psr[ii].name)
    for ii in range(len(psr)):
        parameters.append('gam_dm_'+psr[ii].name)
if args.incGWB:
    parameters.append("Agwb")
    if not args.fix_slope:
        parameters.append("gam_gwb")
    if args.incCorr:
        for ii in range(num_anis_params):
            parameters.append('clm_{0}'.format(ii+1))
if args.cgw_search:
    parameters += ["chirpmass", "qratio", "dist", "orb-freq",
                   "phi", "costheta", "cosiota", "gwpol",
                   "gwgamma", "l0"]
    if args.ecc_search:
        parameters.append("ecc")


n_params = len(parameters)
if rank==0:
    print "\n You are searching for the following parameters: {0}\n".format(parameters)
    print "\n The total number of parameters is {0}\n".format(n_params)


# Start the sampling off with some reasonable parameter choices
x0 = np.log10(np.array([p.Redamp for p in psr]))
x0 = np.append(x0,np.array([p.Redind for p in psr]))
if args.dmVar:
    x0 = np.append(x0,np.log10(np.array([p.Redamp for p in psr])))
    x0 = np.append(x0,np.array([p.Redind for p in psr]))
if args.incGWB:
    x0 = np.append(x0,-15.0)
    if not args.fix_slope:
        x0 = np.append(x0,13./3.)
    if args.incCorr:
        x0 = np.append(x0,np.zeros(num_anis_params))
if args.cgw_search:
    x0 = np.append(x0,np.array([9.0, 0.5, 1.5, -8.0, 0.5,
                                0.5, 0.5, 0.5, 0.5, 0.5]))
    if args.ecc_search:
        x0 = np.append(x0,0.1)

if rank==0:
    print "\n Your initial parameters are {0}\n".format(x0)

# Make a reasonable covariance matrix to commence sampling
cov_diag = 0.5*np.ones(len(psr))
cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))
if args.dmVar:
    cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))
    cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))
if args.incGWB:
    cov_diag = np.append(cov_diag,0.5)
    if not args.fix_slope:
        cov_diag = np.append(cov_diag,0.5)
    if args.incCorr:
        cov_diag = np.append(cov_diag,0.05*np.ones(num_anis_params))
if args.cgw_search:
    cov_diag = np.append(cov_diag,0.2*np.ones(10))
    if args.ecc_search:
        cov_diag = np.append(cov_diag,0.05)

if rank==0:
    print "\n Running a quick profile on the likelihood to estimate evaluation speed...\n"
    cProfile.run('lnprob(x0)')

#####################
# Now, we sample.....
#####################

file_tag = 'nanograv'
if args.incGWB:    
    if args.fix_slope:
        gamma_tag = '_gam4p33'
    else:
        gamma_tag = '_gamVary'

    if args.incCorr:
        file_tag += '_gwb{0}_Lmax{1}{2}{3}'.format(args.limit_or_detect_gwb,args.LMAX,evol_anis_tag,gamma_tag)
    else:
        file_tag += '_gwb{0}_noCorr{1}'.format(args.limit_or_detect_gwb,gamma_tag)
if args.cgw_search:
    if args.ecc_search:
        file_tag += '_ecgw'
    else:
        file_tag += '_ccgw'
file_tag += '_red{0}_nmodes{1}'.format(args.limit_or_detect_red,args.nmodes)



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

    
sampler = PAL.PTSampler(ndim=n_params,logl=lnprob,logp=my_prior,cov=np.diag(cov_diag),\
                        outDir='./chains_nanoAnalysis/'+file_tag, resume=False)

if rank == 0:
    if args.incCorr:
        # Copy the anisotropy modefile into the results directory
        if args.anis_modefile is not None:
            os.system('cp {0} {1}'.format(args.anis_modefile,'./chains_nanoAnalysis/'+file_tag))

    # Printing out the list of searched parameters
    fil = open('./chains_nanoAnalysis/'+file_tag+'/parameter_list.txt','w')
    for ii,parm in enumerate(parameters):
        print >>fil, ii, parm
    fil.close()

#####################################
# MCMC jump proposals
#####################################

# red noise draws (from Justin Ellis' PAL2)
def drawFromRedNoisePrior(parameters, iter, beta):

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
        
            #q[:npsr] = np.random.uniform(pmin[0], pmax[0], npsr)
            q[ii] = np.random.uniform(pmin[ii], pmax[ii])
            qxy += 0
        
        elif args.limit_or_detect_red == 'limit':
        
            #q[:npsr] = np.random.uniform(pmin[0], pmax[0], npsr)
            q[ii] = np.random.uniform(pmin[ii], pmax[ii])
            qxy += 0

            #Ared = np.log10(np.random.uniform(10 ** Ared_ll, 10 ** Ared_ul, len(Ared)))
            #qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])
    
        #q[npsr:2*npsr] = np.random.uniform(pmin[npsr], pmax[npsr], npsr)
        q[npsr+ii] = np.random.uniform(pmin[npsr+ii], pmax[npsr+ii])
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
        
            #q[2*npsr:3*npsr] = np.random.uniform(pmin[2*npsr], pmax[2*npsr], npsr)
            q[2*npsr+ii] = np.random.uniform(pmin[2*npsr+ii], pmax[2*npsr+ii])
            qxy += 0
        
        elif args.limit_or_detect_dm == 'limit':
            
            #q[2*npsr:3*npsr] = np.random.uniform(pmin[2*npsr], pmax[2*npsr], npsr)
            q[2*npsr+ii] = np.random.uniform(pmin[2*npsr+ii], pmax[2*npsr+ii])
            qxy += 0

            #Ared = np.log10(np.random.uniform(10 ** Ared_ll, 10 ** Ared_ul, len(Ared)))
            #qxy += np.log(10 ** parameters[parind] / 10 ** q[parind])
    
        #q[3*npsr:4*npsr] = np.random.uniform(pmin[3*npsr], pmax[3*npsr], npsr)
        q[3*npsr+ii] = np.random.uniform(pmin[3*npsr+ii], pmax[3*npsr+ii])
        qxy += 0

    return q, qxy


# gwb draws (from Justin Ellis' PAL2)
def drawFromGWBPrior(parameters, iter, beta):

    # post-jump parameters
    q = parameters.copy()

    # transition probability
    qxy = 0

    npsr = len(psr)
    pct = 2*npsr
    
    if args.dmVar==True:
        pct += 2*npsr

    # log prior
    if args.limit_or_detect_gwb == 'detect':
        
        Agwb_samp = np.random.uniform(pmin[pct], pmax[pct])
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

  

# add jump proposals
sampler.addProposalToCycle(drawFromRedNoisePrior, 10)
if args.dmVar:
    sampler.addProposalToCycle(drawFromDMNoisePrior, 10)
if args.incGWB:
    sampler.addProposalToCycle(drawFromGWBPrior, 10)
#if args.cgw_search:
#    sampler.addProposalToCycle(drawFromCWPrior, 10)


sampler.sample(p0=x0,Niter=1e6,thin=10)
