#!/usr/bin/env python

"""
Created by stevertaylor
Copyright (c) 2014 Stephen R. Taylor

Code contributions by Rutger van Haasteren (piccard) and Justin Ellis (PAL/PAL2).

"""

from __future__ import division
import os, math, optparse, time, cProfile
import json, sys
import cPickle as pickle
from time import gmtime, strftime
from collections import OrderedDict
import h5py as h5

import numpy as np
from numpy import *
from numpy import random

from scipy import integrate
from scipy import optimize
from scipy import constants as sc
from scipy import special as ss
from scipy import linalg as sl
from scipy.interpolate import interp1d

import numexpr as ne
import ephem
from ephem import *

import libstempo as T2

import NX01_AnisCoefficients as anis
import NX01_utils as utils
import NX01_psr

try:
    import NX01_jitter as jitter
except ImportError:
    print "You do not have NX01_jitter.so. " \
      "Trying to make the .so file now..."
    import pyximport
    pyximport.install(setup_args={"include_dirs":np.get_include()},
                      reload_support=True)
    try:
        import NX01_jitter as jitter
    except ImportError:
        error_warning = """\
         _____ __  __ _____   ____  _____ _______   ______ _____  _____   ____  _____  _ _          
        |_   _|  \/  |  __ \ / __ \|  __ \__   __| |  ____|  __ \|  __ \ / __ \|  __ \| | |         
          | | | \  / | |__) | |  | | |__) | | |    | |__  | |__) | |__) | |  | | |__) | | |         
          | | | |\/| |  ___/| |  | |  _  /  | |    |  __| |  _  /|  _  /| |  | |  _  /| | |         
         _| |_| |  | | |    | |__| | | \ \  | |    | |____| | \ \| | \ \| |__| | | \ \|_|_|         
        |_____|_|  |_|_|     \____/|_|  \_\ |_|    |______|_|  \_\_|  \_\\____/|_|  \_(_|_)         
         _____ ____  __  __ _____ _____ _      ______        _ _____ _______ _______ ______ _____  
        / ____/ __ \|  \/  |  __ \_   _| |    |  ____|      | |_   _|__   __|__   __|  ____|  __ \ 
       | |   | |  | | \  / | |__) || | | |    | |__         | | | |    | |     | |  | |__  | |__) |
       | |   | |  | | |\/| |  ___/ | | | |    |  __|    _   | | | |    | |     | |  |  __| |  _  / 
       | |___| |__| | |  | | |    _| |_| |____| |____  | |__| |_| |_   | |     | |  | |____| | \ \ 
        \_____\____/|_|  |_|_|   |_____|______|______|  \____/|_____|  |_|     |_|  |______|_|  \_\
        """
        print error_warning
        print "You need to run: " \
          "python setup-cython.py build_ext --inplace"
        sys.exit()
    

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    print 'Do not have mpi4py package.'
    import nompi4py as MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

f1yr = 1.0/(365.25*86400.0)

parser = optparse.OptionParser(description = "NX01 - It's been a long road, getting from there to here...")

############################
############################

parser.add_option('--jsonModel', dest='jsonModel', action='store', type=str, default = None,
                   help='Do you want to provide model arguments from a JSON file? (default = None)')
parser.add_option('--from-h5', dest='from_h5', action='store_true', default = False,
                   help='Do you want to read in pulsars from hdf5 files instead of directly via libstempo? (default = False)')
parser.add_option('--psrlist', dest='psrlist', action='store', type=str, default = None,
                   help='Provide path to file containing list of pulsars and their respective par/tim paths')
parser.add_option('--sysflag_target', dest='sysflag_target', action='store', type=str, default = 'f',
                   help='If you are supplying pulsar noise files, then specify which system flag you want to target (default = f)')
parser.add_option('--parfile', dest='parfile', action='store', type=str, default = None,
                   help='Provide path to a pulsar par file for single-pulsar analysis (default = None)')
parser.add_option('--timfile', dest='timfile', action='store', type=str, default = None,
                   help='Provide path to a pulsar tim file for single-pulsar analysis (default = None)')
parser.add_option('--jitterbin', dest='jitterbin', action='store', type=float, default = 10.0,
                   help='Provide size of jitter binning for single-pulsar analysis (default = 10 seconds)')
parser.add_option('--grab_planets', dest='grab_planets', action='store_true', default=False,
                   help='Do you want to grab the planetary position vectors in your single-pulsar analysis (default = False)')
parser.add_option('--nmodes', dest='nmodes', action='store', type=int, default=None,
                   help='Number of modes in low-rank time-frequency approximation')
parser.add_option('--cadence', dest='cadence', action='store', type=float,
                   help='Instead of nmodes, provide the observational cadence.')
parser.add_option('--incDM', dest='incDM', action='store_true', default=False,
                   help='Search for DM variations in the data as a Gaussian process (False)? (default=False)')
parser.add_option('--varyWhite', dest='varyWhite', action='store_true', default=False,
                   help='Search for per-pulsar white-noise parameters? (default=False)')
parser.add_option('--sampler', dest='sampler', action='store', type=str, default='ptmcmc',
                   help='Which sampler do you want to use: PTMCMC (ptmcmc), MultiNest (mnest), or Polychord (pchord) (default = ptmcmc)')
parser.add_option('--ins', dest='ins', action='store_true', default=False,
                   help='Switch on importance nested sampling for MultiNest (default = False)')
parser.add_option('--nlive', dest='nlive', action='store', type=int, default=500,
                   help='Number of live points for MultiNest or Polychord (default = 500)')
parser.add_option('--sampleEff', dest='sampleEff', action='store', type=float, default=0.3,
                   help='Sampling efficiency for MultiNest (default = 0.3)')
parser.add_option('--constEff', dest='constEff', action='store_true', default=False,
                   help='Run MultiNest in constant efficiency mode? (default = False)')
parser.add_option('--nchords', dest='nchords', action='store', type=int, default=1,
                   help='Number of chords for Polychord (default = 1)')
parser.add_option('--resume', dest='resume', action='store_true', default=False,
                   help='Do you want to resume the sampler (default = False)')
parser.add_option('--niter', dest='niter', action='store', type=float, default=5e6,
                   help='Number of MCMC iterations for PTMCMC sampler (default = 5e6)')
parser.add_option('--writeHotChains', dest='writeHotChains', action='store_true', default=False,
                   help='Given a PTMCMC sampler, do you want to write out the hot chain samples? (default = False)')
parser.add_option('--hotChain', dest='hotChain', action='store_true', default=False,
                   help='Given a PTMCMC sampler, do you want to use a T=1e80 hot chain? (default = False)')
parser.add_option('--softParam', dest='softParam', action='store', type=float, default=1.0,
                   help='Artifical temperature by which to soften likelihood (default = 1.0)')
parser.add_option('--shortFileTag', dest='shortFileTag', action='store', type=str, default=None,
                  help='Provide a shorter file tag for MultiNest runs? (default = None)')
parser.add_option('--TmaxType', dest='TmaxType', action='store', type=str, default='pta',
                   help='Which type of Tmax to use to set frequencies: pta (longest baseline over the array), or pulsar (longest pulsar in array) (default = pta)')
parser.add_option('--incGWB', dest='incGWB', action='store_true', default=False,
                  help='Do you want to search for a GWB? (default = False)')
parser.add_option('--gwbSpecModel', dest='gwbSpecModel', action='store', type=str, default='powerlaw',
                  help='What kind of spectral model do you want for the GWB?: powerlaw, spectrum, turnover, gpEnvInterp (default = powerlaw)')
parser.add_option('--fix_gwbTurnKappa', dest='fix_gwbTurnKappa', action='store', type=float, default=None,
                  help='Do you want to fix kappa in the turnover GWB spectral model to a particular value? (stars=10/3, gas=7/3) (default = \'None\')')
parser.add_option('--gwb_fb2env', dest='gwb_fb2env', action='store', type=str, default=None,
                  help='In GWB turnover model, do you want to map the environmental parameters directly? (stars, gas, etc.) (default = \'None\')')
parser.add_option('--gpPickle', dest='gpPickle', action='store', type=str, default='/Users/staylor/Research/PapersInProgress/NPDE/gp4ptas/ecc_gp.pkl',
                  help='Provide the pickle file storing the list of GP objects for when gwbSpecModel is gpEnvInterp or when gwbPrior is gaussProc. Must contain either ecc, stars, or acc in an underscore delimited filename (default = /Users/staylor/Research/PapersInProgress/NPDE/gp4ptas/stars_gaussproc.pkl)')
parser.add_option('--gpKernel', dest='gpKernel', action='store', type=str, default='expsquared',
                  help='What type of kernel to use in the GP emulator? (default = expsquared)')
parser.add_option('--userOrf', dest='userOrf', action='store', type=str, default=None,
                  help='Provide your own ORF in a numpy array of shape (npsr,npsr) or (nfreqs,npsr,npsr) (default = None)')
parser.add_option('--pshift', dest='pshift', action='store_true', default=False,
                  help='Do you want to include random phase shifts in the Fourier design matrices? (default = False)')
parser.add_option('--incCosVar', dest='incCosVar', action='store_true', default=False,
                  help='Do you want to include GP interpolation uncertainties or cosmic variance in your gpEnvInterp model? (default = False)')
parser.add_option('--incCorr', dest='incCorr', action='store_true', default=False,
                  help='Do you want to include cross-correlations in the GWB model? (default = False)')
parser.add_option('--gwbTypeCorr', dest='gwbTypeCorr', action='store', type=str, default='spharmAnis',
                  help='What type of correlated GW signal do you want to model?: custom, spharmAnis, dipoleOrf, modelIndep, pointSrc, clock, gwDisk, psrlocsVary (default = spharmAnis)')
parser.add_option('--gwbModelSelect', dest='gwbModelSelect', action='store_true', default=False,
                  help='Perform model selection between correlated and uncorrelated GWB model? (default = False)')
parser.add_option('--gwbCorrModWgt', dest='gwbCorrModWgt', action='store', type=float, default=1.0,
                  help='Provide estimate of Bayes factor in favor of correlated model to get better model mixing (default = 1.0)')
parser.add_option('--corrJacobian', dest='corrJacobian', action='store', type=str, default='simple',
                  help='What type of Jacobian do you want for the modelIndep ORF element search: simple, full (default = simple)')
parser.add_option('--psrlocsPrior', dest='psrlocsPrior', action='store', type=str, default='normal',
                  help='What type of prior do you want on the pulsar locations in the psrlocsVary correlation model: normal, uniform (default = normal)')
parser.add_option('--fixPointSrcPhi', dest='fixPointSrcPhi', action='store', type=float, default=None,
                  help='Fix the azimuthal sky-location (phi) of a stochastic point-source to a particular value (default = \'None\')')
parser.add_option('--fixPointSrcTheta', dest='fixPointSrcTheta', action='store', type=float, default=None,
                  help='Fix the polar sky-location (theta) of a stochastic point-source to a particular value (default = \'None\')')
parser.add_option('--redSpecModel', dest='redSpecModel', action='store', type=str, default='powerlaw',
                  help='What kind of spectral model do you want for red timing-noise?: powerlaw, spectrum (default = powerlaw)')
parser.add_option('--dmSpecModel', dest='dmSpecModel', action='store', type=str, default='powerlaw',
                  help='What kind of spectral model do you want for DM variations?: powerlaw, spectrum (default = powerlaw)')
parser.add_option('--nmodes_dm', dest='nmodes_dm', action='store', type=int, default=None,
                   help='Number of DM-variation modes in low-rank time-frequency approximation')
parser.add_option('--incEph', dest='incEph', action='store_true', default=False,
                  help='Do you want to search for solar system ephemeris errors? (default = False)')
parser.add_option('--ephSpecModel', dest='ephSpecModel', action='store', type=str, default='powerlaw',
                  help='What kind of spectral model do you want for the solar system ephemeris errors?: powerlaw, spectrum (default = powerlaw)')
parser.add_option('--nmodes_eph', dest='nmodes_eph', action='store', type=int, default=None,
                   help='Number of ephemeris modes in low-rank time-frequency approximation')
parser.add_option('--ephFreqs', dest='ephFreqs', action='store', type=str, default=None,
                  help='Provide the ephemeris-error model frequencies as a comma delimited string (default = None)')
parser.add_option('--incClk', dest='incClk', action='store_true', default=False,
                  help='Do you want to search for clock errors? (default = False)')
parser.add_option('--clkDesign', dest='clkDesign', action='store_true', default=False,
                  help='Do you want to model clock errors in a separate basis from the red-noise and GWB? (default = False)')
parser.add_option('--clkSpecModel', dest='clkSpecModel', action='store', type=str, default='powerlaw',
                  help='What kind of spectral model do you want for the clock errors?: powerlaw, spectrum (default = powerlaw)')
parser.add_option('--incBand', dest='incBand', action='store_true', default=False,
                  help='Do you want to search for radio-band-dependent red-noise? (default = False)')
parser.add_option('--bandSpecModel', dest='bandSpecModel', action='store', type=str, default='powerlaw',
                  help='What kind of spectral model do you want for the radio-band-dependent red-noise?: powerlaw, spectrum (default = powerlaw)')
parser.add_option('--nmodes_band', dest='nmodes_band', action='store', type=int, default=None,
                   help='Number of band modes in low-rank time-frequency approximation')
parser.add_option('--bands', dest='bands', action='store', type=str, default=None,
                  help='Provide the radio-bands for band-noise as a comma delimited string (e.g. [0.0,1.0,2.0,3.0] gives 3 bands) (default = None)')
parser.add_option('--incCm', dest='incCm', action='store_true', default=False,
                  help='Do you want to search for a common uncorrelated noise process? (default = False)')
parser.add_option('--cmSpecModel', dest='cmSpecModel', action='store', type=str, default='powerlaw',
                  help='What kind of spectral model do you want for the common noise process?: powerlaw, spectrum (default = powerlaw)')
parser.add_option('--dirExt', dest='dirExt', action='store', type=str, default='./chains_nanoAnalysis/',
                  help='What master directory name do you want to put this run into? (default = ./chains_nanoAnalysis/)')
parser.add_option('--nwins', dest='nwins', action='store', type=int, default=1,
                   help='Number windows to split the band into (useful for evolving anisotropy searches (default = 1 windows)')
parser.add_option('--lmax', dest='LMAX', action='store', type=int, default=0,
                   help='Maximum multipole in anisotropic search (default = 0, i.e. isotropic-search)')
parser.add_option('--noPhysPrior', dest='noPhysPrior', action='store_true', default=False,
                   help='Switch off test for physicality of anisotropic coefficient sampling (default = False)')
parser.add_option('--use_gpu', dest='use_gpu', action='store_true', default=False,
                  help='Do you want to use the GPU for accelerated linear algebra? (default = False)')
parser.add_option('--sparse_cholesky', dest='sparse_cholesky', action='store_true', default=False,
                  help='Do you want to use a sparse cholesky solver? (default = False)')
parser.add_option('--fix_slope', dest='fix_slope', action='store_true', default=False,
                  help='Do you want to fix the slope of the GWB spectrum? (default = False)')
parser.add_option('--gwbAmpRange', dest='gwbAmpRange', action='store', type=str, default=None,
                  help='Provide a lower and upper log_10(Agwb) range as a comma delimited string (default = None)')
parser.add_option('--gwbStarsRange', dest='gwbStarsRange', action='store', type=str, default='1.0,4.0',
                  help='Provide a lower and upper log_10(rho_stars) range as a comma delimited string (default = None)')
parser.add_option('--gwbEccRange', dest='gwbEccRange', action='store', type=str, default='0.0,0.9',
                  help='Provide a lower and upper e0 range as a comma delimited string (default = None)')
parser.add_option('--gwbPrior', dest='gwbPrior', action='store', type=str, default='uniform',
                   help='Do you want to use a uniform prior on log_10(amplitude) for detection [loguniform], on amplitudes themselves for limits [uniform], an astrophysical prior (only when the amplitude is Agwb: for powerlaw, turnover, gpEnvInterp models) [sesana, mcwilliams], or a gaussian process prior [gaussProc] (default=\'uniform\')?')
parser.add_option('--gwbHyperPrior', dest='gwbHyperPrior', action='store', type=str, default='uniform',
                   help='When gwbPrior=gaussProc, do you want to use a uniform prior on log_10(Agwb) for detection [loguniform], on Agwb itself for limits [uniform], or an astrophysical prior [sesana, mcwilliams] (default=\'uniform\')?')
parser.add_option('--redPrior', dest='redPrior', action='store', type=str, default='uniform',
                   help='Do you want to use a uniform prior on log_10(Ared) for detection [loguniform], on Ared itself for limits [uniform] (default=\'uniform\')?')
parser.add_option('--dmPrior', dest='dmPrior', action='store', type=str, default='uniform',
                   help='Do you want to use a uniform prior on log_10(Adm) for detection [loguniform], on Adm itself for limits [uniform] (default=\'uniform\')?')
parser.add_option('--ephPrior', dest='ephPrior', action='store', type=str, default='uniform',
                   help='Do you want to use a uniform prior on log_10(Aephx,y,z) for detection [loguniform], on Aephx,y,z themselves for limits [uniform] (default=\'uniform\')?')
parser.add_option('--clkPrior', dest='clkPrior', action='store', type=str, default='uniform',
                   help='Do you want to use a uniform prior on log_10(Aclk) for detection [loguniform], on Aclk itself for limits [uniform] (default=\'uniform\')?')
parser.add_option('--bandPrior', dest='bandPrior', action='store', type=str, default='uniform',
                   help='Do you want to use a uniform prior on log_10(Aband) for detection [loguniform], on Aband itself for limits [uniform] (default=\'uniform\')?')
parser.add_option('--cmPrior', dest='cmPrior', action='store', type=str, default='uniform',
                   help='Do you want to use a uniform prior on log_10(Acm) for detection [loguniform], on Acm itself for limits [uniform] (default=\'uniform\')?')
parser.add_option('--anis_modefile', dest='anis_modefile', action='store', type=str, default = None,
                   help='Do you want to provide an anisotropy modefile to split band into frequency windows?')
parser.add_option('--noEcorr', dest='noEcorr', action='store_true', default=False,
                  help='Do you want to ignore correlated white noise terms in noise matrix? (default = False)')
parser.add_option('--fixRed', dest='fixRed', action='store_true', default=False,
                  help='Do you want to perform a fixed power-law red-noise analysis? (default = False)')
parser.add_option('--fixDM', dest='fixDM', action='store_true', default=False,
                  help='Do you want to perform a fixed power-law DM-variations analysis? (default = False)')
parser.add_option('--psrStartIndex', dest='psrStartIndex', action='store', type=int, default=0,
                  help='From your pulsar list, which pulsar index do you want to start with? (default = 0)')
parser.add_option('--psrEndIndex', dest='psrEndIndex', action='store', type=int, default=18,
                  help='From your pulsar list, which pulsar index do you want to end with? (default = 18)')
parser.add_option('--psrIndices', dest='psrIndices', action='store', type=str, default=None,
                  help='Provide a sequence of indices from your pulsar list as a comma delimited string (default = None)')
parser.add_option('--det_signal', dest='det_signal', action='store_true', default=False,
                  help='Do you want to search for a deterministic GW signal? (default = False)')
parser.add_option('--bwm_search', dest='bwm_search', action='store_true', default=False,
                  help='Do you want to search for GW burst with memory (BWM)? (default = False)')
parser.add_option('--bwm_antenna', dest='bwm_antenna', action='store', type=str, default='quad',
                  help='What kind of antenna pattern do you want to use for a BWM? (default = quad)')
parser.add_option('--bwm_model_select', dest='bwm_model_select', action='store_true', default=False,
                  help='Do you want to compute the Bayes Factor for BWM+noise versus noise-only? (default = False)')
parser.add_option('--cgw_search', dest='cgw_search', action='store_true', default=False,
                  help='Do you want to search for a single continuous GW signal? (default = False)')
parser.add_option('--cgwFreqRange', dest='cgwFreqRange', action='store', type=str, default=None,
                  help='Provide a lower and upper log_10(f_orb) range as a comma delimited string (default = None)')
parser.add_option('--cgwModelSelect', dest='cgwModelSelect', action='store_true', default=False,
                  help='Do you want to compute the Bayes factor for CGW+noise versus noise-only? (default = False)')
parser.add_option('--ecc_search', dest='ecc_search', action='store_true', default=False,
                  help='Do you want to search for an eccentric binary? (default = False)')
parser.add_option('--epochTOAs', dest='epochTOAs', action='store_true', default=False,
                  help='Do you want to compute CGW waveforms with the averaged TOAs? (default = False)')
parser.add_option('--psrTerm', dest='psrTerm', action='store_true', default=False,
                  help='Do you want to include the pulsar term in the continuous wave search? (default = False)')
parser.add_option('--periEv', dest='periEv', action='store_true', default=False,
                  help='Do you want to model the binary periapsis evolution? (default = False)')
parser.add_option('--cgwPrior', dest='cgwPrior', action='store', type=str, default='uniform',
                  help='By default this puts a [uniform] prior on the strain amplitude, but can also choose [loguniform] on strain amplitude, or [mdloguniform] which puts separate loguniform priors on mass and distance (default = \'uniform\')')
parser.add_option('--fixcgwFreq', dest='fixcgwFreq', action='store', type=float, default=None,
                  help='Fix the cgw orbital frequency to a particular log10 value (default = \'None\')')
parser.add_option('--fixcgwEcc', dest='fixcgwEcc', action='store', type=float, default=None,
                  help='Fix the cgw eccentricity to a particular value (default = \'None\')')
parser.add_option('--fixcgwPhi', dest='fixcgwPhi', action='store', type=float, default=None,
                  help='Fix the cgw azimuthal sky-location (phi) to a particular value (default = \'None\')')
parser.add_option('--fixcgwTheta', dest='fixcgwTheta', action='store', type=float, default=None,
                  help='Fix the cgw polar sky-location (theta) to a particular value (default = \'None\')')
parser.add_option('--noEccEvolve', dest='noEccEvolve', action='store_true', default=False,
                  help='Do not allow eccentricity to evolve between pulsar- and Earth-term (default = \'False\')')
parser.add_option('--eph_quadratic', dest='eph_quadratic', action='store_true', default=False,
                  help='Do you want to include a deterministic quadratic in the ephemeris model? (default = False)')
parser.add_option('--eph_planetdelta', dest='eph_planetdelta', action='store_true', default=False,
                  help='Do you want to include a deterministic planet-mass perturbation in the ephemeris model? (default = False)')
parser.add_option('--eph_planetnums', dest='eph_planetnums', action='store', type=str, default=None,
                  help='Which planets to include in pertubed-mass model [Mercury=1, Venus=2, etc.] (default = None)')
parser.add_option('--incGWline', dest='incGWline', action='store_true', default=False,
                  help='Do you want to include a single-frequency line in the GW spectrum? (default = False)')
parser.add_option('--gwlinePrior', dest='gwlinePrior', action='store', type=str, default='uniform',
                   help='Do you want to use a uniform prior on log_10(rho_line) for detection [loguniform], on rho_line itself for limits [uniform] (default=\'uniform\')?')
parser.add_option('--constLike', dest='constLike', action='store_true', default=False,
                  help='Do you want to set the likelihood to a constant and thus sample from the prior? (default = False)')

(args, x) = parser.parse_args()

if args.jsonModel is not None:
    
    with open(args.jsonModel) as json_file:
        json_data = json.load(json_file)
        json_file.close()

    args.from_h5 = json_data['from_h5']
    args.psrlist = json_data['psrlist']
    args.nmodes = json_data['nmodes']
    args.cadence = json_data['cadence']
    args.incDM = json_data['incDM']
    args.sampler = json_data['sampler']
    args.writeHotChains = json_data['writeHotChains']
    args.resume = json_data['resume']
    args.incGWB = json_data['incGWB']
    args.gwbSpecModel = json_data['gwbSpecModel']
    args.gpPickle = json_data['gpPickle']
    args.userOrf = json_data['userOrf']
    args.pshift = json_data['pshift']
    args.incCosVar = json_data['incCosVar']
    args.incCorr = json_data['incCorr']
    args.gwbTypeCorr = json_data['gwbTypeCorr']
    args.redSpecModel = json_data['redSpecModel']
    args.dmSpecModel = json_data['dmSpecModel']
    args.dirExt = json_data['dirExt']
    args.nwins = json_data['nwins']
    args.LMAX = json_data['LMAX']
    args.noPhysPrior = json_data['noPhysPrior']
    args.use_gpu = json_data['use_gpu']
    args.fix_slope = json_data['fixSlope']
    args.gwbPrior = json_data['gwbPrior']
    args.gwbHyperPrior = json_data['gwbHyperPrior']
    args.redPrior = json_data['redPrior']
    args.dmPrior = json_data['dmPrior']
    args.ephPrior = json_data['ephPrior']
    args.clkPrior = json_data['clkPrior']
    args.cmPrior = json_data['cmPrior']
    args.anis_modefile = json_data['anis_modefile']
    args.noEcorr = json_data['noEcorr']
    args.fixRed = json_data['fixRed']
    args.fixDM = json_data['fixDM']
    args.incEph = json_data['incEph']
    args.ephSpecModel = json_data['ephSpecModel']
    args.incClk = json_data['incClk']
    args.clkSpecModel = json_data['clkSpecModel']
    args.incCm = json_data['incCm']
    args.cmSpecModel = json_data['cmSpecModel']
    args.psrStartIndex = json_data['psrStartIndex']
    args.psrEndIndex = json_data['psrEndIndex']
    args.psrIndices = json_data['psrIndices']
    args.det_signal = json_data['det_signal']
    args.bwm_search = json_data['bwm_search']
    args.bwm_antenna = json_data['bwm_antenna']
    args.bwm_model_select = json_data['bwm_model_select']
    args.cgw_search = json_data['cgw_search']
    args.ecc_search = json_data['ecc_search']
    args.epochTOAs = json_data['epochTOAs']
    args.psrTerm = json_data['psrTerm']
    args.periEv = json_data['periEv']
    args.cgwPrior = json_data['cgwPrior']
    args.fixcgwFreq = json_data['fixcgwFreq']
    args.fixcgwEcc = json_data['fixcgwEcc']
    args.fixcgwPhi = json_data['fixcgwPhi']
    args.fixcgwTheta = json_data['fixcgwTheta']
    args.noEccEvolve = json_data['noEccEvolve']
    args.incGWline = json_data['incGWline']
    args.gwlinePrior = json_data['gwlinePrior']
    args.constLike = json_data['constLike']


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

if args.sparse_cholesky:
    import scipy.sparse as sps
    import sksparse.cholmod as sks

if args.sampler == 'mnest':
    import pymultinest
elif args.sampler == 'pchord':
    import pypolychord
elif args.sampler == 'ptmcmc':
    import PTMCMCSampler
    from PTMCMCSampler import PTMCMCSampler as ptmcmc

#########################################################################
# PASSING THROUGH TEMPO2 VIA libstempo
#########################################################################

if args.psrlist is not None:
    # name, hdf5-path, par-path, tim-path
    psr_pathinfo = np.genfromtxt(args.psrlist, dtype=str, skip_header=2)

if args.from_h5:

    tmp_psr = []
    
    if args.psrIndices is not None:
        psr_inds = [int(item) for item in args.psrIndices.split(',')]
        for ii,tmp_name in zip(psr_inds,psr_pathinfo[psr_inds,0]):
            tmp_psr.append(h5.File(psr_pathinfo[ii,1], 'r')[tmp_name])

    else:
    
        for ii,tmp_name in enumerate(psr_pathinfo[args.psrStartIndex:args.psrEndIndex,0],
                                     start=args.psrStartIndex):
            tmp_psr.append(h5.File(psr_pathinfo[ii,1], 'r')[tmp_name])

    psr = [NX01_psr.PsrObjFromH5(p) for p in tmp_psr]
    
else:
    
    print 'Are you sure you do not want to use hdf5 files (recommended)?'
    
    t2psr=[]
    if args.parfile is not None and args.timfile is not None:
        
        t2psr.append( T2.tempopulsar(parfile=args.parfile,
                                     timfile=args.timfile,
                                     maxobs=int(4e4)) )
        #t2psr[0].fit(iters=3)
        if np.any(np.isfinite(t2psr[0].residuals())==False)==True:
            t2psr[0] = T2.tempopulsar(parfile=args.parfile,
                                      timfile=args.timfile,
                                      maxobs=int(4e4))
            
    else:
        
        for ii in range(args.psrStartIndex,args.psrEndIndex):
            t2psr.append( T2.tempopulsar( parfile=psr_pathinfo[ii,2],
                                          timfile=psr_pathinfo[ii,3],
                                          maxobs=int(4e4) ) )
            t2psr[ii].fit(iters=3)
            if np.any(np.isfinite(t2psr.residuals())==False)==True:
                t2psr = T2.tempopulsar( parfile=psr_pathinfo[ii,2],
                                        timfile=psr_pathinfo[ii,3],
                                        maxobs=int(4e4) )

    psr = [NX01_psr.PsrObj(p) for p in t2psr]


# Grab all the pulsar quantities
if args.psrlist is not None:
    if args.varyWhite:
        [p.grab_all_vars(rescale=False, sysflag_target=args.sysflag_target) for p in psr]
    elif not args.varyWhite:
        [p.grab_all_vars(rescale=True, sysflag_target=args.sysflag_target) for p in psr]
elif args.parfile is not None and args.timfile is not None:
    [p.grab_all_vars(jitterbin=args.jitterbin, makeGmat=False,
                     fastDesign=True, planetssb=args.grab_planets) for p in psr]

# Now, grab the positions and compute the ORF basis functions
psr_positions = [np.array([psr[ii].psr_locs[0],
                           np.pi/2. - psr[ii].psr_locs[1]])
                           for ii in range(len(psr))]
positions = np.array(psr_positions).copy()

num_corr_params = 0
evol_corr_tag = ''
if args.incGWB and args.incCorr:
    
    if args.gwbTypeCorr == 'modelIndep':

        gwfreqs_per_win = int(1.*args.nmodes/(1.*args.nwins)) 
        corr_modefreqs = np.arange(1,args.nmodes+1)
        corr_modefreqs = np.reshape(corr_modefreqs,
                                    (args.nwins,gwfreqs_per_win))

        tmp_nwins = args.nwins

        num_corr_params = tmp_nwins*int(len(psr)*(len(psr)-1)/2)

        if tmp_nwins>1:
            evol_corr_tag = '_evanis'
        else:
            evol_corr_tag = ''

    elif args.gwbTypeCorr == 'pointSrc':

        gwfreqs_per_win = int(1.*args.nmodes/(1.*args.nwins)) 
        corr_modefreqs = np.arange(1,args.nmodes+1)
        corr_modefreqs = np.reshape(corr_modefreqs,
                                    (args.nwins,gwfreqs_per_win))

        tmp_nwins = args.nwins

        if args.fixPointSrcPhi is not None and args.fixPointSrcTheta is not None:
            num_corr_params = 0
        else:
            num_corr_params = 2*tmp_nwins

        if tmp_nwins>1:
            evol_corr_tag = '_evanis'
        else:
            evol_corr_tag = ''
        
    elif args.gwbTypeCorr == 'spharmAnis':
        
        # Computing all the correlation basis-functions for the array.
        CorrCoeff = np.array(anis.CorrBasis(positions,args.LMAX))
        # Computing the values of the spherical-harmonics up to order
        # LMAX on a pre-specified grid  
        harm_sky_vals = utils.SetupPriorSkyGrid(args.LMAX)              
                                                            
        if args.anis_modefile is None:
        
            # getting the number of GW frequencies per window
            gwfreqs_per_win = int(1.*args.nmodes/(1.*args.nwins)) 
            corr_modefreqs = np.arange(1,args.nmodes+1)
            corr_modefreqs = np.reshape(corr_modefreqs,
                                        (args.nwins,gwfreqs_per_win))

            tmp_nwins = args.nwins
    
        else:

            tmp_modefreqs = np.loadtxt(args.anis_modefile, skiprows=2)
            tmp_nwins = tmp_modefreqs.shape[0]
            corr_modefreqs = []
    
            for ii in range(tmp_nwins):
                corr_modefreqs.append(np.arange(tmp_modefreqs[ii,0],
                                                tmp_modefreqs[ii,1]+1))

        num_corr_params = tmp_nwins*(((args.LMAX+1)**2)-1)

        # Create a tag for evolving anisotropy searches
        if (args.LMAX!=0) and (tmp_nwins > 1):
            evol_corr_tag = '_evanis'
        else:
            evol_corr_tag = ''

    elif args.gwbTypeCorr == 'dipoleOrf':

        monoOrf = 2.0*np.sqrt(np.pi)*anis.CorrBasis(positions,0)[0]

        gwfreqs_per_win = int(1.*args.nmodes/(1.*args.nwins)) 
        corr_modefreqs = np.arange(1,args.nmodes+1)
        corr_modefreqs = np.reshape(corr_modefreqs,
                                    (args.nwins,gwfreqs_per_win))

        tmp_nwins = args.nwins

        num_corr_params = 3*tmp_nwins

        if tmp_nwins>1:
            evol_corr_tag = '_evanis'
        else:
            evol_corr_tag = ''

    elif args.gwbTypeCorr == 'gwDisk':

        tmp_nwins = args.nwins
        
        try:
            import healpy as hp
            import AnisCoefficients_pix as pixAnis
            num_corr_params = 4*tmp_nwins

            npsrs = len(positions)
            pphi = positions[:,0]
            ptheta = positions[:,1]
    
            # Create the pixels
            nside=32
            npixels = hp.nside2npix(32)
            pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)
            gwtheta = pixels[0]
            gwphi = pixels[1]

            # Create the signal response matrix
            F_e = pixAnis.signalResponse_fast(ptheta, pphi, gwtheta, gwphi)
            
        except ImportError:
            print "ERROR: Could not import healpy!"
            print "WARNING: Defaulting to H&D search..."

            hp = None
            monoOrf = 2.0*np.sqrt(np.pi)*anis.CorrBasis(positions,0)[0]
            num_corr_params = 0

        gwfreqs_per_win = int(1.*args.nmodes/(1.*args.nwins)) 
        corr_modefreqs = np.arange(1,args.nmodes+1)
        corr_modefreqs = np.reshape(corr_modefreqs,
                                    (args.nwins,gwfreqs_per_win))

        if tmp_nwins>1:
            evol_corr_tag = '_evanis'
        else:
            evol_corr_tag = ''

    elif args.gwbTypeCorr == 'custom':

        if args.userOrf is None:
            print "WARNING: You requested a custom ORF but" \
            " didn't give me an array file!"
            print "WARNING: Proceeding with Hellings and Downs..."

            customOrf = 2.0*np.sqrt(np.pi)*anis.CorrBasis(positions,0)[0]
            
        elif args.userOrf is not None:

            if args.userOrf.split('.')[-1] != 'npy':
                print "You are supplying custom pulsar positions, " \
                "possibly scrambled!"
                custom_positions = np.genfromtxt(args.userOrf,dtype=str,comments='#')
                custom_positions = np.double(custom_positions[:,1:])

                if len(custom_positions)!=len(psr):
                    print "ERROR: Number of custom pulsar positions does not match " \
                      "the number of hdf5 files you gave me!"
                    print "ERROR: Proceeding with Hellings and Downs instead!"
                    customOrf = 2.0*np.sqrt(np.pi)*anis.CorrBasis(positions,0)[0]
                elif len(custom_positions)==len(psr):
                    customOrf = 2.0*np.sqrt(np.pi)*anis.CorrBasis(custom_positions,0)[0]
                    
            elif args.userOrf.split('.')[-1] == 'npy':
                loadOrf = np.load(args.userOrf)
                if np.atleast_3d(loadOrf.T).shape[-1]>1:
                    print "You have given me ORFs for all frequencies!"
                else:
                    print "You have given me a broadband ORF!"

                if (np.atleast_3d(loadOrf.T).shape[0]==len(psr) and
                    np.atleast_3d(loadOrf.T).shape[1]==len(psr)):
                    print "Dimensions match number of pulsars...OK!"
                    customOrf = loadOrf
                else:
                    print "ERROR: Dimensions don't match number of pulsars!"
                    print "ERROR: Proceeding with Hellings and Downs instead!"
                    customOrf = 2.0*np.sqrt(np.pi)*anis.CorrBasis(positions,0)[0]

        num_corr_params = 0

    elif args.gwbTypeCorr == 'psrlocsVary':

        gwfreqs_per_win = int(1.*args.nmodes/(1.*args.nwins)) 
        corr_modefreqs = np.arange(1,args.nmodes+1)
        corr_modefreqs = np.reshape(corr_modefreqs,
                                    (args.nwins,gwfreqs_per_win))

        tmp_nwins = args.nwins

        num_corr_params = 2*len(psr)*tmp_nwins

        if tmp_nwins>1:
            evol_corr_tag = '_evanis'
        else:
            evol_corr_tag = ''

    elif args.gwbTypeCorr == 'clock':

        gwfreqs_per_win = args.nmodes
        corr_modefreqs = np.arange(1,args.nmodes+1)
        corr_modefreqs = np.reshape(corr_modefreqs,
                                    (args.nwins,gwfreqs_per_win))

        tmp_nwins = args.nwins
        num_corr_params = 0
              
#############################################################################
# GETTING MAXIMUM TIME, COMPUTING FOURIER DESIGN MATRICES, AND GETTING MODES 
#############################################################################

if args.TmaxType == 'pta':
    Tmax = np.max([p.toas.max() for p in psr]) - \
      np.min([p.toas.min() for p in psr])
else:
    Tmax = np.max([p.toas.max() - p.toas.min() for p in psr])

### Define number of red noise modes and set sampling frequencies
if args.nmodes is not None:
    nmodes_red = args.nmodes
elif args.nmodes is None and args.cadence is not None:
    nmodes_red = int(round(0.5*Tmax/args.cadence))
fqs_red = np.linspace(1/Tmax, nmodes_red/Tmax, nmodes_red)

### Define number of DM-variation modes and set sampling frequencies
nmodes_dm = args.nmodes_dm
if args.incDM:
    if args.nmodes_dm is not None:
        nmodes_dm = args.nmodes_dm
    else:
        nmodes_dm = nmodes_red
    fqs_dm = np.linspace(1/Tmax, nmodes_dm/Tmax, nmodes_dm)

### Define number of ephemeris-error modes and set sampling frequencies
nmodes_eph = args.nmodes_eph
if args.incEph:
    if args.nmodes_eph is not None:
        nmodes_eph = args.nmodes_eph
    else:
        nmodes_eph = nmodes_red
    ##
    if args.ephFreqs is None:
        fqs_eph = np.linspace(1/Tmax, nmodes_eph/Tmax, nmodes_eph)
    elif args.ephFreqs is not None:
        fqs_eph = np.array([float(item) for item in args.ephFreqs.split(',')])

### Define number of band-noise modes and set sampling frequencies
nmodes_band = args.nmodes_band
if args.incBand:
    if args.nmodes_band is not None:
        nmodes_band = args.nmodes_band
    else:
        nmodes_band = nmodes_red
    fqs_band = np.linspace(1/Tmax, nmodes_band/Tmax, nmodes_band)

    if args.bands is None:
        bands = np.array([0.0, 1.0, 2.0, 3.0])
    elif args.bands is not None:
        bands = np.array([float(item) for item in args.bands.split(',')])

### Make the basis matrices for all rank-reduced processes in model
[p.makeTe(nmodes_red, Tmax, makeDM=args.incDM, nmodes_dm=nmodes_dm,
          makeEph=args.incEph, nmodes_eph=nmodes_eph, ephFreqs=args.ephFreqs,
          makeClk=args.incClk, clkDesign=args.clkDesign,
          makeBand=args.incBand, bands=args.bands, phaseshift=args.pshift) for p in psr]

if args.det_signal:
    # find reference time for all pulsars
    tt = [np.min(p.toas) for p in psr]
    tref = np.min(tt)

############################################
# READ IN PICKLED GAUSSIAN PROCESS INSTANCE
############################################

gp = []
if args.incGWB:
    if args.gwbPrior == 'gaussProc' or \
      args.gwbSpecModel == 'gpEnvInterp':
        
        import george
        import NX01_gaussproc
        from NX01_gaussproc import gaussproc
        
        gppkl = pickle.load( open( args.gpPickle, "rb" ) )
        # Set george kernel parameters to previously-trained MAP
        # Compute factorisation of kernel based on sampled points
        for ii in range(len(gppkl)):
            gp_kparams = np.exp(gppkl[ii].kernel_map)
            if args.gpKernel == 'expsquared':
                gp.append( george.GP( gp_kparams[0] * \
                                      george.kernels.ExpSquaredKernel(gp_kparams[1:],ndim=len(gp_kparams[1:])) ) )
            elif args.gpKernel == 'matern32':
                gp.append( george.GP( gp_kparams[0] * \
                                      george.kernels.Matern32Kernel(gp_kparams[1:],ndim=len(gp_kparams[1:])) ) )
            elif args.gpKernel == 'matern52':
                gp.append( george.GP( gp_kparams[0] * \
                                      george.kernels.Matern52Kernel(gp_kparams[1:],ndim=len(gp_kparams[1:])) ) )
            gp[ii].compute(gppkl[ii].x, gppkl[ii].yerr)
            gwb_popparam_ndims = len(gp_kparams[1:])

        gwb_popparam = args.gpPickle.split('/')[-1].split('_')
        for word in gwb_popparam:
            if word in ['stars','ecc','gas','starsecc']:
                gwb_popparam = word
                break

###############################################################
# CONSTRUCT DIRECT MAPPING FROM FBEND TO ENVIRONMENT PARAMETER
###############################################################

if args.incGWB and args.gwbSpecModel=='turnover' and args.gwb_fb2env is not None:
    
    class binary_env:
        def __init__(self, mechanism=None):
            self.mechanism = mechanism
            if self.mechanism == 'stars':
                self.kappa = 10./3.
                self.interpPath = './data/nano9yr_turnover_mappings/stars_mapping.txt'
            elif self.mechanism == 'gas':
                self.kappa = 7./3.
                self.interpPath = './data/nano9yr_turnover_mappings/gas_mapping.txt'
            # setup fturn vs envParam interpolant
            try:
                self.fil = np.loadtxt(self.interpPath)
            except:
                self.fil = None
                print 'Could not find file!'
    
            if self.fil is not None:
                self.interpolant = interp1d(self.fil[:,0], self.fil[:,1])
            else:
                self.interpolant = None
        def fb_from_env(self, envParam=None):
            fturn = self.interpolant(envParam)
            return fturn / (0.75*self.kappa - 1.0)**(1./self.kappa)

    fb2env = binary_env(mechanism=args.gwb_fb2env)

#######################################
# PRE-COMPUTING WHITE NOISE PROPERTIES 
#######################################

if not args.varyWhite:
    
    loglike1 = 0
    logdet_N = []
    TtNT = []
    d = []
    Jamp = []
    for ii,p in enumerate(psr):

        # compute ( T.T * N^-1 * T )
        # & log determinant of N
        new_err = (p.toaerrs).copy()
        if not args.noEcorr:
        
            if p.ecorrs is not None and len(p.ecorrs)>0:
    
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

if args.gwbAmpRange is not None:
    amp_range = np.array([float(item) for item \
                          in args.gwbAmpRange.split(',')])
if args.gwbStarsRange is not None:
    stars_range = np.array([float(item) for item \
                          in args.gwbStarsRange.split(',')])
if args.gwbEccRange is not None:
    ecc_range = np.array([float(item) for item \
                          in args.gwbEccRange.split(',')])
if args.cgwFreqRange is not None:
    cgw_orbfreq_range = np.array([float(item) for item \
                                  in args.cgwFreqRange.split(',')])
if args.eph_planetdelta:
    if args.eph_planetnums is not None:
        planet_tags = np.array([int(item) for item \
                                in args.eph_planetnums.split(',')])
        num_planets = len(planet_tags)
    else:
        planet_tags = np.arange(1,10)
        num_planets = 9
    

pmin = np.array([])
if not args.fixRed:
    if args.redSpecModel == 'powerlaw':
        pmin = np.append(pmin,-20.0*np.ones(len(psr)))
        pmin = np.append(pmin,0.0*np.ones(len(psr)))
    elif args.redSpecModel == 'spectrum':
        pmin = np.append(pmin,-8.0*np.ones(len(psr)*nmodes_red))
if args.incDM and not args.fixDM:
    if args.dmSpecModel == 'powerlaw':
        pmin = np.append(pmin,-20.0*np.ones(len(psr)))
        pmin = np.append(pmin,0.0*np.ones(len(psr)))
    elif args.dmSpecModel == 'spectrum':
        pmin = np.append(pmin,-8.0*np.ones(len(psr)*nmodes_dm))
if args.varyWhite:
    for ii,p in enumerate(psr):
        systems = p.sysflagdict[args.sysflag_target]
        pmin = np.append(pmin,0.001*np.ones(len(systems)))
        pmin = np.append(pmin,-10.0*np.ones(len(systems)))
        if 'nano-f' in p.sysflagdict.keys() and len(p.sysflagdict['nano-f'].keys())>0:
            pmin = np.append(pmin, -10.0*np.ones(len(p.sysflagdict['nano-f'].keys())))
if args.incBand:
    if args.bandSpecModel == 'powerlaw':
        pmin = np.append(pmin,-20.0*np.ones(len(bands)-1))
        pmin = np.append(pmin,0.0*np.ones(len(bands)-1))
    elif args.bandSpecModel == 'spectrum':
        pmin = np.append(pmin,-30.0*np.ones((len(bands)-1)*nmodes_band))
if args.incClk:
    if args.clkSpecModel == 'powerlaw':
        pmin = np.append(pmin,-20.0)
        pmin = np.append(pmin,0.0)
    elif args.clkSpecModel == 'spectrum':
        pmin = np.append(pmin,-8.0*np.ones(nmodes_red))
if args.incCm:
    if args.cmSpecModel == 'powerlaw':
        pmin = np.append(pmin,-18.0)
        pmin = np.append(pmin,0.0)
    elif args.cmSpecModel == 'spectrum':
        pmin = np.append(pmin,-8.0*np.ones(nmodes_red))
if args.incEph:
    if args.ephSpecModel == 'powerlaw':
        pmin = np.append(pmin,np.array([-20.0,-20.0,-20.0]))
        pmin = np.append(pmin,np.array([0.0,0.0,0.0]))
    elif args.ephSpecModel == 'spectrum':
        pmin = np.append(pmin,-30.0*np.ones(3*nmodes_eph))
if args.incGWB:
    if args.gwbSpecModel == 'powerlaw':
        if args.gwbAmpRange is None:
            pmin = np.append(pmin,-18.0)
        elif args.gwbAmpRange is not None:
            pmin = np.append(pmin,amp_range[0])
        if not args.fix_slope:
            pmin = np.append(pmin,0.0)
    elif args.gwbSpecModel == 'spectrum':
        if args.gwbPrior != 'gaussProc':
            pmin = np.append(pmin,-8.0*np.ones(nmodes_red))
        elif args.gwbPrior == 'gaussProc':
            pmin = np.append(pmin,-5.0*np.ones(nmodes_red))
            pmin = np.append(pmin,-18.0) # Agwb
            if gwb_popparam_ndims == 1:
                if gwb_popparam == 'stars' and args.gwbStarsRange is not None:
                    pmin = np.append(pmin,stars_range[0])
                elif gwb_popparam == 'ecc' and args.gwbEccRange is not None:
                    pmin = np.append(pmin,ecc_range[0])
                else:
                    pmin = np.append(pmin,gppkl[0].x.min())
            else:
                if gwb_popparam == 'starsecc' and args.gwbStarsRange is not None \
                  and args.gwbEccRange is not None:
                    pmin = np.append(pmin,stars_range[0])
                    pmin = np.append(pmin,ecc_range[0])
                else:
                    for col in range(gppkl[0].x.shape[1]):
                        pmin = np.append(pmin,gppkl[0].x[:,col].min())
    elif args.gwbSpecModel == 'turnover':
        pmin = np.append(pmin,-18.0) # Agwb
        if args.gwb_fb2env is not None:
            # kappa is fixed by choice of env
            if args.gwb_fb2env == 'stars':
                pmin = np.append(pmin,0.0) # log(rho)
            elif args.gwb_fb2env == 'gas':
                pmin = np.append(pmin,-3.0) # log(acc-rate)
        elif args.gwb_fb2env is None:
            pmin = np.append(pmin,np.array([0.0,-9.0]))
    elif args.gwbSpecModel == 'gpEnvInterp':
        pmin = np.append(pmin,np.array([-18.0,0.0]))
    if args.incCorr:
        if args.gwbTypeCorr == 'modelIndep':
            pmin = np.append(pmin,0.0*np.ones(num_corr_params))
        elif args.gwbTypeCorr == 'pointSrc':
            if args.fixPointSrcPhi is None and args.fixPointSrcTheta is None:
                pmin = np.append(pmin,np.tile([0.0,-1.0],tmp_nwins))
        elif args.gwbTypeCorr == 'spharmAnis':
            pmin = np.append(pmin,-10.0*np.ones(num_corr_params))
        elif args.gwbTypeCorr == 'dipoleOrf':
            pmin = np.append(pmin,np.tile([0.0,-1.0,0.0],tmp_nwins))
        elif args.gwbTypeCorr == 'gwDisk':
            pmin = np.append(pmin,np.tile([0.0,-1.0,0.0,-2.0],tmp_nwins))
        elif args.gwbTypeCorr == 'psrlocsVary':
            pmin = np.append(pmin,np.tile(np.zeros(len(psr)),tmp_nwins))
            pmin = np.append(pmin,np.tile(-1.0*np.ones(len(psr)),tmp_nwins))
        if args.gwbModelSelect:
            pmin = np.append(pmin,-0.5)
if args.incGWline:
    pmin = np.append(pmin,np.array([-8.0,-10.0,0.0,-1.0]))
if args.det_signal:
    if args.cgw_search:
        if args.cgwFreqRange is None:
            pmin = np.append(pmin,np.array([7.0,0.1,0.0,-17.0,-9.301,
                                            0.0,-1.0,-1.0,0.0,0.0,0.0]))
        elif args.cgwFreqRange is not None:
            pmin = np.append(pmin,np.array([7.0,0.1,0.0,-17.0,cgw_orbfreq_range[0],
                                            0.0,-1.0,-1.0,0.0,0.0,0.0]))
        if args.ecc_search:
            pmin = np.append(pmin,0.0)
        if args.psrTerm:
            # psr distances, pterm-gamm0, pterm-l0
            pmin = np.append(pmin,0.001*np.ones(len(psr)))
            pmin = np.append(pmin,np.zeros(len(psr)))
            pmin = np.append(pmin,np.zeros(len(psr)))
        if args.cgwModelSelect:
            pmin = np.append(pmin,-0.5)
    elif args.bwm_search:
        pmin = np.append(pmin,[np.min([np.min(p.toas) for p in psr]),
                               -18.0,0.0,-1.0,0.0])
        if args.bwm_model_select:
            pmin = np.append(pmin,-0.5)
    if args.eph_quadratic:
        pmin = np.append(pmin,np.tile([-10.0,-10.0],3)) # amps
        pmin = np.append(pmin,np.tile([-1.0,-1.0],3)) # signs
    if args.eph_planetdelta:
        pmin = np.append(pmin,-20.0*np.ones(num_planets)) # amps
        pmin = np.append(pmin,-1.0*np.ones(num_planets)) # signs
        

pmax = np.array([])
if not args.fixRed:
    if args.redSpecModel == 'powerlaw':
        pmax = np.append(pmax,-11.0*np.ones(len(psr)))
        pmax = np.append(pmax,7.0*np.ones(len(psr)))
    elif args.redSpecModel == 'spectrum':
        pmax = np.append(pmax,3.0*np.ones(len(psr)*nmodes_red))
if args.incDM and not args.fixDM:
    if args.dmSpecModel == 'powerlaw':
        # slightly higher than red due to normalisation
        pmax = np.append(pmax,-8.0*np.ones(len(psr)))
        pmax = np.append(pmax,7.0*np.ones(len(psr)))
    elif args.dmSpecModel == 'spectrum':
        pmax = np.append(pmax,3.0*np.ones(len(psr)*nmodes_dm))
if args.varyWhite:
    for ii,p in enumerate(psr):
        systems = p.sysflagdict[args.sysflag_target]
        pmax = np.append(pmax,10.0*np.ones(len(systems)))
        pmax = np.append(pmax,-3.0*np.ones(len(systems)))
        if 'nano-f' in p.sysflagdict.keys() and len(p.sysflagdict['nano-f'].keys())>0:
            pmax = np.append(pmax, -3.0*np.ones(len(p.sysflagdict['nano-f'].keys())))
if args.incBand:
    if args.bandSpecModel == 'powerlaw':
        pmax = np.append(pmax,-11.0*np.ones(len(bands)-1))
        pmax = np.append(pmax,7.0*np.ones(len(bands)-1))
    elif args.bandSpecModel == 'spectrum':
        pmax = np.append(pmax,-3.0*np.ones((len(bands)-1)*nmodes_band))
if args.incClk:
    if args.clkSpecModel == 'powerlaw':
        pmax = np.append(pmax,-11.0)
        pmax = np.append(pmax,7.0)
    elif args.clkSpecModel == 'spectrum':
        pmax = np.append(pmax,3.0*np.ones(nmodes_red))
if args.incCm:
    if args.cmSpecModel == 'powerlaw':
        pmax = np.append(pmax,-11.0)
        pmax = np.append(pmax,7.0)
    elif args.cmSpecModel == 'spectrum':
        pmax = np.append(pmax,3.0*np.ones(nmodes_red))
if args.incEph:
    if args.ephSpecModel == 'powerlaw':
        pmax = np.append(pmax,np.array([-11.0,-11.0,-11.0]))
        pmax = np.append(pmax,np.array([7.0,7.0,7.0]))
    elif args.ephSpecModel == 'spectrum':
        pmax = np.append(pmax,-3.0*np.ones(3*nmodes_eph))
if args.incGWB:
    if args.gwbSpecModel == 'powerlaw':
        if args.gwbAmpRange is None:
            pmax = np.append(pmax,-11.0)
        elif args.gwbAmpRange is not None:
            pmax = np.append(pmax,amp_range[1])
        if not args.fix_slope:
            pmax = np.append(pmax,7.0)
    elif args.gwbSpecModel == 'spectrum':
        if args.gwbPrior != 'gaussProc':
            pmax = np.append(pmax,3.0*np.ones(nmodes_red))
        elif args.gwbPrior == 'gaussProc':
            pmax = np.append(pmax,5.0*np.ones(nmodes_red))
            pmax = np.append(pmax,-11.0) # Agwb
            if gwb_popparam_ndims == 1:
                if gwb_popparam == 'stars' and args.gwbStarsRange is not None:
                    pmax = np.append(pmax,stars_range[1])
                elif gwb_popparam == 'ecc' and args.gwbEccRange is not None:
                    pmax = np.append(pmax,ecc_range[1])
                else:
                    pmax = np.append(pmax,gppkl[0].x.max())
            else:
                if gwb_popparam == 'starsecc' and args.gwbStarsRange is not None \
                  and args.gwbEccRange is not None:
                    pmax = np.append(pmax,stars_range[1])
                    pmax = np.append(pmax,ecc_range[1])
                else:
                    for col in range(gppkl[0].x.shape[1]):
                        pmax = np.append(pmax,gppkl[0].x[:,col].max())
    elif args.gwbSpecModel == 'turnover':
        pmax = np.append(pmax,-11.0) # Agwb
        if args.gwb_fb2env is not None:
            # kappa is fixed by choice of env
            if args.gwb_fb2env == 'stars':
                pmax = np.append(pmax,6.0) # log(rho)
            elif args.gwb_fb2env == 'gas':
                pmax = np.append(pmax,2.0) # log(acc-rate)
        elif args.gwb_fb2env is None:
            pmax = np.append(pmax,np.array([7.0,-7.0]))
    elif args.gwbSpecModel == 'gpEnvInterp':
        pmax = np.append(pmax,np.array([-11.0,0.9]))
    if args.incCorr:
        if args.gwbTypeCorr == 'modelIndep':
            pmax = np.append(pmax,np.pi*np.ones(num_corr_params))
        elif args.gwbTypeCorr == 'pointSrc':
            if args.fixPointSrcPhi is None and args.fixPointSrcTheta is None:
                pmax = np.append(pmax,np.tile([2.0*np.pi,1.0],tmp_nwins))
        elif args.gwbTypeCorr == 'spharmAnis':
            pmax = np.append(pmax,10.0*np.ones(num_corr_params))
        elif args.gwbTypeCorr == 'dipoleOrf':
            pmax = np.append(pmax,np.tile([2.0*np.pi,1.0,1.0],tmp_nwins))
        elif args.gwbTypeCorr == 'gwDisk':
            pmax = np.append(pmax,np.tile([2.0*np.pi,1.0,np.pi,6.0],tmp_nwins))
        elif args.gwbTypeCorr == 'psrlocsVary':
            pmax = np.append(pmax,np.tile(2.0*np.pi*np.ones(len(psr)),tmp_nwins))
            pmax = np.append(pmax,np.tile(np.ones(len(psr)),tmp_nwins))
        if args.gwbModelSelect:
            pmax = np.append(pmax,1.5)
if args.incGWline:
    pmax = np.append(pmax,np.array([3.0,-7.0,2.0*np.pi,1.0]))
if args.det_signal:
    if args.cgw_search:
        if args.cgwFreqRange is None:
            pmax = np.append(pmax,np.array([10.0,1.0,4.0,-11.0,-7.301,2.0*np.pi,
                                            1.0,1.0,np.pi,np.pi,2.0*np.pi]))
        elif args.cgwFreqRange is not None:
            pmax = np.append(pmax,np.array([10.0,1.0,4.0,-11.0,cgw_orbfreq_range[1],2.0*np.pi,
                                            1.0,1.0,np.pi,np.pi,2.0*np.pi]))
        if args.ecc_search:
            pmax = np.append(pmax,0.9)
        if args.psrTerm:
            # psr distances, pterm-gamma0, pterm-l0
            pmax = np.append(pmax,10.0*np.ones(len(psr)))
            pmax = np.append(pmax,2.0*np.pi*np.ones(len(psr)))
            pmax = np.append(pmax,2.0*np.pi*np.ones(len(psr)))
        if args.cgwModelSelect:
            pmax = np.append(pmax,1.5)
    elif args.bwm_search:
        pmax = np.append(pmax,[np.max([np.max(p.toas) for p in psr]),
                               -11.0,2.0*np.pi,1.0,np.pi])
        if args.bwm_model_select:
            pmax = np.append(pmax,1.5)
    if args.eph_quadratic:
        pmax = np.append(pmax,np.tile([0.0,0.0],3)) # amps
        pmax = np.append(pmax,np.tile([1.0,1.0],3)) # signs
    if args.eph_planetdelta:
        pmax = np.append(pmax,-5.0*np.ones(num_planets)) # amps
        pmax = np.append(pmax,1.0*np.ones(num_planets)) # signs
       

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
    if not args.varyWhite:
        loglike1_tmp = loglike1
        dtmp = list(d)
        TtNT_tmp = list(TtNT)
        Jamp_tmp = list(Jamp)
        logdet_Ntmp = list(logdet_N)

    mode_count = 2*nmodes_red
    if args.incDM:
        mode_count += 2*nmodes_dm
    if args.incBand and ((len(bands)-1)>0):
        mode_count += 2*(len(bands)-1)*nmodes_band
    if args.incEph:
        # 2*nmode for x,y,z
        mode_count += 6*nmodes_eph
    if args.incClk and args.clkDesign:
        mode_count += 2*nmodes_red
    
    ###############################
    # Splitting up parameter vector

    param_ct = 0
    
    ###############################
    # Including per-pulsar red noise
    
    if not args.fixRed:
        if args.redSpecModel == 'powerlaw':
            Ared = 10.0**xx[:npsr]
            gam_red = xx[npsr:2*npsr]
            param_ct += 2*npsr
        elif args.redSpecModel == 'spectrum':
            red_spec = (xx[:nmodes_red*npsr].copy()).reshape((npsr,nmodes_red))
            param_ct += npsr*nmodes_red

    ####################################
    # Including per-pulsar DM variations
    
    if args.incDM and not args.fixDM:
        if args.dmSpecModel == 'powerlaw':
            Adm = 10.0**xx[param_ct:param_ct+npsr]
            gam_dm = xx[param_ct+npsr:param_ct+2*npsr]
            param_ct += 2*npsr
        elif args.dmSpecModel == 'spectrum':
            dm_spec = (xx[param_ct:param_ct+nmodes_dm*npsr].copy()).reshape((npsr,nmodes_dm))
            param_ct += npsr*nmodes_dm

    ####################################
    # Including per-pulsar white-noise
    
    if args.varyWhite:
        EFAC = []
        EQUAD = []
        ECORR = []
        for ii,p in enumerate(psr):
            systems = p.sysflagdict[args.sysflag_target]
            
            EFAC.append( xx[param_ct:param_ct+len(systems)] )
            param_ct += len(systems)
    
            EQUAD.append( 10.0**xx[param_ct:param_ct+len(systems)] )
            param_ct += len(systems)

            if 'nano-f' in p.sysflagdict.keys() and len(p.sysflagdict['nano-f'].keys())>0:
                ECORR.append( 10.0**xx[param_ct:param_ct+len(p.sysflagdict['nano-f'].keys())] )
                param_ct += len(p.sysflagdict['nano-f'].keys())

    #########################################
    # Including band-dependent red noise
    
    if args.incBand:
        if args.bandSpecModel == 'powerlaw':
            Aband = [10.0**xx[param_ct+ii] for ii in range((len(bands)-1))]
            param_ct += (len(bands)-1)
            gam_band = [xx[param_ct+ii] for ii in range((len(bands)-1))]
            param_ct += (len(bands)-1)
        elif args.bandSpecModel == 'spectrum':
            band_spec = (xx[param_ct:param_ct+(len(bands)-1)*nmodes_band].copy()).reshape(((len(bands)-1),nmodes_band))
            param_ct += (len(bands)-1)*nmodes_band

    #########################################
    # Including clock errors
    
    if args.incClk:
        if args.clkSpecModel == 'powerlaw':
            Aclk = 10.0**xx[param_ct]
            gam_clk = xx[param_ct+1]
            param_ct += 2
        elif args.clkSpecModel == 'spectrum':
            clk_spec = xx[param_ct:param_ct+nmodes_red].copy()
            param_ct += nmodes_red

    #########################################
    # Including a common uncorrelated process
    
    if args.incCm:
        if args.cmSpecModel == 'powerlaw':
            Acm = 10.0**xx[param_ct]
            gam_cm = xx[param_ct+1]
            param_ct += 2
        elif args.cmSpecModel == 'spectrum':
            cm_spec = xx[param_ct:param_ct+nmodes_red].copy()
            param_ct += nmodes_red

    #########################################
    # Including solar-system ephemeris errors
    
    if args.incEph:
        if args.ephSpecModel == 'powerlaw':
            Aephx = 10.0**xx[param_ct]
            Aephy = 10.0**xx[param_ct+1]
            Aephz = 10.0**xx[param_ct+2]
            gam_ephx = xx[param_ct+3]
            gam_ephy = xx[param_ct+4]
            gam_ephz = xx[param_ct+5]
            param_ct += 6
        elif args.ephSpecModel == 'spectrum':
            eph_spec = (xx[param_ct:param_ct+3*nmodes_eph].copy()).reshape((3,nmodes_eph))
            param_ct += 3*nmodes_eph

    ############################
    # Including a GW background
    
    if args.incGWB:
        # GWB parameters
        if args.gwbSpecModel == 'powerlaw':
            Agwb = 10.0**xx[param_ct]
            param_ct += 1
            if args.fix_slope:
                gam_gwb = 13./3.
            else:
                gam_gwb = xx[param_ct]
                param_ct += 1
        elif args.gwbSpecModel == 'spectrum':
            rho_spec = xx[param_ct:param_ct+nmodes_red]
            param_ct += nmodes_red
            if args.gwbPrior == 'gaussProc':
                Agwb = 10.0**xx[param_ct]
                env_param = xx[param_ct+1:param_ct+1+gwb_popparam_ndims]
                param_ct += 1 + gwb_popparam_ndims
        elif args.gwbSpecModel == 'turnover':
            Agwb = 10.0**xx[param_ct]
            if args.gwb_fb2env is not None:
                kappaturn = fb2env.kappa
                fbend = fb2env.fb_from_env(envParam=10.0**xx[param_ct+1])
                param_ct += 2
            elif args.gwb_fb2env is None:
                kappaturn = xx[param_ct+1]
                fbend = 10.0**xx[param_ct+2]
                param_ct += 3
        elif args.gwbSpecModel == 'gpEnvInterp':
            Agwb = 10.0**xx[param_ct]
            ecc = xx[param_ct+1]
            param_ct += 2

        gwb_modindex = 0
        if args.incCorr:
            # Anisotropy parameters
            orf_coeffs = xx[param_ct:param_ct+num_corr_params]
            param_ct += num_corr_params

            if args.gwbModelSelect:
                # '0' is uncorrelated GWB, '1' is correlated GWB
                gwb_modindex = int(np.rint(xx[param_ct]))
                param_ct += 1
            elif not args.gwbModelSelect:
                gwb_modindex = 1

    ###############################
    # Including a single GW line

    if args.incGWline:
        spec_gwline = xx[param_ct]
        freq_gwline = 10.0**xx[param_ct+1]
        phi_gwline = xx[param_ct+2]
        theta_gwline = np.arccos(xx[param_ct+3])
        param_ct += 4

    ###############################
    # Creating continuous GW signal

    if args.det_signal:
        if args.cgw_search:
            if args.cgwModelSelect:
                cgw_params = xx[param_ct:-1]
                # '0' is noise-only, '1' is CGW
                nmodel = int(np.rint(xx[-1]))
            else:
                cgw_params = xx[param_ct:]
        elif args.bwm_search:
            if args.bwm_model_select:
                bwm_params = xx[param_ct:-1]
                # '0' is noise-only, '1' is BWM
                nmodel = int(np.rint(xx[-1]))
            else:
                bwm_params = xx[param_ct:]
        # fix this for single GW signals as well as eph_quadratic / eph_planetdelta
        if args.eph_quadratic:
            xquad1_amp, xquad2_amp, \
              yquad1_amp, yquad2_amp, \
              zquad1_amp, zquad2_amp = xx[param_ct:param_ct+6]
            xquad1_sign, xquad2_sign, \
              yquad1_sign, yquad2_sign, \
              zquad1_sign, zquad2_sign = xx[param_ct+6:param_ct+12]
        if args.eph_planetdelta:
            planet_delta_amp = xx[param_ct:param_ct+num_planets]
            planet_delta_sign = xx[param_ct+num_planets:param_ct+2*num_planets]
            
    ############################
    ############################
    # Now, evaluating likelihood

    if args.constLike:

        if args.incGWB and args.incCorr:
            if args.gwbTypeCorr == 'spharmAnis':
            
                ################################################
                # Reshaping freq-dependent anis coefficients,
                # and testing for power distribution physicality.
            
                orf_coeffs = orf_coeffs.reshape((tmp_nwins,
                                                ((args.LMAX+1)**2)-1))
                clm = np.array([[0.0]*((args.LMAX+1)**2)
                                for ii in range(tmp_nwins)])
                clm[:,0] = 2.0*np.sqrt(np.pi)

                if args.LMAX!=0:

                    for kk in range(tmp_nwins):
                        for ii in range(1,((args.LMAX+1)**2)):
                            clm[kk,ii] = orf_coeffs[kk,ii-1]   

                        if not args.noPhysPrior:
                            # Testing for physicality of power distribution.
                            if (utils.PhysPrior(clm[kk],harm_sky_vals) == 'Unphysical'):
                                return -np.inf

            elif args.gwbTypeCorr == 'psrlocsVary':

                ################################################
                # Reshaping freq-dependent corr coefficients
                
                orf_coeffs = orf_coeffs.reshape((2,tmp_nwins*len(psr)))
                varyPhi = orf_coeffs[0,:].reshape((tmp_nwins,len(psr)))
                varyTheta = np.arccos(orf_coeffs[1,:]).reshape((tmp_nwins,len(psr)))

                # constant likelihood for one window only
                varyLocs = np.zeros((len(psr),2))
                varyLocs[:,0] = varyPhi[0,:]
                varyLocs[:,1] = varyTheta[0,:]

                logLike = 0.0

            elif args.gwbTypeCorr == 'modelIndep':

                npairs = int(npsr*(npsr-1)/2)
                phi_corr = orf_coeffs.copy().reshape((tmp_nwins,npairs))
 
                for ii in range(tmp_nwins): # number of frequency windows
                    for jj in range(len(corr_modefreqs[ii])): # number of frequencies in this window
                        upper_triang = np.zeros((npsr,npsr))
                        phi_els = np.array([[0.0]*kk for kk in range(1,npsr)])

                        ct=0
                        for aa in range(len(phi_els)):
                            for bb in range(len(phi_els[aa])):
                                phi_els[aa][bb] = phi_corr[ii,ct]
                                ct += 1

                        upper_triang[0,0] = 1.
                        for bb in range(1,upper_triang.shape[1]):
                            upper_triang[0,bb] = np.cos(phi_els[bb-1][0])
                        for aa in range(1,upper_triang.shape[1]):
                            upper_triang[aa,aa] = np.prod( np.sin(phi_els[aa-1]) )
                        for aa in range(1,upper_triang.shape[1]):
                            for bb in range(aa+1,upper_triang.shape[1]):
                                upper_triang[aa,bb] = np.cos(phi_els[bb-1][aa]) * \
                                np.prod( np.sin(np.array(phi_els[bb-1])[0:aa]) )

                logLike = 0.0
        
        else:

            logLike = 0.0
        
    elif not args.constLike:

        if args.varyWhite:
            
            loglike1_tmp = 0
            logdet_Ntmp = []
            TtNT_tmp = []
            dtmp = []
            Jamp_tmp = []
            for ii,p in enumerate(psr):

                scaled_err = (p.toaerrs).copy()
                systems = p.sysflagdict[args.sysflag_target]
                for jj,sysname in enumerate(systems):
                    scaled_err[systems[sysname]] *= EFAC[ii][jj] 
                ###
                white_noise = np.ones(len(scaled_err))
                for jj,sysname in enumerate(systems):
                    white_noise[systems[sysname]] *= EQUAD[ii][jj]
    
                new_err = np.sqrt( scaled_err**2.0 + white_noise**2.0 )
                ########

                # compute ( T.T * N^-1 * T )
                # & log determinant of N
                if not args.noEcorr:
        
                    if 'nano-f' in p.sysflagdict.keys() and len(ECORR[ii])>0:

                        Jamp_tmp.append(np.ones(len(p.epflags)))
                        for jj,nano_sysname in enumerate(p.sysflagdict['nano-f'].keys()):
                            Jamp_tmp[ii][np.where(p.epflags==nano_sysname)] *= \
                              ECORR[ii][jj]**2.0

                        Nx = jitter.cython_block_shermor_0D(p.res, new_err**2.,
                                                            Jamp_tmp[ii], p.Uinds)
                        dtmp.append(np.dot(p.Te.T, Nx))
            
                        logdet_N_dummy, TtNT_dummy = \
                        jitter.cython_block_shermor_2D(p.Te, new_err**2.,
                                                        Jamp_tmp[ii], p.Uinds)
                        logdet_Ntmp.append(logdet_N_dummy)
                        TtNT_tmp.append(TtNT_dummy)
            
                        det_dummy, dtNdt = \
                        jitter.cython_block_shermor_1D(p.res, new_err**2.,
                                                        Jamp_tmp[ii], p.Uinds)

                    else:
            
                        dtmp.append(np.dot(p.Te.T, p.res/( new_err**2.0 )))
            
                        N = 1./( new_err**2.0 )
                        right = (N*p.Te.T).T
                        TtNT_tmp.append(np.dot(p.Te.T, right))
    
                        logdet_Ntmp.append(np.sum(np.log( new_err**2.0 )))
        
                        # triple product in likelihood function
                        dtNdt = np.sum(p.res**2.0/( new_err**2.0 ))
        
                else:
        
                    dtmp.append(np.dot(p.Te.T, p.res/( new_err**2.0 )))
            
                    N = 1./( new_err**2.0 )
                    right = (N*p.Te.T).T
                    TtNT_tmp.append(np.dot(p.Te.T, right))

                    logdet_Ntmp.append(np.sum(np.log( new_err**2.0 )))
        
                    # triple product in likelihood function
                    dtNdt = np.sum(p.res**2.0/( new_err**2.0 ))

                loglike1_tmp += -0.5 * (logdet_Ntmp[ii] + dtNdt)

    
        if args.det_signal:
            if args.cgw_search:

                if args.ecc_search:
                    binary_params = 12
                    logmass, qr, logdist, loghstrain, logorbfreq, gwphi,\
                      costheta, cosinc, gwpol, gwgamma0, l0, e0 = cgw_params[:binary_params]
                else:
                    binary_params = 11
                    logmass, qr, logdist, loghstrain, logorbfreq, gwphi,\
                      costheta, cosinc, gwpol, gwgamma0, l0 = cgw_params[:binary_params]

                mc = 10.0**logmass
                dist = 10.0**logdist
                hstrain = 10.0**loghstrain
                orbfreq = 10.0**logorbfreq
                gwtheta = np.arccos(costheta)
                gwinc = np.arccos(cosinc)

                if args.psrTerm:
                    pterm_params = cgw_params[binary_params:]
                    psrdists = pterm_params[:npsr]
                    psrgp0 = pterm_params[npsr:2*npsr]
                    psrlp0 = pterm_params[2*npsr:]
                elif not args.psrTerm:
                    psrdists = np.array([None]*npsr)
                    psrgp0 = np.array([None]*npsr)
                    psrlp0 = np.array([None]*npsr)
            
                cgw_res = []
                detres = []

                #########################
                # Sometimes we might want
                # to fix cgw parameters
                
                if args.fixcgwFreq is None:
                    orbfreq_tmp = orbfreq
                elif args.fixcgwFreq is not None:
                    orbfreq_tmp = 10**args.fixcgwFreq

                if args.fixcgwPhi is None:
                    gwphi_tmp = gwphi
                elif args.fixcgwPhi is not None:
                    gwphi_tmp = args.fixcgwPhi

                if args.fixcgwTheta is None:
                    gwtheta_tmp = gwtheta
                elif args.fixcgwTheta is not None:
                    gwtheta_tmp = args.fixcgwTheta
                
                if args.ecc_search:
                    if args.fixcgwEcc is None:
                        ecc_tmp = e0
                    elif args.fixcgwEcc is not None:
                        ecc_tmp = args.fixcgwEcc
                    gwgamma_tmp = gwgamma0
                elif not args.ecc_search:
                    ecc_tmp = 0.0
                    gwgamma_tmp = gwgamma0

                ########################

                if args.cgwPrior == 'uniform' or args.cgwPrior == 'loguniform':
                    hstrain_tmp = hstrain
                elif args.cgwPrior == 'mdloguniform':
                    hstrain_tmp = None
                    
                for ii,p in enumerate(psr):

                    if args.cgwModelSelect and nmodel == 0:
                        
                        cgw_res.append( np.zeros(len(p.toas)) )
                        
                    elif (args.cgwModelSelect and nmodel == 1) or not args.cgwModelSelect:
                           
                        tmp_res = utils.ecc_cgw_signal(p, gwtheta_tmp, gwphi_tmp, mc,
                                                    dist, hstrain_tmp, orbfreq_tmp,
                                                    gwinc, gwpol, gwgamma_tmp, ecc_tmp,
                                                    l0, qr, nmax=10000, pd=psrdists[ii],
                                                    gpx=psrgp0[ii], lpx=psrlp0[ii],
                                                    periEv=args.periEv, psrTerm=args.psrTerm,
                                                    tref=tref, epochTOAs=args.epochTOAs,
                                                    noEccEvolve=args.noEccEvolve)
                    
                        if args.epochTOAs:
                            cgw_res.append(np.ones(len(p.toas)))
                            for cc, swave in enumerate(tmp_res):
                                cgw_res[ii][p.detsig_Uinds[cc,0]:p.detsig_Uinds[cc,1]] *= swave
                        elif not args.epochTOAs:
                            cgw_res.append(tmp_res)
                        
                    detres.append( p.res - cgw_res[ii] )


            elif args.bwm_search:

                bwm_res = []
                detres = []
                for ii,p in enumerate(psr):
                    if args.bwm_model_select:
                        if nmodel == 0:
                            bwm_res.append( np.zeros(len(p.toas)) )
                        elif nmodel == 1:
                            bwm_res.append( utils.bwmsignal(bwm_params,p,
                                                            antennaPattern=args.bwm_antenna) )
                    else:
                        bwm_res.append( utils.bwmsignal(bwm_params,p,
                                                        antennaPattern=args.bwm_antenna) )
                    detres.append( p.res - bwm_res[ii] )

            if args.eph_quadratic:

                detres = []
                for ii, p in enumerate(psr):

                    # define the pulsar position vector
                    pphi = p.psr_locs[0]
                    ptheta = np.pi/2. - p.psr_locs[1]
                    x = np.sin(ptheta)*np.cos(pphi)
                    y = np.sin(ptheta)*np.sin(pphi)
                    z = np.cos(ptheta)

                    normtime = (p.toas - tref)/365.25
                    x_quad = (np.sign(xquad1_sign) * 10.0**xquad1_amp * normtime + \
                              np.sign(xquad2_sign) * 10.0**xquad2_amp * normtime**2.0) * x
                    y_quad = (np.sign(yquad1_sign) * 10.0**yquad1_amp * normtime + \
                              np.sign(yquad2_sign) * 10.0**yquad2_amp * normtime**2.0) * y
                    z_quad = (np.sign(zquad1_sign) * 10.0**zquad1_amp * normtime + \
                              np.sign(zquad2_sign) * 10.0**zquad2_amp * normtime**2.0) * z

                    # need to alter this if you want a single GW source too
                    detres.append( p.res - x_quad - y_quad - z_quad )

            if args.eph_planetdelta:

                detres = []
                for ii, p in enumerate(psr):

                    # define the pulsar position vector
                    pphi = p.psr_locs[0]
                    ptheta = np.pi/2. - p.psr_locs[1]
                    x = np.sin(ptheta)*np.cos(pphi)
                    y = np.sin(ptheta)*np.sin(pphi)
                    z = np.cos(ptheta)
                    psr_posvec = np.array([x,y,z])

                    planet_delta_signal = np.zeros(p.toas.shape)
                    # sum over planets
                    dummy_tags = planet_tags - 1
                    for jj,tag in enumerate(dummy_tags):
                        planet_delta_signal += (np.sign(planet_delta_sign[jj]) * 10.0**planet_delta_amp[jj] * \
                                                np.dot(p.planet_ssb[:,tag,:3],psr_posvec))
                    
                    # need to alter this if you want a single GW source too
                    detres.append( p.res - planet_delta_signal)

            #############################################################
            # Recomputing some noise quantities involving 'residuals'.
            # Unfortunately necessary when we have a deterministic signal.
        
            loglike1_tmp = 0
            dtNdt = []
            for ii,p in enumerate(psr):

                # compute ( T.T * N^-1 * T )
                # & log determinant of N

                if args.varyWhite:
                    scaled_err = (p.toaerrs).copy()
                    systems = p.sysflagdict[args.sysflag_target]
                    for jj,sysname in enumerate(systems):
                        scaled_err[systems[sysname]] *= EFAC[ii][jj] 
                    ###
                    white_noise = np.ones(len(scaled_err))
                    for jj,sysname in enumerate(systems):
                        white_noise[systems[sysname]] *= EQUAD[ii][jj]
    
                    new_err = np.sqrt( scaled_err**2.0 + white_noise**2.0 )
                elif not args.varyWhite:
                    new_err = (p.toaerrs).copy()
                    
                if not args.noEcorr:

                    if (args.varyWhite and len(ECORR[ii]>0)) or \
                      (not args.varyWhite and p.ecorrs is not None and len(p.ecorrs)>0):
                        Nx = jitter.cython_block_shermor_0D(detres[ii], new_err**2.,
                                                            Jamp_tmp[ii], p.Uinds)
                        dtmp[ii] = np.dot(p.Te.T, Nx)
                        det_dummy, dtNdt_dummy = \
                        jitter.cython_block_shermor_1D(detres[ii], new_err**2.,
                                                        Jamp_tmp[ii], p.Uinds)
                        dtNdt.append(dtNdt_dummy)

                    else:
            
                        dtmp[ii] = np.dot(p.Te.T, detres[ii]/( new_err**2.0 ))
                        dtNdt.append(np.sum(detres[ii]**2.0/( new_err**2.0 )))
                
                else:
        
                    dtmp[ii] = np.dot(p.Te.T, detres[ii]/( new_err**2.0 ))
                    dtNdt.append(np.sum(detres[ii]**2.0/( new_err**2.0 )))

                loglike1_tmp += -0.5 * (logdet_Ntmp[ii] + dtNdt[ii])
        
        
        if args.incGWB and args.incCorr:
            ## (option to de-restrict clms by phys prior)... and gwb_modindex==1:
            
            if args.gwbTypeCorr == 'modelIndep':

                npairs = int(npsr*(npsr-1)/2)
                phi_corr = orf_coeffs.copy().reshape((tmp_nwins,npairs))

                ############################################################
                # Computing frequency-dependent overlap reduction functions.

                ORF=[]
                for ii in range(tmp_nwins): # number of frequency windows
                    for jj in range(len(corr_modefreqs[ii])): # number of frequencies in this window
                        upper_triang = np.zeros((npsr,npsr))
                        phi_els = np.array([[0.0]*kk for kk in range(1,npsr)])
                        ct=0
                        for aa in range(len(phi_els)):
                            for bb in range(len(phi_els[aa])):
                                phi_els[aa][bb] = phi_corr[ii,ct]
                                ct += 1

                        upper_triang[0,0] = 1.
                        for bb in range(1,upper_triang.shape[1]):
                            upper_triang[0,bb] = np.cos(phi_els[bb-1][0])
                        for aa in range(1,upper_triang.shape[1]):
                            upper_triang[aa,aa] = np.prod( np.sin(phi_els[aa-1]) )
                        for aa in range(1,upper_triang.shape[1]):
                            for bb in range(aa+1,upper_triang.shape[1]):
                                upper_triang[aa,bb] = np.cos(phi_els[bb-1][aa]) * \
                                np.prod( np.sin(np.array(phi_els[bb-1])[0:aa]) )   

                        ORF.append(np.dot( upper_triang.T, upper_triang ))
       
                if args.incDM:
                    for ii in range(nmodes_dm): 
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incEph:
                    for kk in range(3): # x,y,z
                        for ii in range(nmodes_eph):
                            ORF.append( np.zeros((npsr,npsr)) )

                if args.incClk and args.clkDesign:
                    for ii in range(nmodes_red): 
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incBand:
                    for kk in range(len(bands)-1):
                        for ii in range(nmodes_band):
                            ORF.append( np.zeros((npsr,npsr)) )

                ORF = np.array(ORF)
                ORFtot = np.zeros((mode_count,npsr,npsr)) # shouldn't be applying ORF to dmfreqs,
                                                          # but the projection of GW spec onto dmfreqs
                                                          # is defined as zero below.
                ORFtot[0::2] = ORF
                ORFtot[1::2] = ORF

            elif args.gwbTypeCorr == 'pointSrc':

                if args.fixPointSrcPhi is not None and args.fixPointSrcTheta is not None:
                    gwphi = np.tile(args.fixPointSrcPhi,tmp_nwins)
                    gwtheta = np.tile(args.fixPointSrcTheta,tmp_nwins)
                else:
                    orf_coeffs = orf_coeffs.reshape((tmp_nwins,2))
                    gwphi, cosgwtheta = orf_coeffs[:,0], orf_coeffs[:,1]
                    gwtheta = np.arccos(cosgwtheta)

                corr_curve=np.zeros((tmp_nwins,npsr,npsr))

                Fp = np.zeros((tmp_nwins,npsr))
                Fc = np.zeros((tmp_nwins,npsr))
                for kk in range(tmp_nwins):
                    for ii in range(npsr):
                        Fp[kk,ii], Fc[kk,ii] = \
                        utils.fplus_fcross(psr[ii], gwtheta[kk], gwphi[kk])

                for kk in range(tmp_nwins):
                    for ii in range(npsr):
                        for jj in range(ii,npsr):
                            corr_curve[kk,ii,jj] = 4.0*np.pi * (3.0/(8.0*np.pi)) * \
                              (Fp[kk,ii]*Fp[kk,jj] + Fc[kk,ii]*Fc[kk,jj])
                            corr_curve[kk,jj,ii] = corr_curve[kk,ii,jj]

                            if ii == jj:
                                # scaling for pulsar-term
                                corr_curve[kk,ii,jj] *= 2.0

                ORF=[]
                for ii in range(tmp_nwins): # number of frequency windows
                    for jj in range(len(corr_modefreqs[ii])): # number of frequencies in this window
                        ORF.append( corr_curve[ii,:,:] )
                    
                if args.incDM:
                    for ii in range(nmodes_dm):
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incEph:
                    for kk in range(3): # x,y,z
                        for ii in range(nmodes_eph): 
                            ORF.append( np.zeros((npsr,npsr)) )

                if args.incClk and args.clkDesign:
                    for ii in range(nmodes_red): 
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incBand:
                    for kk in range(len(bands)-1):
                        for ii in range(nmodes_band):
                            ORF.append( np.zeros((npsr,npsr)) )

                ORF = np.array(ORF)
                ORFtot = np.zeros((mode_count,npsr,npsr)) # shouldn't be applying ORF to dmfreqs,
                                                          # but the projection of GW spec onto dmfreqs
                                                          # is defined as zero below.
                ORFtot[0::2] = ORF
                ORFtot[1::2] = ORF
            
            elif args.gwbTypeCorr == 'spharmAnis':
            
                ################################################
                # Reshaping freq-dependent anis coefficients,
                # and testing for power distribution physicality.
            
                orf_coeffs = orf_coeffs.reshape((tmp_nwins,
                                                ((args.LMAX+1)**2)-1))
                clm = np.array([[0.0]*((args.LMAX+1)**2)
                                for ii in range(tmp_nwins)])
                clm[:,0] = 2.0*np.sqrt(np.pi)

                if args.LMAX!=0:

                    for kk in range(tmp_nwins):
                        for ii in range(1,((args.LMAX+1)**2)):
                            clm[kk,ii] = orf_coeffs[kk,ii-1]   

                        if not args.noPhysPrior:
                            # Testing for physicality of power distribution.
                            if (utils.PhysPrior(clm[kk],harm_sky_vals) == 'Unphysical'):
                                return -np.inf

                ############################################################
                # Computing frequency-dependent overlap reduction functions.
        
                ORF=[]
                for ii in range(tmp_nwins): # number of frequency windows
                    for jj in range(len(corr_modefreqs[ii])): # number of frequencies in this window
                        ORF.append( sum(clm[ii,kk]*CorrCoeff[kk]
                                        for kk in range(len(CorrCoeff))) )
                if args.incDM:
                    for ii in range(nmodes_dm): 
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incEph:
                    for kk in range(3): # x,y,z
                        for ii in range(nmodes_eph):
                            ORF.append( np.zeros((npsr,npsr)) )

                if args.incClk and args.clkDesign:
                    for ii in range(nmodes_red): 
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incBand:
                    for kk in range(len(bands)-1):
                        for ii in range(nmodes_band):
                            ORF.append( np.zeros((npsr,npsr)) )

                ORF = np.array(ORF)
                ORFtot = np.zeros((mode_count,npsr,npsr)) # shouldn't be applying ORF to dmfreqs,
                                                          # but the projection of GW spec onto dmfreqs
                                                          # is defined as zero below.
                ORFtot[0::2] = ORF
                ORFtot[1::2] = ORF

            elif args.gwbTypeCorr == 'dipoleOrf':
            
                orf_coeffs = orf_coeffs.reshape((tmp_nwins,3))
                dipphi, dipcostheta, dipwgt = \
                  orf_coeffs[:,0], orf_coeffs[:,1], orf_coeffs[:,2]
                diptheta = np.arccos(dipcostheta)
                dipvec = np.array([np.sin(diptheta)*np.cos(dipphi),
                                   np.sin(diptheta)*np.sin(dipphi),
                                   np.cos(diptheta)]).T

                psrvec = np.array([np.sin(positions[:,1])*np.cos(positions[:,0]),
                                   np.sin(positions[:,1])*np.sin(positions[:,0]),
                                   np.cos(positions[:,1])]).T

                gammaDip = np.zeros((tmp_nwins,npsr,npsr))
                for kk in range(tmp_nwins):
                    for ii in range(npsr):
                        for jj in range(ii,npsr):
                            # dot product of psr and dipole position vectors
                            cpsra = np.dot(psrvec[ii,:],dipvec[kk,:])
                            cpsrb = np.dot(psrvec[jj,:],dipvec[kk,:])
                            # angular separation between pulsars
                            if ii==jj:
                                zetaab = 0.0
                            else:
                                zetaab = np.arccos(np.dot(psrvec[ii,:],psrvec[jj,:]))

                            # maximal-dipole orf expression from Anholm et al. (2009)
                            gammaDip[kk,ii,jj] = (3.0/8.0) * (cpsra+cpsrb) * \
                              ( np.cos(zetaab) - (4.0/3.0) - \
                                4.0*np.tan(zetaab/2.)**2.0*np.log(np.sin(zetaab/2.)) ) 
                           
                            gammaDip[kk,jj,ii] = gammaDip[kk,ii,jj]

                            if ii == jj:
                                # scaling for pulsar-term
                                gammaDip[kk,ii,jj] *= 2.0
                

                ############################################################
                # Computing frequency-dependent overlap reduction functions.
        
                ORF=[]
                for ii in range(tmp_nwins): # number of frequency windows
                    for jj in range(len(corr_modefreqs[ii])): # number of frequencies in this window
                        ORF.append( monoOrf + dipwgt[ii]*gammaDip[ii,:,:] )
                        
                if args.incDM:
                    for ii in range(nmodes_dm): 
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incEph:
                    for kk in range(3): # x,y,z
                        for ii in range(nmodes_eph): 
                            ORF.append( np.zeros((npsr,npsr)) )

                if args.incClk and args.clkDesign:
                    for ii in range(nmodes_red): 
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incBand:
                    for kk in range(len(bands)-1):
                        for ii in range(nmodes_band):
                            ORF.append( np.zeros((npsr,npsr)) )

                ORF = np.array(ORF)
                ORFtot = np.zeros((mode_count,npsr,npsr)) # shouldn't be applying ORF to dmfreqs,
                                                          # but the projection of GW spec onto dmfreqs
                                                          # is defined as zero below.
                ORFtot[0::2] = ORF
                ORFtot[1::2] = ORF

            elif args.gwbTypeCorr == 'custom':
            
                ############################################################
                # Computing frequency-dependent overlap reduction functions.
        
                ORF=[]
                for ii in range(nmodes_red): # number of frequencies
                    if np.atleast_3d(customOrf.T).shape[-1]>1:
                        ORF.append( customOrf[ii,:,:] )
                    else:
                        ORF.append( customOrf )
                        
                if args.incDM:
                    for ii in range(nmodes_dm): # number of frequencies
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incEph:
                    for kk in range(3): # x,y,z
                        for ii in range(nmodes_eph): # number of frequencies
                            ORF.append( np.zeros((npsr,npsr)) )

                if args.incClk and args.clkDesign:
                    for ii in range(nmodes_red): 
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incBand:
                    for kk in range(len(bands)-1):
                        for ii in range(nmodes_band):
                            ORF.append( np.zeros((npsr,npsr)) )

                ORF = np.array(ORF)
                ORFtot = np.zeros((mode_count,npsr,npsr)) # shouldn't be applying ORF to dmfreqs
                                                          # or ephemeris freqs, but the 
                                                          # projection of GW spec onto dmfreqs
                                                          # is defined as zero below.
                ORFtot[0::2] = ORF
                ORFtot[1::2] = ORF

            elif args.gwbTypeCorr == 'gwDisk':
            
                ################################################
                # Reshaping freq-dependent anis coefficients,
                # and testing for power distribution physicality.

                if hp is not None:
                    orf_coeffs = orf_coeffs.reshape((tmp_nwins,4))
                    diskphi, diskcostheta, diskradius, diskwgt = \
                      orf_coeffs[:,0], orf_coeffs[:,1], orf_coeffs[:,2], orf_coeffs[:,3]
                    disktheta = np.arccos(diskcostheta)
                    diskvec = np.array([np.sin(disktheta)*np.cos(diskphi),
                                        np.sin(disktheta)*np.sin(diskphi),
                                        np.cos(disktheta)]).T

                    gammaDisk = np.zeros((tmp_nwins,npsr,npsr))
                    for kk in range(tmp_nwins):
                        m = np.ones(hp.nside2npix(nside=32))
                        qd = hp.query_disc(nside=hp.npix2nside(len(m)),
                                           vec=diskvec[kk,:],
                                           radius=diskradius[kk])
                        m[qd] *= 10.0**diskwgt[kk]
                        m /= np.mean(m)
                        gammaDisk[kk,:,:] = pixAnis.orfFromMap_fast(psr_locs=positions,
                                                                    usermap=m, response=F_e)

                ############################################################
                # Computing frequency-dependent overlap reduction functions.
        
                ORF=[]
                for ii in range(tmp_nwins): # number of frequency windows
                    for jj in range(len(corr_modefreqs[ii])): # number of frequencies in this window
                        if hp is not None:
                            ORF.append( gammaDisk[ii,:,:] )
                        elif hp is None:
                            ORF.append( monoOrf )
                        
                if args.incDM:
                    for ii in range(nmodes_dm):
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incEph:
                    for kk in range(3): # x,y,z
                        for ii in range(nmodes_eph): 
                            ORF.append( np.zeros((npsr,npsr)) )

                if args.incClk and args.clkDesign:
                    for ii in range(nmodes_red): 
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incBand:
                    for kk in range(len(bands)-1):
                        for ii in range(nmodes_band):
                            ORF.append( np.zeros((npsr,npsr)) )

                ORF = np.array(ORF)
                ORFtot = np.zeros((mode_count,npsr,npsr)) # shouldn't be applying ORF to dmfreqs,
                                                          # but the projection of GW spec onto dmfreqs
                                                          # is defined as zero below.
                ORFtot[0::2] = ORF
                ORFtot[1::2] = ORF

            elif args.gwbTypeCorr == 'psrlocsVary':
            
                ################################################
                # Reshaping freq-dependent anis coefficients,
                # and testing for power distribution physicality.
            
                orf_coeffs = orf_coeffs.reshape((2,tmp_nwins*len(psr)))
                varyPhi = orf_coeffs[0,:].reshape((tmp_nwins,len(psr)))
                varyTheta = np.arccos(orf_coeffs[1,:]).reshape((tmp_nwins,len(psr)))

                ############################################################
                # Computing frequency-dependent overlap reduction functions.
        
                ORF=[]
                for ii in range(tmp_nwins): # number of frequency windows
                    varyLocs = np.zeros((len(psr),2))
                    varyLocs[:,0] = varyPhi[ii,:]
                    varyLocs[:,1] = varyTheta[ii,:]
                    varyLocs[0,:] = psr[0].psr_locs[0], np.pi/2. - psr[0].psr_locs[1]
                    monoOrf = 2.0*np.sqrt(np.pi)*anis.CorrBasis(varyLocs,0)[0]
                    for jj in range(len(corr_modefreqs[ii])): # number of frequencies in this window
                        ORF.append( monoOrf )
                        
                if args.incDM:
                    for ii in range(nmodes_dm): 
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incEph:
                    for kk in range(3): # x,y,z
                        for ii in range(nmodes_eph): 
                            ORF.append( np.zeros((npsr,npsr)) )

                if args.incClk and args.clkDesign:
                    for ii in range(nmodes_red): 
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incBand:
                    for kk in range(len(bands)-1):
                        for ii in range(nmodes_band):
                            ORF.append( np.zeros((npsr,npsr)) )

                ORF = np.array(ORF)
                ORFtot = np.zeros((mode_count,npsr,npsr)) # shouldn't be applying ORF to dmfreqs,
                                                          # but the projection of GW spec onto dmfreqs
                                                          # is defined as zero below.
                ORFtot[0::2] = ORF
                ORFtot[1::2] = ORF
                
            elif args.gwbTypeCorr == 'clock':

                ORF=[]
                for ii in range(tmp_nwins): # number of frequency windows
                    for jj in range(len(corr_modefreqs[ii])): # number of frequencies in this window
                        ORF.append( np.ones((npsr,npsr)) + 1e-5*np.diag(np.ones(npsr)) ) # clock signal is completely correlated
                        
                if args.incDM:
                    for ii in range(nmodes_dm): 
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incEph:
                    for kk in range(3): # x,y,z
                        for ii in range(nmodes_eph): 
                            ORF.append( np.zeros((npsr,npsr)) )

                if args.incClk and args.clkDesign:
                    for ii in range(nmodes_red): 
                        ORF.append( np.zeros((npsr,npsr)) )

                if args.incBand:
                    for kk in range(len(bands)-1):
                        for ii in range(nmodes_band):
                            ORF.append( np.zeros((npsr,npsr)) )

                ORF = np.array(ORF)
                ORFtot = np.zeros((mode_count,npsr,npsr)) # shouldn't be applying ORF to dmfreqs,
                                                          # but the projection of GW spec onto dmfreqs
                                                          # is defined as zero below.
                ORFtot[0::2] = ORF
                ORFtot[1::2] = ORF
            

        if args.incGWline:
    
            gwline_orf = np.zeros((npsr,npsr))

            Fp = np.zeros(npsr)
            Fc = np.zeros(npsr)
            for ii in range(npsr):
                Fp[ii], Fc[ii] = utils.fplus_fcross(psr[ii], theta_gwline, phi_gwline)
        
            for ii in range(npsr):
                for jj in range(ii,npsr):
                    gwline_orf[ii,jj] = (3.0/(8.0*np.pi)) * (Fp[ii]*Fp[jj] + Fc[ii]*Fc[jj])
                    gwline_orf[jj,ii] = gwline_orf[ii,jj]

                    if ii == jj:
                        # scaling for pulsar-term
                        gwline_orf[ii,jj] *= 2.0
        
        ################################################
        # parameterize intrinsic red noise as power law
    
        Tspan = (1/fqs_red[0])*86400.0

        # parameterize intrinsic red-noise and DM-variations
        kappa = []
        for ii in range(npsr):
        
            # Construct red noise signal
            if args.fixRed:
                Ared_tmp = np.max([psr[ii].Redamp, psr[ii].parRedamp])
                gam_red_tmp = np.max([psr[ii].Redind, psr[ii].parRedind])

                red_kappa_tmp = np.log10( Ared_tmp**2/12/np.pi**2 * \
                                    f1yr**(gam_red_tmp-3) * \
                                    (fqs_red/86400.0)**(-gam_red_tmp)/Tspan )
            
            if not args.fixRed:
                if args.redSpecModel == 'powerlaw':
                    Ared_tmp = Ared[ii]
                    gam_red_tmp = gam_red[ii]
                    
                    red_kappa_tmp = np.log10( Ared_tmp**2/12/np.pi**2 * \
                                            f1yr**(gam_red_tmp-3) * \
                                            (fqs_red/86400.0)**(-gam_red_tmp)/Tspan )
                elif args.redSpecModel == 'spectrum':
                    red_kappa_tmp = np.log10( 10.0**(2.0*red_spec[ii,:]) / Tspan)

            # Construct DM-variations signal (if appropriate)
            if args.incDM:
                if args.fixDM:
                    Adm_tmp = np.max([psr[ii].DMamp, psr[ii].parDMamp])
                    gam_dm_tmp = np.max([psr[ii].DMind, psr[ii].parDMind])
                    
                    dm_kappa_tmp = np.log10( Adm_tmp**2/12/np.pi**2 * \
                                            f1yr**(gam_dm_tmp-3) * \
                                            (fqs_dm/86400.0)**(-gam_dm_tmp)/Tspan )

                if not args.fixDM:
                    if args.dmSpecModel == 'powerlaw':
                        Adm_tmp = Adm[ii]
                        gam_dm_tmp = gam_dm[ii]
                    
                        dm_kappa_tmp = np.log10( Adm_tmp**2/12/np.pi**2 * \
                                                f1yr**(gam_dm_tmp-3) * \
                                                (fqs_dm/86400.0)**(-gam_dm_tmp)/Tspan )
                    elif args.dmSpecModel == 'spectrum':
                        dm_kappa_tmp = np.log10( 10.0**(2.0*dm_spec[ii,:]) / Tspan)

            if not args.incDM:
                dm_kappa_tmp = np.array([])

            if args.incEph:
                eph_padding = np.zeros(3*nmodes_eph)
            elif not args.incEph:
                eph_padding = np.array([])

            if args.incClk and args.clkDesign:
                clk_padding = np.zeros(nmodes_red)
            else:
                clk_padding = np.array([])

            if args.incBand:
                band_padding = np.zeros((len(bands)-1)*nmodes_band)
            elif not args.incBand:
                band_padding = np.array([])

            # Now create total red signal for each pulsar
            kappa.append(np.concatenate((10**red_kappa_tmp,
                                         10**dm_kappa_tmp,
                                         eph_padding,
                                         clk_padding,
                                         band_padding)))
    
                
        ###################################
        # construct elements of sigma array
    
        sigdiag = []
        if args.incGWB:

            if args.gwbSpecModel == 'powerlaw':
                rho = np.log10(Agwb**2/12/np.pi**2 * \
                            f1yr**(gam_gwb-3) * \
                            (fqs_red/86400.0)**(-gam_gwb)/Tspan)
            elif args.gwbSpecModel == 'spectrum':
                if args.gwbPrior != 'gaussProc':
                    rho = np.log10( 10.0**(2.0*rho_spec) / Tspan )
                elif args.gwbPrior == 'gaussProc':
                    if gwb_popparam == 'starsecc':
                        rho_pred = np.zeros((len(fqs_red),2))
                        for ii,freq in enumerate(fqs_red):
                            mu_pred, cov_pred = gp[ii].predict(gppkl[ii].y, [env_param])
                            if np.diag(cov_pred) < 0.0:
                                rho_pred[ii,0], rho_pred[ii,1] = mu_pred, 1e-5 * mu_pred
                            else:
                                rho_pred[ii,0], rho_pred[ii,1] = mu_pred, np.sqrt(np.diag(cov_pred))

                        # transforming from zero-mean unit-variance variable to rho
                        rho = 2.0*np.log10(Agwb) - np.log10(12.0 * np.pi**2.0 * (fqs_red/86400.0)**3.0 * Tspan) + \
                          rho_spec*rho_pred[:,1] + rho_pred[:,0]
                    else: 
                        rho_pred = np.zeros((len(fqs_red),2))
                        for ii,freq in enumerate(fqs_red):
                            mu_pred, cov_pred = gp[ii].predict(gppkl[ii].y, env_param)
                            if np.diag(cov_pred) < 0.0:
                                rho_pred[ii,0], rho_pred[ii,1] = mu_pred, 1e-5 * mu_pred
                            else:
                                rho_pred[ii,0], rho_pred[ii,1] = mu_pred, np.sqrt(np.diag(cov_pred))

                        # transforming from zero-mean unit-variance variable to rho
                        rho = 2.0*np.log10(Agwb) - np.log10(Tspan) + rho_spec*rho_pred[:,1] + rho_pred[:,0]
                    
            elif args.gwbSpecModel == 'turnover':
                rho = np.log10(Agwb**2/12/np.pi**2 * \
                            f1yr**(13.0/3.0-3.0) * \
                            (fqs_red/86400.0)**(-13.0/3.0) / \
                            (1.0+(fbend*86400.0/fqs_red)**kappaturn)/Tspan)
            elif args.gwbSpecModel == 'gpEnvInterp':
                #### CURRENTLY OUT OF USAGE ####
                '''
                hc_pred = np.zeros((len(fqs_red),2))
                for ii,freq in enumerate(fqs_red):
                    hc_pred[ii,0], mse = gp[ii].gp.predict(ecc, eval_MSE=True)
                    hc_pred[ii,1] = np.sqrt(mse)

                if not args.incCosVar:
                    hc = Agwb * hc_pred[:,0]
                elif args.incCosVar:
                    hc = Agwb * (hc_pred[:,0] + np.random.normal(0.0,1.0,len(fqs_red)) * hc_pred[:,1])

                rho = np.log10( hc**2 / (12.0*np.pi**2.0) / (fqs_red/86400.0)**3.0 / Tspan )

                '''


            if args.incDM:
                dm_padding = np.zeros(nmodes_dm)
            elif not args.incDM:
                dm_padding = np.array([])
                
            if args.incEph:
                eph_padding = np.zeros(3*nmodes_eph)
            elif not args.incEph:
                eph_padding = np.array([])

            if args.incClk and args.clkDesign:
                clk_padding = np.zeros(nmodes_red)
            else:
                clk_padding = np.array([])

            if args.incBand:
                band_padding = np.zeros((len(bands)-1)*nmodes_band)
            elif not args.incBand:
                band_padding = np.array([])
                    
            gwbspec = np.concatenate( (10**rho, dm_padding,
                                       eph_padding, clk_padding,
                                       band_padding) )
            
            if args.incCorr:
                sig_gwboffdiag = []

        if args.incGWline:
        
            rho_line = np.zeros(nmodes_red)
            idx = np.argmin(np.abs(fqs_red/86400.0 - freq_gwline))
            rho_line[idx] = 10.0**(2.0*spec_gwline) / Tspan

            if args.incDM:
                dm_padding = np.zeros(nmodes_dm)
            elif not args.incDM:
                dm_padding = np.array([])
            
            if args.incEph:
                eph_padding = np.zeros(3*nmodes_eph)
            elif not args.incEph:
                eph_padding = np.array([])

            if args.incClk and args.clkDesign:
                clk_padding = np.zeros(nmodes_red)
            else:
                clk_padding = np.array([])

            if args.incBand:
                band_padding = np.zeros((len(bands)-1)*nmodes_band)
            elif not args.incBand:
                band_padding = np.array([])
            
            gwline_spec = np.concatenate( (rho_line,
                                           dm_padding,
                                           eph_padding,
                                           clk_padding,
                                           band_padding) )
           
            if args.incCorr:
                sig_gwlineoffdiag = []


        if args.incClk:

            if args.clkSpecModel == 'powerlaw':
                kappa_clk = np.log10(Aclk**2/12/np.pi**2 * \
                            f1yr**(gam_clk-3) * \
                            (fqs_red/86400.0)**(-gam_clk)/Tspan)
            elif args.clkSpecModel == 'spectrum':
                kappa_clk = np.log10( 10.0**(2.0*clk_spec) / Tspan )


            if args.incDM:
                dm_padding = np.zeros(nmodes_dm)
            elif not args.incDM:
                dm_padding = np.array([])
                
            if args.incEph:
                eph_padding = np.zeros(3*nmodes_eph)
            elif not args.incEph:
                eph_padding = np.array([])

            if args.incBand:
                band_padding = np.zeros((len(bands)-1)*nmodes_band)
            elif not args.incBand:
                band_padding = np.array([])

            if args.incClk and args.clkDesign:
                clk_padding = np.zeros(nmodes_red)
                clkspec = np.concatenate( (clk_padding,
                                           dm_padding,
                                           eph_padding,
                                           10**kappa_clk,
                                           band_padding) )
            else:
                clk_padding = np.array([])
                clkspec = np.concatenate( (10**kappa_clk,
                                           dm_padding,
                                           eph_padding,
                                           clk_padding,
                                           band_padding) )

            if args.incCorr:
                sig_clkoffdiag = []


        if args.incCm:

            if args.cmSpecModel == 'powerlaw':
                kappa_cm = np.log10(Acm**2/12/np.pi**2 * \
                            f1yr**(gam_cm-3) * \
                            (fqs_red/86400.0)**(-gam_cm)/Tspan)
            elif args.cmSpecModel == 'spectrum':
                kappa_cm = np.log10( 10.0**(2.0*cm_spec) / Tspan )


            if args.incDM:
                dm_padding = np.zeros(nmodes_dm)
            elif not args.incDM:
                dm_padding = np.array([])
                
            if args.incEph:
                eph_padding = np.zeros(3*nmodes_eph)
            elif not args.incEph:
                eph_padding = np.array([])

            if args.incClk and args.clkDesign:
                clk_padding = np.zeros(nmodes_red)
            else:
                clk_padding = np.array([])

            if args.incBand:
                band_padding = np.zeros((len(bands)-1)*nmodes_band)
            elif not args.incBand:
                band_padding = np.array([])
                    
            cmspec = np.concatenate( (10**kappa_cm, dm_padding,
                                      eph_padding, clk_padding,
                                      band_padding) )
    
                
        if args.incEph:

            if args.ephSpecModel == 'powerlaw':
                kappa_ephx = np.log10(Aephx**2/12/np.pi**2 * \
                                      f1yr**(gam_ephx-3) * \
                                      (fqs_eph/86400.0)**(-gam_ephx)/Tspan)
                kappa_ephy = np.log10(Aephy**2/12/np.pi**2 * \
                                      f1yr**(gam_ephy-3) * \
                                      (fqs_eph/86400.0)**(-gam_ephy)/Tspan)
                kappa_ephz = np.log10(Aephz**2/12/np.pi**2 * \
                                      f1yr**(gam_ephz-3) * \
                                      (fqs_eph/86400.0)**(-gam_ephz)/Tspan)
            elif args.ephSpecModel == 'spectrum':
                kappa_ephx = np.log10( 10.0**(2.0*eph_spec[0,:]))
                kappa_ephy = np.log10( 10.0**(2.0*eph_spec[1,:]))
                kappa_ephz = np.log10( 10.0**(2.0*eph_spec[2,:]))

            red_padding = np.zeros(nmodes_red)
            if args.incDM:
                dm_padding = np.zeros(nmodes_dm)
            elif not args.incDM:
                dm_padding = np.array([])

            if args.incClk and args.clkDesign:
                clk_padding = np.zeros(nmodes_red)
            else:
                clk_padding = np.array([])

            if args.incBand:
                band_padding = np.zeros((len(bands)-1)*nmodes_band)
            elif not args.incBand:
                band_padding = np.array([])

            eph_kappa = np.concatenate( (red_padding, dm_padding,
                                         10**kappa_ephx, 10**kappa_ephy,
                                         10**kappa_ephz, clk_padding,
                                         band_padding) )

        if args.incBand:

            kappa_band = np.array([])
            if args.bandSpecModel == 'powerlaw':
                for ii in range(len(bands)-1):
                    kappa_band = np.append(kappa_band,
                                           np.log10(Aband[ii]**2/12/np.pi**2 * \
                                                    f1yr**(gam_band[ii]-3) * \
                                                    (fqs_band/86400.0)**(-gam_band[ii])/Tspan))
               
            elif args.bandSpecModel == 'spectrum':
                for ii in range(len(bands)-1):
                    kappa_band = np.append(kappa_band,
                                           np.log10( 10.0**(2.0*band_spec[ii,:])))
                
            red_padding = np.zeros(nmodes_red)
            if args.incDM:
                dm_padding = np.zeros(nmodes_dm)
            elif not args.incDM:
                dm_padding = np.array([])

            if args.incEph:
                eph_padding = np.zeros(3*nmodes_eph)
            elif not args.incEph:
                eph_padding = np.array([])

            if args.incClk and args.clkDesign:
                clk_padding = np.zeros(nmodes_red)
            else:
                clk_padding = np.array([])

            band_kappa = np.concatenate( (red_padding, dm_padding,
                                         eph_padding, clk_padding,
                                         10.0**kappa_band) )
        

        for ii in range(npsr):
            tot = np.zeros(mode_count)

            # diagonal terms
            tot[0::2] = kappa[ii]
            tot[1::2] = kappa[ii] 

            if args.incGWB:
            
                if args.incCorr and gwb_modindex==1:
                
                    offdiag = np.zeros(mode_count)

                    # off diagonal terms
                    offdiag[0::2] = gwbspec
                    offdiag[1::2] = gwbspec

                    # diagonal terms
                    tot[0::2] += ORF[:,ii,ii]*gwbspec
                    tot[1::2] += ORF[:,ii,ii]*gwbspec

                    sig_gwboffdiag.append(offdiag)
                
                if not args.incCorr or gwb_modindex==0:
                    
                    # diagonal terms
                    tot[0::2] += gwbspec
                    tot[1::2] += gwbspec

            if args.incGWline:
            
                if args.incCorr:
                
                    offdiag = np.zeros(mode_count)

                    # off diagonal terms
                    offdiag[0::2] = gwline_spec
                    offdiag[1::2] = gwline_spec

                    # diagonal terms
                    tot[0::2] += gwline_orf[ii,ii]*gwline_spec
                    tot[1::2] += gwline_orf[ii,ii]*gwline_spec

                    sig_gwlineoffdiag.append(offdiag)
                
                if not args.incCorr:
                
                    # diagonal terms
                    tot[0::2] += gwline_spec
                    tot[1::2] += gwline_spec

            if args.incClk:
            
                if args.incCorr:
                
                    offdiag = np.zeros(mode_count)

                    # off diagonal terms
                    # [clock errors are full correlated]
                    offdiag[0::2] = clkspec
                    offdiag[1::2] = clkspec

                    # diagonal terms
                    if args.clkDesign:
                        tot[0::2] += (1.0 + 1e-5)*clkspec
                        tot[1::2] += (1.0 + 1e-5)*clkspec
                    elif not args.clkDesign:
                        tot[0::2] += clkspec
                        tot[1::2] += clkspec

                    sig_clkoffdiag.append(offdiag)
                
                if not args.incCorr:
                
                    # diagonal terms
                    tot[0::2] += clkspec
                    tot[1::2] += clkspec

            if args.incCm:

                # diagonal terms
                tot[0::2] += cmspec
                tot[1::2] += cmspec

            if args.incEph:

                # diagonal terms
                tot[0::2] += eph_kappa
                tot[1::2] += eph_kappa

            if args.incBand:

                # diagonal terms
                tot[0::2] += band_kappa
                tot[1::2] += band_kappa
                
                
            # fill in lists of arrays
            sigdiag.append(tot)


        ###############################################
        # Computing Phi and Sigma matrices without GWB
    
        if not args.incGWB and not args.incGWline and not args.incClk:

            for ii,p in enumerate(psr):
            
                # compute Phi inverse 
                red_phi = np.diag(1./sigdiag[ii])
                logdet_Phi = np.sum(np.log(sigdiag[ii]))

                # now fill in real covariance matrix
                Phi = np.zeros( TtNT_tmp[ii].shape ) 
                for kk in range(0,mode_count):
                    Phi[kk+p.Gc.shape[1],kk+p.Gc.shape[1]] = red_phi[kk,kk]

                # symmeterize Phi
                Phi = Phi + Phi.T - np.diag(np.diag(Phi))
    
                # compute sigma
                Sigma = TtNT_tmp[ii] + Phi
                
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
        

        if args.incGWB or args.incGWline or args.incClk:
    
            if not args.incCorr or (args.incCorr and args.incGWB and gwb_modindex==0
                                    and not args.incGWline and not args.incClk):
            
                for ii,p in enumerate(psr):
            
                    # compute Phi inverse 
                    red_phi = np.diag(1./sigdiag[ii])
                    logdet_Phi = np.sum(np.log(sigdiag[ii]))

                    # now fill in real covariance matrix
                    Phi = np.zeros( TtNT_tmp[ii].shape ) 
                    for kk in range(0,mode_count):
                        Phi[kk+p.Gc.shape[1],kk+p.Gc.shape[1]] = red_phi[kk,kk]

                    # symmeterize Phi
                    Phi = Phi + Phi.T - np.diag(np.diag(Phi))
    
                    # compute sigma
                    Sigma = TtNT_tmp[ii] + Phi
                    
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

            elif args.incCorr:
        
                #####################
                # compute Phi matrix

                smallMatrix = np.zeros((mode_count, npsr, npsr))
                for ii in range(npsr):
                    for jj in range(ii,npsr):

                        if ii == jj:
                            smallMatrix[:,ii,jj] = sigdiag[jj] 
                        else:
                            if args.incGWB and gwb_modindex==1:
                                smallMatrix[:,ii,jj] += ORFtot[:,ii,jj] * sig_gwboffdiag[jj]
                            if args.incGWline:
                                smallMatrix[:,ii,jj] += sig_gwlineoffdiag[jj]
                            if args.incClk:
                                smallMatrix[:,ii,jj] += sig_clkoffdiag[jj]
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


                bigTtNT = sl.block_diag(*TtNT_tmp)
                Phi = np.zeros_like( bigTtNT )
    
                # now fill in real covariance matrix
                ind = [0]
                ind = np.append(ind,np.cumsum([TtNT_tmp[ii].shape[0]
                                            for ii in range(npsr)]))
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
                        if args.sparse_cholesky:
                            sparseSigma = sps.csc_matrix(Sigma)
                            cf = sks.cholesky(sparseSigma)
                            expval2 = cf(dtmp)
                            logdet_Sigma = cf.logdet()
                        else:
                            cf = sl.cho_factor(Sigma)
                            expval2 = sl.cho_solve(cf, dtmp)
                            logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

                    except np.linalg.LinAlgError or sks.CholmodError:
                    
                        print 'Cholesky Decomposition Failed second time!! Breaking...'
                        return -np.inf


                    logLike = -0.5 * (logdet_Phi + logdet_Sigma) + \
                      0.5 * (np.dot(dtmp, expval2)) + \
                      loglike1_tmp

    

    ################################################
    # Multiplying likelihood by appropriate Jacobian
    
    if args.incGWB:
        ### powerlaw spectral model ###
        if args.gwbSpecModel == 'powerlaw':
            if args.gwbPrior == 'uniform':
                priorfac_gwb = np.log(Agwb * np.log(10.0))
            elif args.gwbPrior == 'loguniform':
                priorfac_gwb = 0.0
            elif args.gwbPrior == 'sesana':
                mu = -15.0
                sig = 0.22
                priorfac_gwb = np.log( np.exp( -0.5 * (np.log10(Agwb) - mu)**2.0 / sig**2.0)
                                    / np.sqrt(2.0*np.pi*sig**2.0) / np.log(10.0) )
            elif args.gwbPrior == 'mcwilliams':
                mu = -14.4
                sig = 0.26
                priorfac_gwb = np.log( np.exp( -0.5 * (np.log10(Agwb) - mu)**2.0 / sig**2.0)
                                    / np.sqrt(2.0*np.pi*sig**2.0) / np.log(10.0) )
                
        ### free spectral model ###
        elif args.gwbSpecModel == 'spectrum':
            if args.gwbPrior == 'uniform':
                priorfac_gwb = np.sum(np.log(10.0**rho_spec * np.log(10.0)))
            elif args.gwbPrior == 'loguniform':
                priorfac_gwb = 0.0
            elif args.gwbPrior == 'gaussProc':
                '''
                hc_pred = np.zeros((len(fqs_red),2))
                for ii,freq in enumerate(fqs_red):
                    hc_pred[ii,0], mse = gp[ii].predict(ecc, eval_MSE=True)
                    hc_pred[ii,1] = np.sqrt(mse)
                psd_mean = Agwb**2.0 * hc_pred[:,0]**2.0 / (12.0*np.pi**2.0) / (fqs_red/86400.0)**3.0 / Tspan
                psd_std = 2.0 * psd_mean * hc_pred[:,1] / hc_pred[:,0]

                priorfac_gwb = np.sum( np.log(2.0 * 10.0**rho * np.log(10.0))
                                       - 0.5*np.log(2.0 * np.pi * psd_std**2.0)
                                       - 0.5*(10.0**rho - psd_mean)**2.0 / psd_std**2.0  )
                '''
                priorfac_gwb = np.sum( - 0.5*np.log(2.0 * np.pi) - 0.5 * rho_spec**2.0)

                ### adding hyper prior on strain amplitude ###
                if args.gwbHyperPrior == 'uniform':
                    priorfac_gwb += np.log(Agwb * np.log(10.0))
                elif args.gwbHyperPrior == 'loguniform':
                    priorfac_gwb += 0.0
                elif args.gwbHyperPrior == 'sesana':
                    mu = -15.0
                    sig = 0.22
                    priorfac_gwb += np.log( np.exp( -0.5 * (np.log10(Agwb) - mu)**2.0 / sig**2.0)
                                        / np.sqrt(2.0*np.pi*sig**2.0) / np.log(10.0) )
                elif args.gwbHyperPrior == 'mcwilliams':
                    mu = -14.4
                    sig = 0.26
                    priorfac_gwb += np.log( np.exp( -0.5 * (np.log10(Agwb) - mu)**2.0 / sig**2.0)
                                        / np.sqrt(2.0*np.pi*sig**2.0) / np.log(10.0) )
                    
                
        ### turnover spectral model ###
        elif args.gwbSpecModel == 'turnover':
            if args.gwbPrior == 'uniform':
                priorfac_gwb = np.log(Agwb * np.log(10.0))
            elif args.gwbPrior == 'loguniform':
                priorfac_gwb = 0.0
            elif args.gwbPrior == 'sesana':
                mu = -15.0
                sig = 0.22
                priorfac_gwb = np.log( np.exp( -0.5 * (np.log10(Agwb) - mu)**2.0 / sig**2.0)
                                    / np.sqrt(2.0*np.pi*sig**2.0) / np.log(10.0) )
            elif args.gwbPrior == 'mcwilliams':
                mu = -14.4
                sig = 0.26
                priorfac_gwb = np.log( np.exp( -0.5 * (np.log10(Agwb) - mu)**2.0 / sig**2.0)
                                    / np.sqrt(2.0*np.pi*sig**2.0) / np.log(10.0) )
            

        ### gp interpolation spectral model ###
        elif args.gwbSpecModel == 'gpEnvInterp':
            #### CURRENTLY OUT OF USAGE ####
            '''
            if args.gwbPrior == 'uniform':
                priorfac_gwb = np.log(Agwb * np.log(10.0))
            elif args.gwbPrior == 'loguniform':
                priorfac_gwb = 0.0
            elif args.gwbPrior == 'sesana':
                mu = -15.0
                sig = 0.22
                priorfac_gwb = np.log( np.exp( -0.5 * (np.log10(Agwb) - mu)**2.0 / sig**2.0)
                                    / np.sqrt(2.0*np.pi*sig**2.0) / np.log(10.0) )
            elif args.gwbPrior == 'mcwilliams':
                mu = -14.4
                sig = 0.26
                priorfac_gwb = np.log( np.exp( -0.5 * (np.log10(Agwb) - mu)**2.0 / sig**2.0)
                                    / np.sqrt(2.0*np.pi*sig**2.0) / np.log(10.0) )
            '''
                
    elif not args.incGWB:
        priorfac_gwb = 0.0
         

    if args.incGWline:
        if args.gwlinePrior == 'uniform':
            priorfac_gwline = np.log(10.0**spec_gwline * np.log(10.0))
        elif args.gwlinePrior == 'loguniform':
            priorfac_gwline = 0.0
    elif not args.incGWline:
        priorfac_gwline = 0.0
        

    if not args.fixRed:
        ### powerlaw spectral model ###
        if args.redSpecModel == 'powerlaw':
            if args.redPrior == 'uniform':
                priorfac_red = np.sum(np.log(Ared * np.log(10.0)))
            elif args.redPrior == 'loguniform':
                priorfac_red = 0.0
        ### free spectral model ###
        elif args.redSpecModel == 'spectrum':
            if args.redPrior == 'uniform':
                priorfac_red = np.sum(np.log(10.0**red_spec * np.log(10.0)))
            elif args.redPrior == 'loguniform':
                priorfac_red = 0.0
    elif args.fixRed:
        priorfac_red = 0.0
        

    if args.incDM and not args.fixDM:
        ### powerlaw spectral model ###
        if args.dmSpecModel == 'powerlaw':
            if args.dmPrior == 'uniform':
                priorfac_dm = np.sum(np.log(Adm * np.log(10.0)))
            elif args.dmPrior == 'loguniform':
                priorfac_dm = 0.0
        ### free spectral model ###
        elif args.dmSpecModel == 'spectrum':
            if args.dmPrior == 'uniform':
                priorfac_dm = np.sum(np.log(10.0**dm_spec * np.log(10.0)))
            elif args.dmPrior == 'loguniform':
                priorfac_dm = 0.0
    elif args.fixDM or not args.incDM:
        priorfac_dm = 0.0

        
    priorfac_band = 0.0
    if args.incBand:
        ### powerlaw spectral model ###
        if args.bandSpecModel == 'powerlaw':
            if args.bandPrior == 'uniform':
                for ii in range(len(bands)-1):
                    priorfac_band += np.log(Aband[ii] * np.log(10.0)) 
            elif args.bandPrior == 'loguniform':
                priorfac_band = 0.0
        ### free spectral model ###
        elif args.bandSpecModel == 'spectrum':
            if args.bandPrior == 'uniform':
                for ii in range(len(bands)-1):
                    priorfac_band += np.sum(np.log(10.0**band_spec[ii,:] * np.log(10.0)))
            elif args.bandPrior == 'loguniform':
                priorfac_band = 0.0
    elif not args.incBand:
        priorfac_band = 0.0

        
    if args.incClk:
        ### powerlaw spectral model ###
        if args.clkSpecModel == 'powerlaw':
            if args.clkPrior == 'uniform':
                priorfac_clk = np.log(Aclk * np.log(10.0)) 
            elif args.clkPrior == 'loguniform':
                priorfac_clk = 0.0
        ### free spectral model ###
        elif args.clkSpecModel == 'spectrum':
            if args.clkPrior == 'uniform':
                priorfac_clk = np.sum(np.log(10.0**clk_spec * np.log(10.0)))
            elif args.clkPrior == 'loguniform':
                priorfac_clk = 0.0
    elif not args.incClk:
        priorfac_clk = 0.0


    if args.incCm:
        ### powerlaw spectral model ###
        if args.cmSpecModel == 'powerlaw':
            if args.cmPrior == 'uniform':
                priorfac_cm = np.log(Acm * np.log(10.0)) 
            elif args.cmPrior == 'loguniform':
                priorfac_cm = 0.0
        ### free spectral model ###
        elif args.cmSpecModel == 'spectrum':
            if args.cmPrior == 'uniform':
                priorfac_cm = np.sum(np.log(10.0**cm_spec * np.log(10.0)))
            elif args.cmPrior == 'loguniform':
                priorfac_cm = 0.0
    elif not args.incCm:
        priorfac_cm = 0.0
        

    if args.incEph:
        ### powerlaw spectral model ###
        if args.ephSpecModel == 'powerlaw':
            if args.ephPrior == 'uniform':
                priorfac_eph = np.log(Aephx * np.log(10.0)) + \
                  np.log(Aephy * np.log(10.0)) + \
                  np.log(Aephz * np.log(10.0)) 
            elif args.ephPrior == 'loguniform':
                priorfac_eph = 0.0
        ### free spectral model ###
        elif args.ephSpecModel == 'spectrum':
            if args.ephPrior == 'uniform':
                priorfac_eph = np.sum(np.log(10.0**eph_spec[0,:] * np.log(10.0))) + \
                  np.sum(np.log(10.0**eph_spec[1,:] * np.log(10.0))) + \
                  np.sum(np.log(10.0**eph_spec[2,:] * np.log(10.0)))
            elif args.ephPrior == 'loguniform':
                priorfac_eph = 0.0
    elif not args.incEph:
        priorfac_eph = 0.0


    priorfac_corr = 0.0
    if args.incGWB and args.incCorr:
        if args.gwbTypeCorr == 'modelIndep':
            jacobian = np.zeros((npairs,npairs))
            if args.corrJacobian == 'full':
                ct = 0
                for ii in range(len(phi_els)):
                    for jj in range(len(phi_els[ii])):
                        
                        dummy_utriang = upper_triang[jj:,ii+1].copy()
                        dummy_utriang[0] = -np.sin(phi_els[ii][jj]) * dummy_utriang[0] / np.cos(phi_els[ii][jj])
                        dummy_utriang[1:] = np.cos(phi_els[ii][jj]) * dummy_utriang[1:] / np.sin(phi_els[ii][jj])
    
                        dummy_utriang = np.append(np.zeros(len(upper_triang[:jj,ii+1])), dummy_utriang)
    
                        deriv = np.zeros_like(upper_triang)
                        deriv[:,ii+1] = np.dot(upper_triang.T, dummy_utriang)
                        deriv = deriv + deriv.T

                        jacobian[:,ct] = deriv[np.triu_indices(npsr,k=1)] 
                        ct += 1

                tmp = np.linalg.slogdet(jacobian)
                priorfac_corr = 0.5*tmp[1] 
            elif args.corrJacobian == 'simple':
                priorfac_corr = np.sum(np.log(np.abs(np.array([-np.sin(phi_els[ii][0])
                                                               for ii in range(len(phi_els))]))))
            else:
                priorfac_corr = 0.0

        ### Gaussian prior on modeled psr positions ###
        ### Currently assumes only one frequency window ###
        elif args.gwbTypeCorr == 'psrlocsVary':
            priorfac_corr = 0.0
            for ii,p in enumerate(psr):
                if args.psrlocsPrior == 'normal':
                    sig = 0.5
                    priorfac_corr += np.log( np.exp( -0.5 * (varyLocs[ii,0] - p.psr_locs[0])**2.0 / sig**2.0) / \
                                    np.sqrt(2.0*np.pi*sig**2.0) ) + \
                                    np.log( np.exp( -0.5 * (varyLocs[ii,1] - np.pi/2. + p.psr_locs[1])**2.0 / sig**2.0) / \
                                    np.sqrt(2.0*np.pi*sig**2.0) )
                elif args.psrlocsPrior == 'uniform':
                    if np.abs(varyLocs[ii,0] - p.psr_locs[0]) <= 1.0 and \
                      np.abs(varyLocs[ii,1] - np.pi/2. + p.psr_locs[1]) <= 1.0:
                        priorfac_corr += 0.0
                    else:
                        priorfac_corr += -np.inf
                else:
                    priorfac_corr = 0.0
        else:
            priorfac_corr = 0.0
    elif not args.incGWB and not args.incCorr:
        priorfac_corr = 0.0

    ### Reweighting corr-vs-uncorr GWB models ###
    ### to ensure proper mixing ###
    priorfac_gwbmod = 0.0
    if args.incGWB and args.incCorr and args.gwbModelSelect:
        if gwb_modindex == 0:
            priorfac_gwbmod = ( np.log( args.gwbCorrModWgt / (1.0 + args.gwbCorrModWgt) )
                                - np.log(1.0/2.0) )
        elif gwb_modindex == 1:
            priorfac_gwbmod = ( np.log( 1.0 / (1.0 + args.gwbCorrModWgt) )
                                - np.log(1.0/2.0) )
        
    ### Jacobian and prior on cgw properties ###
    if args.det_signal:
        if args.cgw_search:
            ### uniform prior ###
            if args.cgwPrior == 'uniform':
                priorfac_detsig = np.log(hstrain * np.log(10.0))
            elif args.cgwPrior == 'loguniform':
                priorfac_detsig = 0.0
            ### pulsar distance prior ###
            if args.psrTerm:
                for ii, p in enumerate(psr):
                    mu = p.h5Obj['pdist'].value
                    sig = p.h5Obj['pdistErr'].value
                    priorfac_detsig += \
                      np.log( np.exp( -0.5 * (psrdists[ii] - mu)**2.0 / sig**2.0) / \
                              np.sqrt(2.0*np.pi*sig**2.0) )
        else:
            priorfac_detsig = 0.0
    elif not args.det_signal:
        priorfac_detsig = 0.0
        

    #####################################
    # Finally, return the log-likelihood
    
    return (1.0/args.softParam) * (logLike + priorfac_gwb + priorfac_gwbmod + priorfac_gwline + \
                                   priorfac_red + priorfac_dm + priorfac_clk + \
                                   priorfac_cm + priorfac_eph + priorfac_band + \
                                   priorfac_corr + priorfac_detsig)
     


#########################
#########################

# Set up the parameter list

parameters=[]
if not args.fixRed:
    if args.redSpecModel == 'powerlaw':
        [parameters.append('Ared_'+p.name) for p in psr]
        [parameters.append('gam_red_'+p.name) for p in psr]
    elif args.redSpecModel == 'spectrum':
        for ii in range(len(psr)):
            for jj in range(nmodes_red):
                parameters.append('redSpec'+'_{0}_'.format(jj+1)+psr[ii].name)
if args.incDM and not args.fixDM:
    if args.dmSpecModel == 'powerlaw':
        [parameters.append('Adm_'+p.name) for p in psr]
        [parameters.append('gam_dm_'+p.name) for p in psr]
    elif args.dmSpecModel == 'spectrum':
        for ii in range(len(psr)):
            for jj in range(nmodes_dm):
                parameters.append('dmSpec'+'_{0}_'.format(jj+1)+psr[ii].name)
if args.varyWhite:
    for ii,p in enumerate(psr):
        systems = p.sysflagdict[args.sysflag_target]
        for jj in range(len(systems)):
            parameters.append('EFAC_'+p.name+'_'+systems.keys()[jj])
        for jj in range(len(systems)):
            parameters.append('EQUAD_'+p.name+'_'+systems.keys()[jj])
        if 'nano-f' in p.sysflagdict.keys() and len(p.sysflagdict['nano-f'].keys())>0:
            for jj,nano_sysname in enumerate(p.sysflagdict['nano-f'].keys()):
                parameters.append('ECORR_'+p.name+'_'+nano_sysname)
if args.incBand:
    if args.bandSpecModel == 'powerlaw':
        parameters += ['Aband_'+str(ii) for ii in range(len(bands)-1)]
        parameters += ['gam_band_'+str(ii) for ii in range(len(bands)-1)]
    elif args.bandSpecModel == 'spectrum':
        for ii in range(len(bands)-1):
            for jj in range(nmodes_band):
                parameters.append('bandSpec_band'+str(ii)+'_mode'+str(jj))
if args.incClk:
    if args.clkSpecModel == 'powerlaw':
        parameters += ['Aclk', 'gam_clk']
    elif args.clkSpecModel == 'spectrum':
        for jj in range(nmodes_red):
            parameters.append('clkSpec'+'_{0}'.format(jj+1))
if args.incCm:
    if args.cmSpecModel == 'powerlaw':
        parameters += ['Acm', 'gam_cm']
    elif args.cmSpecModel == 'spectrum':
        for jj in range(nmodes_red):
            parameters.append('cmSpec'+'_{0}'.format(jj+1))
if args.incEph:
    if args.ephSpecModel == 'powerlaw':
        parameters += ['Aephx', 'Aephy', 'Aephz']
        parameters += ['gam_ephx', 'gam_ephy', 'gam_ephz']
    elif args.ephSpecModel == 'spectrum':
        for jj in range(nmodes_eph):
            parameters.append('ephxSpec'+'_{0}'.format(jj+1))
        for jj in range(nmodes_eph):
            parameters.append('ephySpec'+'_{0}'.format(jj+1))
        for jj in range(nmodes_eph):
            parameters.append('ephzSpec'+'_{0}'.format(jj+1))
if args.incGWB:
    if args.gwbSpecModel == 'powerlaw':
        parameters.append("Agwb")
        if not args.fix_slope:
            parameters.append("gam_gwb")
    elif args.gwbSpecModel == 'spectrum':
        for ii in range(nmodes_red):
            parameters.append('gwbSpec_{0}'.format(ii+1))
        if args.gwbPrior == 'gaussProc':
            if gwb_popparam == 'starsecc':
                parameters += ["Agwb","stars","ecc"]
            else:
                parameters += ["Agwb",gwb_popparam]
    elif args.gwbSpecModel == 'turnover':
        parameters += ["Agwb"]
        if args.gwb_fb2env is not None:
            parameters += [args.gwb_fb2env]
        elif args.gwb_fb2env is None:
            parameters += ["kappa", "fbend"]
    elif args.gwbSpecModel == 'gpEnvInterp':
        parameters += ["Agwb", "ecc"]
    if args.incCorr:
        if args.gwbTypeCorr == 'modelIndep':
            for ii in range(tmp_nwins): 
                for jj in range(int(len(psr)*(len(psr)-1)/2)):
                    parameters.append('phi_corr_win{0}_val{1}'.format(ii+1,jj+1))
        elif args.gwbTypeCorr == 'pointSrc':
            if args.fixPointSrcPhi is None and args.fixPointSrcTheta is None:
                for ii in range(tmp_nwins):
                    parameters += ["gwb_phi_win{0}".format(ii+1),
                                   "gwb_costheta_win{0}".format(ii+1)]
        elif args.gwbTypeCorr == 'spharmAnis':
            for ii in range(tmp_nwins): 
                for jj in range((args.LMAX+1)**2 - 1):
                    parameters.append('clm_win{0}_val{1}'.format(ii+1,jj+1))
        elif args.gwbTypeCorr == 'dipoleOrf':
            for ii in range(tmp_nwins):
                parameters += ["gwdip_phi_win{0}".format(ii+1),
                               "gwdip_costheta_win{0}".format(ii+1),
                               "gwdip_wgt_win{0}".format(ii+1)]
        elif args.gwbTypeCorr == 'gwDisk':
            for ii in range(tmp_nwins):
                parameters += ["gwdisk_phi_win{0}".format(ii+1),
                               "gwdisk_costheta_win{0}".format(ii+1),
                               "gwdisk_radius_win{0}".format(ii+1),
                               "gwdisk_wgt_win{0}".format(ii+1)]
        elif args.gwbTypeCorr == 'psrlocsVary':
            for ii in range(tmp_nwins):
                for jj in range(len(psr)):
                    parameters.append("gwphi_win{0}_psr{1}".format(ii+1,jj+1))
            for ii in range(tmp_nwins):
                for jj in range(len(psr)):
                    parameters.append("gwctheta_win{0}_psr{1}".format(ii+1,jj+1))
        if args.gwbModelSelect:
            parameters.append("gwb_modindex")
if args.incGWline:
    parameters += ["spec_gwline", "freq_gwline",
                   "phi_gwline", "costheta_gwline"]
if args.det_signal:
    if args.cgw_search:
        parameters += ["chirpmass", "qratio", "dist", "h_strain", "orb-freq",
                       "phi", "costheta", "cosiota", "gwpol", "gwgamma", "l0"]
        if args.ecc_search:
            parameters.append("ecc")
        if args.psrTerm:
            [parameters.append('pdist_'+p.name) for p in psr]
            [parameters.append('gp0_'+p.name) for p in psr]
            [parameters.append('lp0_'+p.name) for p in psr]
        if args.cgwModelSelect:
            parameters.append("nmodel")
    elif args.bwm_search:
        parameters += ["burst_mjd", "burst_strain",
                       "phi", "costheta", "gwpol"]
        if args.bwm_model_select:
            parameters.append("nmodel")
    if args.eph_quadratic:
        parameters += ["eph_xquad1amp", "eph_xquad2amp", "eph_yquad1amp",
                       "eph_yquad2amp", "eph_zquad1amp", "eph_zquad2amp",
                       "eph_xquad1sign", "eph_xquad2sign", "eph_yquad1sign",
                       "eph_yquad2sign", "eph_zquad1sign", "eph_zquad2sign"]
    if args.eph_planetdelta:
        parameters += ["planet{0}_delta_amp".format(ii) for ii in planet_tags]
        parameters += ["planet{0}_delta_sign".format(ii) for ii in planet_tags]


n_params = len(parameters)
if rank==0:
    print "\n You are searching for the following parameters: {0}\n".format(parameters)
    print "\n The total number of parameters is {0}\n".format(n_params)


# Define a unique file tag

file_tag = 'pta'
if args.constLike:
    file_tag += '_constLike'
if args.incGWB:
    if args.gwbSpecModel == 'powerlaw':
        if args.fix_slope:
            gamma_tag = '_gam4p33'
        else:
            gamma_tag = '_gamVary'
    elif args.gwbSpecModel == 'spectrum':
        gamma_tag = '_gwbSpec'
        if args.gwbPrior == 'gaussProc':
            gamma_tag += gwb_popparam+'Hyper{0}'.format(args.gwbHyperPrior)
    elif args.gwbSpecModel == 'turnover':
        gamma_tag = '_gwbTurn'
        if args.gwb_fb2env is not None:
            gamma_tag += 'fb2env'+args.gwb_fb2env
    elif args.gwbSpecModel == 'gpEnvInterp':
        gamma_tag = '_gwbGP'+gwb_popparam
        if args.incCosVar:
            gamma_tag += 'cosvar'
    if args.incCorr:
        if args.gwbTypeCorr == 'modelIndep':
            file_tag += '_gwb{0}_miCorr{1}{2}'.format(args.gwbPrior,
                                                      evol_corr_tag,gamma_tag)
        elif args.gwbTypeCorr == 'pointSrc':
            if args.fixPointSrcPhi is not None and args.fixPointSrcTheta is not None:
                dummy_fixpsrc = 'Fix'
            else:
                dummy_fixpsrc = ''
            file_tag += '_gwb{0}_pntSrc{1}{2}{3}'.format(args.gwbPrior,dummy_fixpsrc,
                                                         evol_corr_tag,gamma_tag)
        elif args.gwbTypeCorr == 'spharmAnis':
            if args.noPhysPrior:
                physprior_tag = '_noPhysPrior'
            elif not args.noPhysPrior:
                physprior_tag = ''
            file_tag += '_gwb{0}_Lmax{1}{2}{3}{4}'.format(args.gwbPrior,
                                                       args.LMAX,physprior_tag,
                                                       evol_corr_tag,gamma_tag)
        elif args.gwbTypeCorr == 'dipoleOrf':
            file_tag += '_gwb{0}_dip{1}{2}'.format(args.gwbPrior,
                                                   evol_corr_tag,gamma_tag)
        elif args.gwbTypeCorr == 'clock':
            file_tag += '_gwb{0}_fulcorr{1}{2}'.format(args.gwbPrior,
                                                         evol_corr_tag,gamma_tag)
        elif args.gwbTypeCorr == 'custom':
            file_tag += '_gwb{0}_cstmOrf{1}{2}'.format(args.gwbPrior,
                                                       evol_corr_tag,gamma_tag)
        elif args.gwbTypeCorr == 'gwDisk':
            file_tag += '_gwb{0}_gwDisk{1}{2}'.format(args.gwbPrior,
                                                      evol_corr_tag,gamma_tag)
        elif args.gwbTypeCorr == 'psrlocsVary':
            file_tag += '_gwb{0}_psrlocVar{1}{2}'.format(args.gwbPrior,
                                                           evol_corr_tag,gamma_tag)
        if args.gwbModelSelect:
            file_tag += 'ModSct'
    else:
        if args.gwbSpecModel == 'powerlaw':
            if args.fix_slope:
                gamma_tag = '_gam4p33'
            else:
                gamma_tag = '_gamVary'
        file_tag += '_gwb{0}_noCorr{1}'.format(args.gwbPrior,gamma_tag)
if args.pshift:
    file_tag += '_pshift'
if args.incGWline:
    if args.incCorr:
        file_tag += '_gwline{0}'.format(args.gwlinePrior)
    elif not args.incCorr:
        file_tag += '_gwline{0}_noCorr'.format(args.gwlinePrior)
if args.det_signal:
    if args.cgw_search:
        cgwtag = ''
        if args.fixcgwFreq is not None:
            cgwtag += 'fixFreq'
        if args.fixcgwPhi is not None:
            cgwtag += 'fixPhi'
        if args.fixcgwTheta is not None:
            cgwtag += 'fixTheta'
        if args.ecc_search:
            if args.fixcgwEcc is not None:
                cgwtag += 'fixEcc'
            file_tag += '_ecgw'+args.cgwPrior+cgwtag
        else:
            file_tag += '_ccgw'+args.cgwPrior+cgwtag
        if args.psrTerm:
            file_tag += 'psrTerm'
        if args.cgwModelSelect:
            file_tag += 'ModSct'
    elif args.bwm_search:
        file_tag += '_bwm'+args.bwm_antenna
        if args.bwm_model_select:
            file_tag += 'ModSct'
    if args.eph_quadratic:
        file_tag += '_ephquad'
    if args.eph_planetdelta:
        file_tag += '_ephplanetdelta'
if args.fixRed:
    red_tag = '_redFix'+'nm{0}'.format(nmodes_red)
elif not args.fixRed:
    red_tag = '_red'+args.redPrior+args.redSpecModel+'nm{0}'.format(nmodes_red)
if args.incDM:
    if args.fixDM:
        dm_tag = 'dmFix'+'nm{0}'.format(nmodes_dm)
    elif not args.fixDM:
        dm_tag = '_dm'+args.dmPrior+args.dmSpecModel+'nm{0}'.format(nmodes_dm)
elif not args.incDM:
    dm_tag = ''
if args.varyWhite:
    file_tag += '_varyWhite'
if args.incBand:
    band_tag = '_band'+args.bandPrior+args.bandSpecModel+'nm{0}'.format(nmodes_band)
elif not args.incBand:
    band_tag = ''
if args.incClk:
    clk_tag = '_clk'+args.clkPrior+args.clkSpecModel
elif not args.incClk:
    clk_tag = ''
if args.incCm:
    cm_tag = '_cm'+args.cmPrior+args.cmSpecModel
elif not args.incCm:
    cm_tag = ''
if args.incEph:
    eph_tag = '_eph'+args.ephPrior+args.ephSpecModel+'nm{0}'.format(nmodes_eph)
elif not args.incEph:
    eph_tag = ''
file_tag += red_tag + dm_tag + band_tag + \
  clk_tag + cm_tag + eph_tag


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

if args.sampler == 'mnest':

    if args.shortFileTag is not None:
        dir_name = args.dirExt+args.shortFileTag+'_mnest'
    else:
        dir_name = args.dirExt+file_tag+'_mnest'

    if rank == 0:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        if args.incCorr:
            # Copy the anisotropy modefile into the results directory
            if args.anis_modefile is not None:
                os.system('cp {0} {1}'.format(args.anis_modefile,dir_name))

        # Printing out the list of searched parameters
        fil = open(dir_name+'/parameter_list.txt','w')
        for ii,parm in enumerate(parameters):
            print >>fil, ii, parm
        fil.close()

        # Printing out the array of frequencies in the rank-reduced spectrum
        np.save(dir_name+'/freq_array_red.npy', fqs_red/86400.0)
        if args.incDM:
            np.save(dir_name+'/freq_array_dm.npy', fqs_dm/86400.0)
        if args.incBand:
            np.save(dir_name+'/freq_array_band.npy', fqs_band/86400.0)
        if args.incEph:
            np.save(dir_name+'/freq_array_eph.npy', fqs_eph/86400.0)

        # Printing out the array of random phase shifts
        psr_phaseshifts = OrderedDict.fromkeys([p.name for p in psr])
        for ii,name in enumerate(psr_phaseshifts):
            psr_phaseshifts[name] = list(psr[ii].ranphase)
        with open(dir_name+'/psr_phaseshifts.json', 'w') as fp:
            json.dump(psr_phaseshifts, fp)
        fp.close()

        # Saving command-line arguments to file
        with open(dir_name+'/run_args.json', 'w') as frun:
            json.dump(vars(args), frun)
        frun.close()

    def prior_func(xx,ndim,nparams):
        for ii in range(nparams):
            xx[ii] = pmin[ii] + xx[ii]*(pmax[ii]-pmin[ii])
            
    def like_func(xx,ndim,nparams):
        xx = np.array([xx[ii] for ii in range(nparams)])
        return lnprob(xx)        
    
    pymultinest.run(like_func, prior_func, n_params,
                    importance_nested_sampling = args.ins,
                    resume = args.resume, verbose = True, 
                    n_live_points = args.nlive,
                    outputfiles_basename=u'{0}/mnest_'.format(dir_name), 
                    sampling_efficiency = args.sampleEff,
                    const_efficiency_mode = args.constEff)

elif args.sampler == 'pchord':

    dir_name = args.dirExt+file_tag+'_pchord'

    if rank == 0:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if not os.path.exists(dir_name+'/clusters'):
            os.mkdir(dir_name+'/clusters')
        
        if args.incCorr:
            # Copy the anisotropy modefile into the results directory
            if args.anis_modefile is not None:
                os.system('cp {0} {1}'.format(args.anis_modefile,dir_name))

        # Printing out the list of searched parameters
        fil = open(dir_name+'/parameter_list.txt','w')
        for ii,parm in enumerate(parameters):
            print >>fil, ii, parm
        fil.close()

        # Printing out the array of frequencies in the rank-reduced spectrum
        np.save(dir_name+'/freq_array_red.npy', fqs_red/86400.0)
        if args.incDM:
            np.save(dir_name+'/freq_array_dm.npy', fqs_dm/86400.0)
        if args.incBand:
            np.save(dir_name+'/freq_array_band.npy', fqs_band/86400.0)
        if args.incEph:
            np.save(dir_name+'/freq_array_eph.npy', fqs_eph/86400.0)

        # Printing out the array of random phase shifts
        psr_phaseshifts = OrderedDict.fromkeys([p.name for p in psr])
        for ii,name in enumerate(psr_phaseshifts):
            psr_phaseshifts[name] = list(psr[ii].ranphase)
        with open(dir_name+'/psr_phaseshifts.json', 'w') as fp:
            json.dump(psr_phaseshifts, fp)
        fp.close()

        # Saving command-line arguments to file
        with open(dir_name+'/run_args.json', 'w') as frun:
            json.dump(vars(args), frun)
        frun.close()

    def prior_func(xx):
        for ii in range(len(xx)):
            xx[ii] = pmin[ii] + xx[ii]*(pmax[ii]-pmin[ii])
        return xx
            
    def like_func(xx):
        xx = np.array([xx[ii] for ii in range(len(xx))])
        return lnprob(xx)

    pypolychord.run(like_func, prior_func, n_params,
                    n_live = args.nlive, n_chords = args.nchords,
                    output_basename='{0}/pchord_'.format(dir_name))

elif args.sampler == 'ptmcmc':
    
    # Start the sampling off with some reasonable parameter choices
    x0 = np.array([])
    if not args.fixRed:
        if args.redSpecModel == 'powerlaw':
            # starting red parameters at single pulsar values
            startRedamp = np.log10(np.array([np.max([p.parRedamp, p.Redamp]) for p in psr]))
            startRedind = np.array([np.max([p.parRedind, p.Redind]) for p in psr])
            x0 = np.append(x0,startRedamp)
            x0 = np.append(x0,startRedind)
        elif args.redSpecModel == 'spectrum':
            x0 = np.append(x0,np.random.uniform(-7.0,-3.0,len(psr)*nmodes_red))
    if args.incDM and not args.fixDM:
        if args.dmSpecModel == 'powerlaw':
            # starting dm parameters at single pulsar values
            startDMamp = np.log10(np.array([np.max([p.parDMamp, p.DMamp]) for p in psr]))
            startDMind = np.array([np.max([p.parDMind, p.DMind]) for p in psr])
            x0 = np.append(x0,startDMamp)
            x0 = np.append(x0,startDMind)
        elif args.dmSpecModel == 'spectrum':
            x0 = np.append(x0,np.random.uniform(-7.0,-3.0,len(psr)*nmodes_dm))
    if args.varyWhite:
        for ii,p in enumerate(psr):
            systems = p.sysflagdict[args.sysflag_target]
            x0 = np.append(x0,np.random.uniform(0.75,1.25,len(systems)))
            x0 = np.append(x0,np.random.uniform(-10.0,-5.0,len(systems)))
            if 'nano-f' in p.sysflagdict.keys() and len(p.sysflagdict['nano-f'].keys())>0:
                x0 = np.append(x0, np.random.uniform(-8.5,-5.0,len(p.sysflagdict['nano-f'].keys())))
    if args.incBand:
        if args.bandSpecModel == 'powerlaw':
            x0 = np.append(x0,np.random.uniform(-20.0,-11.0,len(bands)-1))
            x0 = np.append(x0,np.random.uniform(0.0,7.0,len(bands)-1))
        elif args.bandSpecModel == 'spectrum':
            x0 = np.append(x0,np.random.uniform(-30.0,-3.0,(len(bands)-1)*nmodes_band))
    if args.incClk:
        if args.clkSpecModel == 'powerlaw':
            # starting clock parameters at random positions
            x0 = np.append(x0,np.random.uniform(-20.0,-11.0))
            x0 = np.append(x0,np.random.uniform(0.0,7.0))
        elif args.clkSpecModel == 'spectrum':
            x0 = np.append(x0,np.random.uniform(-7.0,-3.0,nmodes_red))
    if args.incCm:
        if args.cmSpecModel == 'powerlaw':
            # starting cm parameters at random positions
            x0 = np.append(x0,np.random.uniform(-20.0,-11.0))
            x0 = np.append(x0,np.random.uniform(0.0,7.0))
        elif args.cmSpecModel == 'spectrum':
            x0 = np.append(x0,np.random.uniform(-7.0,-3.0,nmodes_red))
    if args.incEph:
        if args.ephSpecModel == 'powerlaw':
            # starting eph parameters at random positions
            x0 = np.append(x0,np.random.uniform(-20.0,-11.0,3))
            x0 = np.append(x0,np.random.uniform(0.0,7.0,3))
        elif args.ephSpecModel == 'spectrum':
            x0 = np.append(x0,np.random.uniform(-7.0,-3.0,3*nmodes_eph))
    if args.incGWB:
        if args.gwbSpecModel == 'powerlaw':
            x0 = np.append(x0,-15.0)
            if not args.fix_slope:
                x0 = np.append(x0,13./3.)
        elif args.gwbSpecModel == 'spectrum':
            if args.gwbPrior != 'gaussProc':
                x0 = np.append(x0,np.random.uniform(-7.0,-3.0,nmodes_red))
            elif args.gwbPrior == 'gaussProc':
                x0 = np.append(x0,np.random.uniform(-5.0,5.0,nmodes_red))
                x0 = np.append(x0,-15.0)
                if gwb_popparam == 'ecc':
                    x0 = np.append(x0,0.8)
                elif gwb_popparam == 'stars':
                    x0 = np.append(x0,5.0)
                elif gwb_popparam == 'gas':
                    x0 = np.append(x0,0.0)
                elif gwb_popparam == 'starsecc':
                    x0 = np.append(x0,np.array([np.random.uniform(stars_range[0],stars_range[1]),
                                                np.random.uniform(ecc_range[0],ecc_range[1])]))
        elif args.gwbSpecModel == 'turnover':
            x0 = np.append(x0,-15.0)
            if args.gwb_fb2env is not None:
                if args.gwb_fb2env == 'stars':
                    x0 = np.append(x0,5.0)
                elif args.gwb_fb2env == 'gas':
                    x0 = np.append(x0,0.0)
            elif args.gwb_fb2env is None:
               x0 = np.append(x0,np.array([13./3.,-8.0]))
        elif args.gwbSpecModel == 'gpEnvInterp':
            x0 = np.append(x0,np.array([-15.0,0.2]))
        if args.incCorr:
            if args.gwbTypeCorr == 'modelIndep':
                x0 = np.append(x0,np.random.uniform(0.0,np.pi,num_corr_params))
            elif args.gwbTypeCorr == 'pointSrc':
                if args.fixPointSrcPhi is None and args.fixPointSrcTheta is None:
                    x0 = np.append(x0,np.tile([0.5,0.5],tmp_nwins))
            elif args.gwbTypeCorr == 'spharmAnis':
                x0 = np.append(x0,np.zeros(num_corr_params))
            elif args.gwbTypeCorr == 'dipoleOrf':
                x0 = np.append(x0,np.tile([0.5,0.5,0.5],tmp_nwins))
            elif args.gwbTypeCorr == 'gwDisk':
                x0 = np.append(x0,np.tile([0.5,0.5,0.1,0.0],tmp_nwins))
            elif args.gwbTypeCorr == 'psrlocsVary':
                x0 = np.append(x0,np.tile(positions[:,0],tmp_nwins))
                x0 = np.append(x0,np.tile(np.cos(positions[:,1]),tmp_nwins))
            if args.gwbModelSelect:
                x0 = np.append(x0,0.2)
    if args.incGWline:
        x0 = np.append(x0,np.array([-6.0,-8.0,0.5,0.5]))
    if args.det_signal:
        if args.cgw_search:
            x0 = np.append(x0,np.array([9.0, 0.5, 1.5, -15.0, -8.0,
                                        0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
            if args.ecc_search:
                x0 = np.append(x0,0.1)
            if args.psrTerm:
                x0 = np.append(x0,np.array([p.h5Obj['pdist'].value
                                            for p in psr]))
                x0 = np.append(x0,np.random.uniform(0.0,2.0*np.pi,len(psr)))
                x0 = np.append(x0,np.random.uniform(0.0,2.0*np.pi,len(psr)))
            if args.cgwModelSelect:
                x0 = np.append(x0,0.4)
        elif args.bwm_search:
            x0 = np.append(x0,np.array([55100.0,-14.0,0.3,0.5,0.7]))
            if args.bwm_model_select:
                x0 = np.append(x0,0.4)
        if args.eph_quadratic:
            x0 = np.append(x0,np.array(np.tile([-7.0],6)))
            x0 = np.append(x0,np.random.uniform(-1.0,1.0,6))
        if args.eph_planetdelta:
            x0 = np.append(x0,np.random.uniform(-20.0,-5.0,num_planets))
            x0 = np.append(x0,np.random.uniform(-1.0,1.0,num_planets))

    if rank==0:
        print "\n Your initial parameters are {0}\n".format(x0)

    # Make a reasonable covariance matrix to commence sampling
    cov_diag = np.array([])
    if not args.fixRed:
        if args.redSpecModel == 'powerlaw':
            cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))
            cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))
        elif args.redSpecModel == 'spectrum':
            cov_diag = np.append(cov_diag,0.1*np.ones(len(psr)*nmodes_red))
    if args.incDM and not args.fixDM:
        if args.dmSpecModel == 'powerlaw':
            cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))
            cov_diag = np.append(cov_diag,0.5*np.ones(len(psr)))
        elif args.dmSpecModel == 'spectrum':
            cov_diag = np.append(cov_diag,0.1*np.ones(len(psr)*nmodes_dm))
    if args.varyWhite:
        for ii,p in enumerate(psr):
            systems = p.sysflagdict[args.sysflag_target]
            cov_diag = np.append(cov_diag,0.5*np.ones(len(systems)))
            cov_diag = np.append(cov_diag,0.5*np.ones(len(systems)))
            if 'nano-f' in p.sysflagdict.keys() and len(p.sysflagdict['nano-f'].keys())>0:
                cov_diag = np.append(cov_diag,0.5*np.ones(len(p.sysflagdict['nano-f'].keys())))
    if args.incBand:
        if args.bandSpecModel == 'powerlaw':
            cov_diag = np.append(cov_diag,0.5*np.ones(len(bands)-1))
            cov_diag = np.append(cov_diag,0.5*np.ones(len(bands)-1))
        elif args.bandSpecModel == 'spectrum':
            cov_diag = np.append(cov_diag,0.1*np.ones((len(bands)-1)*nmodes_band))
    if args.incClk:
        if args.clkSpecModel == 'powerlaw':
            cov_diag = np.append(cov_diag,np.array([0.5,0.5]))
        elif args.clkSpecModel == 'spectrum':
            cov_diag = np.append(cov_diag,0.1*np.ones(nmodes_red))
    if args.incCm:
        if args.cmSpecModel == 'powerlaw':
            cov_diag = np.append(cov_diag,np.array([0.5,0.5]))
        elif args.cmSpecModel == 'spectrum':
            cov_diag = np.append(cov_diag,0.1*np.ones(nmodes_red))
    if args.incEph:
        if args.ephSpecModel == 'powerlaw':
            cov_diag = np.append(cov_diag,np.array([0.5,0.5,0.5]))
            cov_diag = np.append(cov_diag,np.array([0.5,0.5,0.5]))
        elif args.ephSpecModel == 'spectrum':
            cov_diag = np.append(cov_diag,0.1*np.ones(3*nmodes_eph))
    if args.incGWB:
        if args.gwbSpecModel == 'powerlaw':
            cov_diag = np.append(cov_diag,0.5)
            if not args.fix_slope:
                cov_diag = np.append(cov_diag,0.5)
        elif args.gwbSpecModel == 'spectrum':
            cov_diag = np.append(cov_diag,0.5*np.ones(nmodes_red))
            if args.gwbPrior == 'gaussProc':
                # covariance is appropriate for all physical mechanisms
                if gwb_popparam == 'starsecc':
                    cov_diag = np.append(cov_diag,np.array([0.5,0.05,0.05]))
                else:
                    cov_diag = np.append(cov_diag,np.array([0.5,0.05])) 
        elif args.gwbSpecModel == 'turnover':
            cov_diag = np.append(cov_diag,0.5)
            if args.gwb_fb2env is not None:
                cov_diag = np.append(cov_diag,0.2)
            elif args.gwb_fb2env is None:
                cov_diag = np.append(cov_diag,np.array([0.5,0.1]))
        elif args.gwbSpecModel == 'gpEnvInterp':
            cov_diag = np.append(cov_diag,np.array([0.5,0.05]))
        if args.incCorr:
            cov_diag = np.append(cov_diag,0.05*np.ones(num_corr_params))
            if args.gwbModelSelect:
                cov_diag = np.append(cov_diag,0.1)
    if args.incGWline:
        cov_diag = np.append(cov_diag,np.array([0.1,0.1,0.1,0.1]))
    if args.det_signal:
        if args.cgw_search:
            cov_diag = np.append(cov_diag,0.2*np.ones(11))
            if args.ecc_search:
                cov_diag = np.append(cov_diag,0.05)
            if args.psrTerm:
                cov_diag = np.append(cov_diag,np.array([p.h5Obj['pdistErr'].value
                                                        for p in psr])**2.0)
                cov_diag = np.append(cov_diag,0.2*np.ones(len(psr)))
                cov_diag = np.append(cov_diag,0.2*np.ones(len(psr)))
            if args.cgwModelSelect:
                cov_diag = np.append(cov_diag,0.1)
        elif args.bwm_search:
            cov_diag = np.append(cov_diag,np.array([100.0,0.1,0.1,0.1,0.1]))
            if args.bwm_model_select:
                cov_diag = np.append(cov_diag,0.1)
        if args.eph_quadratic:
            cov_diag = np.append(cov_diag,np.tile(0.1,6))
            cov_diag = np.append(cov_diag,np.tile(0.1,6))
        if args.eph_planetdelta:
            cov_diag = np.append(cov_diag,np.tile(0.1,num_planets))
            cov_diag = np.append(cov_diag,np.tile(0.1,num_planets))

    if rank==0:
        print "\n Running a quick profile on the likelihood to estimate evaluation speed...\n"
        cProfile.run('lnprob(x0)')

    ########################################
    # Creating parameter sampling groupings
    
    ind = []
    param_ct = 0
    ##### red noise #####
    if not args.fixRed:
        if args.redSpecModel == 'powerlaw':
            rdamps = [ii for ii in range(len(psr))]
            rdgam = [ii+len(psr) for ii in rdamps]
            ids = [list(aa) for aa in zip(rdamps,rdgam)]
            [ind.append(id) for id in ids if len(id) > 0]
            param_ct += 2*len(psr)
        elif args.redSpecModel == 'spectrum':
            ids = np.arange(0,nmodes_red*len(psr)).reshape((len(psr),nmodes_red))
            [ind.append(id) for id in ids if len(id) > 0]
            param_ct += nmodes_red*len(psr)
            
    ##### DM noise #####
    if args.incDM and not args.fixDM:
        if args.dmSpecModel == 'powerlaw':
            dmamps = [param_ct+ii for ii in range(len(psr))]
            dmgam = [ii+len(psr) for ii in dmamps]
            ids = [list(aa) for aa in zip(dmamps,dmgam)]
            [ind.append(id) for id in ids if len(id) > 0]
            param_ct += 2*len(psr)
        elif args.dmSpecModel == 'spectrum':
            ids = np.arange(param_ct,param_ct+nmodes_dm*len(psr)).reshape((len(psr),nmodes_dm))
            [ind.append(id) for id in ids if len(id) > 0]
            param_ct += nmodes_dm*len(psr)

    ###### White noise ######
    if args.varyWhite:
        for ii,p in enumerate(psr):
            systems = p.sysflagdict[args.sysflag_target]
            efacs = [param_ct+ii for ii in range(len(systems))]
            ids = [efacs]
            [ind.append(id) for id in ids if len(id) > 0]
            param_ct += len(systems)
            ##
            equads = [param_ct+ii for ii in range(len(systems))]
            ids = [equads]
            [ind.append(id) for id in ids if len(id) > 0]
            param_ct += len(systems)
            ##
            if 'nano-f' in p.sysflagdict.keys():
                ecorrs = [param_ct+ii for ii
                          in range(len(p.sysflagdict['nano-f'].keys()))]
                ids = [ecorrs]
                [ind.append(id) for id in ids if len(id) > 0]
                param_ct += len(p.sysflagdict['nano-f'].keys())

    ##### Band noise #######
    if args.incBand:
        if args.bandSpecModel == 'powerlaw':
            bandamps = [param_ct+ii for ii in range(len(bands)-1)]
            param_ct += (len(bands)-1)
            ##
            bandgam = [param_ct+ii for ii in range(len(bands)-1)]
            param_ct += (len(bands)-1)
            ##
            ids = [list(aa) for aa in zip(bandamps,bandgam)]
            [ind.append(id) for id in ids if len(id) > 0]
        elif args.bandSpecModel == 'spectrum':
            ids = np.arange(param_ct,param_ct+(len(bands)-1)*nmodes_band).reshape(((len(bands)-1),nmodes_band))
            [ind.append(id) for id in ids if len(id) > 0]
            param_ct += (len(bands)-1)*nmodes_band
    
    ##### Clock errors #####
    if args.incClk:
        if args.clkSpecModel == 'powerlaw':
            ids = [[param_ct,param_ct+1]]
            [ind.append(id) for id in ids if len(id) > 0]
            param_ct += 2
        elif args.clkSpecModel == 'spectrum':
            ids = [np.arange(param_ct,param_ct+nmodes_red)]
            [ind.append(id) for id in ids if len(id) > 0]
            param_ct += nmodes_red

    ##### Common noise #####
    if args.incCm:
        if args.cmSpecModel == 'powerlaw':
            ids = [[param_ct,param_ct+1]]
            [ind.append(id) for id in ids if len(id) > 0]
            param_ct += 2
        elif args.cmSpecModel == 'spectrum':
            ids = [np.arange(param_ct,param_ct+nmodes_red)]
            [ind.append(id) for id in ids if len(id) > 0]
            param_ct += nmodes_red

    ##### Ephemeris errors #####
    if args.incEph:
        if args.ephSpecModel == 'powerlaw':
            ephamps = [param_ct,param_ct+1,param_ct+2]
            ephgam = [param_ct+3,param_ct+4,param_ct+5]
            ids = [list(aa) for aa in zip(ephamps,ephgam)]
            [ind.append(id) for id in ids if len(id) > 0]
            param_ct += 6
        elif args.ephSpecModel == 'spectrum':
            ids = np.arange(param_ct,param_ct+3*nmodes_eph).reshape((3,nmodes_eph))
            [ind.append(id) for id in ids if len(id) > 0]
            param_ct += 3*nmodes_eph
        
    ##### GWB #####
    if args.incGWB:
        if args.gwbSpecModel == 'powerlaw':
            if args.fix_slope:
                ids = [[param_ct]]
                param_ct += 1
            elif not args.fix_slope:
                ids = [[param_ct,param_ct+1]]
                param_ct += 2
            [ind.append(id) for id in ids]
        elif args.gwbSpecModel == 'spectrum':
            spec_inds = range(param_ct,param_ct+nmodes_red)
            ids_spec = [np.array(spec_inds)]
            [ind.append(id) for id in ids_spec]
            param_ct += nmodes_red
            if args.gwbPrior == 'gaussProc':
                ids_gp = [np.arange(param_ct,param_ct+1+gwb_popparam_ndims)]
                [ind.append(id) for id in ids_gp]
                param_ct += 1 + gwb_popparam_ndims
        elif args.gwbSpecModel == 'turnover':
            if args.gwb_fb2env is not None:
                ids = [np.arange(param_ct,param_ct+2)]
                [ind.append(id) for id in ids]
                param_ct += 2
            elif args.gwb_fb2env is None:
                ids = [np.arange(param_ct,param_ct+3)]
                [ind.append(id) for id in ids]
                param_ct += 3
        elif args.gwbSpecModel == 'gpEnvInterp':
            ids = [np.arange(param_ct,param_ct+2)]
            [ind.append(id) for id in ids]
            param_ct += 2

    ##### GWB correlations #####
    if args.incGWB and args.incCorr and num_corr_params>0:
        ids = [np.arange(param_ct,param_ct+num_corr_params)]
        [ind.append(id) for id in ids]
        if args.gwbTypeCorr == 'modelIndep':
            mm_ct = param_ct
            # sample group for each cholesky column at each window
            for ii in range(args.nwins):
                for nn in range(1,len(psr)):
                    ids = [np.arange(mm_ct,mm_ct+nn)]
                    [ind.append(id) for id in ids]
                    mm_ct += nn
        elif args.gwbTypeCorr == 'spharmAnis' and args.LMAX>0:
            mm_ct = param_ct
            # sample group for each multipole at each window
            for ii in range(args.nwins):
                for ll in range(1,args.LMAX+1):
                    ids = [np.arange(mm_ct,mm_ct+(2*ll+1))]
                    [ind.append(id) for id in ids]
                    mm_ct += 2*ll+1
        elif args.gwbTypeCorr == 'pointSrc' and \
          args.fixPointSrcPhi is None and args.fixPointSrcTheta is None:
            mm_ct = param_ct
            for ii in range(args.nwins):
                ids = [np.array([mm_ct,mm_ct+1])]
                [ind.append(id) for id in ids]
                mm_ct += 2
        elif args.gwbTypeCorr == 'dipoleOrf':
            mm_ct = param_ct
            for ii in range(args.nwins):
                ids = [np.array([mm_ct,mm_ct+1]),
                       np.array([mm_ct+2])]
                [ind.append(id) for id in ids]
                mm_ct += 3
        elif args.gwbTypeCorr == 'gwDisk':
            mm_ct = param_ct
            for ii in range(args.nwins):
                ids = [np.array([mm_ct,mm_ct+1]),
                       np.array([mm_ct+2]),
                       np.array([mm_ct+3])]
                [ind.append(id) for id in ids]
                mm_ct += 4
        elif args.gwbTypeCorr == 'psrlocsVary':
            mm_ct = param_ct
            vphi = []
            for ii in range(args.nwins):
                vphi = [mm_ct+ii for ii in range(len(psr))]
                vctheta = [ii+len(psr) for ii in vphi]
                ids = [list(aa) for aa in zip(vphi,vctheta)]
                [ind.append(id) for id in ids if len(id) > 0]
                mm_ct += 2*len(psr)
        param_ct += num_corr_params

    ##### GW model select #####
    if args.incGWB and args.incCorr and args.gwbModelSelect:
        ids = [[param_ct]]
        [ind.append(id) for id in ids]
        param_ct += 1
            
    ##### GW line #####
    if args.incGWline:
        ids = [np.arange(param_ct,param_ct+4)]
        param_ct += 4
        [ind.append(id) for id in ids]
       
    ##### DET SIGNAL #####
    if args.det_signal:
        ##### CW #####
        if args.cgw_search:
            ids = [np.arange(param_ct,param_ct+5),
                   [param_ct+4]]
            [ind.append(id) for id in ids]
            if args.cgwPrior=='uniform':
                ids = [np.arange(param_ct+3,param_ct+5)]
                [ind.append(id) for id in ids]
            if args.ecc_search:
                ids = [np.arange(param_ct,param_ct+12)]
                param_ct += 12
            elif not args.ecc_search:
                ids = [np.arange(param_ct,param_ct+11)]
                param_ct += 11
            [ind.append(id) for id in ids]
            if args.psrTerm:
                ids = [np.arange(param_ct,param_ct+len(psr)),
                       np.arange(param_ct+len(psr),param_ct+2*len(psr)),
                       np.arange(param_ct+2*len(psr),param_ct+3*len(psr))]
                param_ct += 3*len(psr)
                [ind.append(id) for id in ids]
        ##### BWM #####
        elif args.bwm_search:
            ids = [np.arange(param_ct,param_ct+5)]
            param_ct += 5
            [ind.append(id) for id in ids]
        ##### EPHEMERIS QUADRATIC #####
        if args.eph_quadratic:
            # amplitudes
            ids = [np.arange(param_ct,param_ct+6)]
            param_ct += 6
            [ind.append(id) for id in ids]
            # signs
            ids = [np.arange(param_ct,param_ct+6)]
            param_ct += 6
            [ind.append(id) for id in ids]
        ##### EPHEMERIS PLANET DELTA #####
        if args.eph_planetdelta:
            # amplitudes
            ids = [np.arange(param_ct,param_ct+num_planets)]
            param_ct += num_planets
            [ind.append(id) for id in ids]
            # signs
            ids = [np.arange(param_ct,param_ct+num_planets)]
            param_ct += num_planets
            [ind.append(id) for id in ids]
         
            
    ##### all parameters #####
    all_inds = range(len(x0))
    ind.insert(0, all_inds)
    if rank == 0:
        print "Your parameter index groupings for sampling are {0}".format(ind)

    
    sampler = ptmcmc.PTSampler(ndim=n_params,logl=lnprob,logp=my_prior,
                            cov=np.diag(cov_diag),
                            outDir=args.dirExt+file_tag,
                            resume=args.resume, groups=ind)

    if rank == 0:
        if args.incCorr:
            # Copy the anisotropy modefile into the results directory
            if args.anis_modefile is not None:
                os.system('cp {0} {1}'.format(args.anis_modefile,
                                              args.dirExt+file_tag))

        # Printing out the list of searched parameters
        fil = open(args.dirExt+file_tag+'/parameter_list.txt','w')
        for ii,parm in enumerate(parameters):
            print >>fil, ii, parm
        fil.close()

        # Printing out the array of frequencies in the rank-reduced spectrum
        np.save(args.dirExt+file_tag+'/freq_array_red.npy', fqs_red/86400.0)
        if args.incDM:
            np.save(args.dirExt+file_tag+'/freq_array_dm.npy', fqs_dm/86400.0)
        if args.incBand:
            np.save(args.dirExt+file_tag+'/freq_array_band.npy', fqs_band/86400.0)
        if args.incEph:
            np.save(args.dirExt+file_tag+'/freq_array_eph.npy', fqs_eph/86400.0)

        # Printing out the array of random phase shifts
        psr_phaseshifts = OrderedDict.fromkeys([p.name for p in psr])
        for ii,name in enumerate(psr_phaseshifts):
            psr_phaseshifts[name] = list(psr[ii].ranphase)
        with open(args.dirExt+file_tag+'/psr_phaseshifts.json', 'w') as fp:
            json.dump(psr_phaseshifts, fp)
        fp.close()

        # Saving command-line arguments to file
        with open(args.dirExt+file_tag+'/run_args.json', 'w') as frun:
            json.dump(vars(args), frun)
        frun.close()

    #####################################
    # MCMC jump proposals
    #####################################

    # red noise draws (from Justin Ellis' PAL2)
    def drawFromRedNoisePowerlawPrior(parameters, iter, beta):
    
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)

        ind = np.unique(np.random.randint(0, npsr, 1))

        for ii in ind:
            # log prior
            if args.redPrior == 'loguniform':
                q[ii] = np.random.uniform(pmin[ii], pmax[ii])
                qxy += 0
            elif args.redPrior == 'uniform':
                q[ii] = np.random.uniform(pmin[ii], pmax[ii])
                qxy += 0
    
            q[npsr+ii] = np.random.uniform(pmin[npsr+ii], pmax[npsr+ii])
            qxy += 0

        return q, qxy

    # red noise draws (from Justin Ellis' PAL2)
    def drawFromRedNoiseSpectrumPrior(parameters, iter, beta):
    
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)

        ind = np.unique(np.random.randint(0, npsr*nmodes_red, 1))

        for ii in ind:
            # log prior
            if args.redPrior == 'loguniform':
                q[ii] = np.random.uniform(pmin[ii], pmax[ii])
                qxy += 0
            elif args.redPrior == 'uniform':
                q[ii] = np.random.uniform(pmin[ii], pmax[ii])
                qxy += 0

        return q, qxy

    # dm var draws 
    def drawFromDMNoisePowerlawPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct += npsr*nmodes_red

        ind = np.unique(np.random.randint(0, npsr, 1))

        for ii in ind:
            # log prior
            if args.dmPrior == 'loguniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0
            elif args.dmPrior == 'uniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0
    
            q[pct+npsr+ii] = np.random.uniform(pmin[pct+npsr+ii], pmax[pct+npsr+ii])
            qxy += 0

        return q, qxy

    def drawFromDMNoiseSpectrumPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct += npsr*nmodes_red

        ind = np.unique(np.random.randint(0, npsr*nmodes_dm, 1))

        for ii in ind:
            # log prior
            if args.dmPrior == 'loguniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0
            elif args.dmPrior == 'uniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0

        return q, qxy

    # white noise draws
    def drawFromWhiteNoisePrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct += npsr*nmodes_red

        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        ind = np.unique(np.random.randint(0, len(psr), 1))
        for ii,p in enumerate(psr):
            if ii == ind:
                break
            else:
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        systems = psr[ind[0]].sysflagdict[args.sysflag_target]
        q[pct:pct+len(systems)] = np.random.uniform(pmin[pct:pct+len(systems)],
                                                    pmax[pct:pct+len(systems)])
        qxy += 0
        q[pct:pct+len(systems)] = np.random.uniform(pmin[pct:pct+len(systems)],
                                                    pmax[pct:pct+len(systems)])
        qxy += 0
        if 'nano-f' in psr[ind[0]].sysflagdict.keys() and len(psr[ind[0]].sysflagdict['nano-f'].keys())>0:
            q[pct:pct+len(psr[ind[0]].sysflagdict['nano-f'].keys())] = \
              np.random.uniform(pmin[pct:pct+len(psr[ind[0]].sysflagdict['nano-f'].keys())],
                                pmax[pct:pct+len(psr[ind[0]].sysflagdict['nano-f'].keys())])
            qxy += 0

        return q, qxy

    # ephemeris error draws 
    def drawFromBandNoisePowerlawPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        # choose a band for varying
        ind = np.random.randint(0, len(bands)-1, 1)

        for ii in ind:
            # amplitude
            if args.bandPrior == 'loguniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0
            elif args.bandPrior == 'uniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0
            # gamma
            q[pct+(len(bands)-1)+ii] = np.random.uniform(pmin[pct+(len(bands)-1)+ii],
                                                         pmax[pct+(len(bands)-1)+ii])
            qxy += 0
        
        return q, qxy

    def drawFromBandNoiseSpectrumPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        # choose from full list of band spectral values
        ind = np.unique(np.random.randint(0, (len(bands)-1)*nmodes_band, 1))

        for ii in ind:
            if args.bandPrior == 'loguniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0
            elif args.bandPrior == 'uniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0
        
        return q, qxy
    
    # clock draws 
    def drawFromClkNoisePowerlawPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct += npsr*nmodes_red

        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band

        if args.clkPrior == 'loguniform':
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0
        elif args.clkPrior == 'uniform':
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0
    
        q[pct+1] = np.random.uniform(pmin[pct+1], pmax[pct+1])
        qxy += 0

        return q, qxy

    def drawFromClkNoiseSpectrumPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct += npsr*nmodes_red

        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band

        ind = np.unique(np.random.randint(0, nmodes_red, 1))

        for ii in ind:
            # log prior
            if args.clkPrior == 'loguniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0
            elif args.clkPrior == 'uniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0

        return q, qxy

    # clock draws 
    def drawFromCmNoisePowerlawPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct += npsr*nmodes_red

        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band

        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.cmPrior == 'loguniform':
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0
        elif args.cmPrior == 'uniform':
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0
    
        q[pct+1] = np.random.uniform(pmin[pct+1], pmax[pct+1])
        qxy += 0

        return q, qxy

    def drawFromClkNoiseSpectrumPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct += npsr*nmodes_red

        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        ind = np.unique(np.random.randint(0, nmodes_red, 1))

        for ii in ind:
            # log prior
            if args.cmPrior == 'loguniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0
            elif args.cmPrior == 'uniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0

        return q, qxy

    # ephemeris error draws 
    def drawFromEphNoisePowerlawPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        # choose either x,y or z for varying
        ind = np.random.randint(0, 3, 1)
        
        # amplitude
        if args.ephPrior == 'loguniform':
            q[pct+2*ind] = np.random.uniform(pmin[pct+2*ind], pmax[pct+2*ind])
            qxy += 0
        elif args.ephPrior == 'uniform':
            q[pct+2*ind] = np.random.uniform(pmin[pct+2*ind], pmax[pct+2*ind])
            qxy += 0

        # gamma
        q[pct+2*ind+1] = np.random.uniform(pmin[pct+2*ind+1], pmax[pct+2*ind+1])
        qxy += 0
        
        return q, qxy

    def drawFromEphNoiseSpectrumPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
        
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        # choose from full list of x,y,z spectral values
        ind = np.unique(np.random.randint(0, 3*nmodes_eph, 1))

        for ii in ind:
            if args.ephPrior == 'loguniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0
            elif args.ephPrior == 'uniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0
        
        return q, qxy
    
    # gwb draws 
    def drawFromGWBPowerlawPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                if 'nano-f' in p.sysflagdict.keys():
                    pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        # amplitude
        if args.gwbPrior == 'loguniform':
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0
        elif args.gwbPrior == 'uniform':
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0
        elif args.gwbPrior == 'sesana':
            mu = -15
            sig = 0.22
            q[pct] = mu + np.random.randn() * sig
            qxy -= (mu - parameters[pct]) ** 2 / 2 / \
              sig ** 2 - (mu - q[pct]) ** 2 / 2 / s ** 2
        elif args.gwbPrior == 'mcwilliams':
            mu = -14.4
            sig = 0.26
            q[pct] = mu + np.random.randn() * sig
            qxy -= (mu - parameters[pct]) ** 2 / 2 / \
              sig ** 2 - (mu - q[pct]) ** 2 / 2 / s ** 2

        # gamma
        if not args.fix_slope:
            q[pct+1] = np.random.uniform(pmin[pct+1], pmax[pct+1])
            qxy += 0
        
        return q, qxy

    def drawFromGWBSpectrumPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        if args.gwbPrior == 'gaussProc':
            ind = np.arange(0, nmodes_red)

            '''
            Agwb = q[pct+nmode]
            ecc = q[pct+nmode+1]
                
            hc_pred = np.zeros((len(fqs_red),2))
            for ii,freq in enumerate(fqs_red):
                hc_pred[ii,0], mse = gp[ii].predict(ecc, eval_MSE=True)
                hc_pred[ii,1] = np.sqrt(mse)
            psd_mean = Agwb**2.0 * hc_pred[:,0]**2.0 / \
              (12.0*np.pi**2.0) / (fqs_red/86400.0)**3.0 / Tmax
            psd_std = 2.0 * psd_mean * hc_pred[:,1] / hc_pred[:,0]
            '''
        else:
            ind = np.unique(np.random.randint(0, nmodes_red, 1))

        for ii in ind:
            if args.gwbPrior == 'loguniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0
            elif args.gwbPrior == 'uniform':
                q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
                qxy += 0
            elif args.gwbPrior == 'gaussProc':
                mu = 0.0 #0.5 * np.log10(psd_mean[ii])
                sig = 1.0 #0.5 * np.log10(psd_std[ii])
                q[pct+ii] = mu + np.random.randn() * sig
                qxy -= (mu - parameters[pct+ii]) ** 2 / 2 / \
                  sig ** 2 - (mu - q[pct+ii]) ** 2 / 2 / sig ** 2
                
        return q, qxy


    def drawFromGWBTurnoverPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        # amplitude
        if args.gwbPrior == 'loguniform':
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0
        elif args.gwbPrior == 'uniform':
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0
        elif args.gwbPrior == 'sesana':
            mu = -15
            sig = 0.22
            q[pct] = mu + np.random.randn() * sig
            qxy -= (mu - parameters[pct]) ** 2 / 2 / \
              sig ** 2 - (mu - q[pct]) ** 2 / 2 / sig ** 2
        elif args.gwbPrior == 'mcwilliams':
            mu = -14.4
            sig = 0.26
            q[pct] = mu + np.random.randn() * sig
            qxy -= (mu - parameters[pct]) ** 2 / 2 / \
              sig ** 2 - (mu - q[pct]) ** 2 / 2 / sig ** 2

        if args.gwb_fb2env is not None:
            # environmental parameter
            q[pct+1] = np.random.uniform(pmin[pct+1], pmax[pct+1])
            qxy += 0
    
        elif args.gwb_fb2env is None:
            # kappa
            q[pct+1] = np.random.uniform(pmin[pct+1], pmax[pct+1])
            qxy += 0
    
            # fbend
            q[pct+2] = np.random.uniform(pmin[pct+2], pmax[pct+2])
            qxy += 0
            
        return q, qxy

    def drawFromGWBGaussProcPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        # amplitude
        if args.gwbPrior == 'loguniform':
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0
        elif args.gwbPrior == 'uniform':
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0
        elif args.gwbPrior == 'sesana':
            mu = -15
            sig = 0.22
            q[pct] = mu + np.random.randn() * sig
            qxy -= (mu - parameters[pct]) ** 2 / 2 / \
              sig ** 2 - (mu - q[pct]) ** 2 / 2 / sig ** 2
        elif args.gwbPrior == 'mcwilliams':
            mu = -14.4
            sig = 0.26
            q[pct] = mu + np.random.randn() * sig
            qxy -= (mu - parameters[pct]) ** 2 / 2 / \
              sig ** 2 - (mu - q[pct]) ** 2 / 2 / sig ** 2

        # eccentricity
        q[pct+1] = np.random.uniform(pmin[pct+1], pmax[pct+1])
        qxy += 0
        
        return q, qxy

    def drawFromGWBSpectrumHyperPrior(parameters, iter, beta):
        """
        Only for the free spectral model.

        """

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        # adding nmodes of gwb spectrum
        pct += nmodes_red
           
        # hyper priors on spectral parameters: amplitude
        if args.gwbHyperPrior == 'loguniform':
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0
        elif args.gwbHyperPrior == 'uniform':
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0
        elif args.gwbHyperPrior == 'sesana':
            mu = -15
            sig = 0.22
            q[pct] = mu + np.random.randn() * sig
            qxy -= (mu - parameters[pct]) ** 2 / 2 / \
              sig ** 2 - (mu - q[pct]) ** 2 / 2 / sig ** 2
        elif args.gwbHyperPrior == 'mcwilliams':
            mu = -14.4
            sig = 0.26
            q[pct] = mu + np.random.randn() * sig
            qxy -= (mu - parameters[pct]) ** 2 / 2 / \
              sig ** 2 - (mu - q[pct]) ** 2 / 2 / sig ** 2

        # hyper priors on spectral parameters:
        ind = np.unique(np.random.randint(0, gwb_popparam_ndims, 1))
        q[pct+1+ind] = np.random.uniform(pmin[pct+1+ind], pmax[pct+1+ind])
        qxy += 0
        
        return q, qxy
    
    def drawFromGWBcorrPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        if args.incGWB:
            if args.gwbSpecModel == 'powerlaw':
                pct += 1
                if not args.fix_slope:
                    pct += 1
            elif args.gwbSpecModel == 'spectrum':
                pct += nmodes_red
                if args.gwbPrior == 'gaussProc':
                    pct += 1 + gwb_popparam_ndims
            elif args.gwbSpecModel == 'turnover':
                if args.gwb_fb2env is not None:
                    pct += 2
                elif args.gwb_fb2env is None:
                    pct += 3
            elif args.gwbSpecModel == 'gpEnvInterp':
                pct += 2

        if args.gwbTypeCorr == 'modelIndep':
            col = np.random.randint(1, npsr, 1)
            ind = np.unique(np.random.randint(0, num_corr_params, col))
        else:
            ind = np.unique(np.random.randint(0, num_corr_params, 1))

        for ii in ind:
            q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
            qxy += 0

        return q, qxy

    # gwb model index draws
    def drawFromGWBModSelectPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        if args.incGWB:
            if args.gwbSpecModel == 'powerlaw':
                pct += 1
                if not args.fix_slope:
                    pct += 1
            elif args.gwbSpecModel == 'spectrum':
                pct += nmodes_red
                if args.gwbPrior == 'gaussProc':
                    pct += 1 + gwb_popparam_ndims
            elif args.gwbSpecModel == 'turnover':
                if args.gwb_fb2env is not None:
                    pct += 2
                elif args.gwb_fb2env is None:
                    pct += 3
            elif args.gwbSpecModel == 'gpEnvInterp':
                pct += 2

            if args.incCorr:
                pct += num_corr_params
                if args.gwbModelSelect:
                    q[pct] = np.random.uniform(pmin[pct], pmax[pct])
                    qxy += 0

        return q, qxy

    # gwline draws 
    def drawFromGWlinePrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        if args.incGWB:
            if args.gwbSpecModel == 'powerlaw':
                pct += 1
                if not args.fix_slope:
                    pct += 1
            elif args.gwbSpecModel == 'spectrum':
                pct += nmodes_red
                if args.gwbPrior == 'gaussProc':
                    pct += 1+ gwb_popparam_ndims
            elif args.gwbSpecModel == 'turnover':
                if args.gwb_fb2env is not None:
                    pct += 2
                elif args.gwb_fb2env is None:
                    pct += 3
            elif args.gwbSpecModel == 'gpEnvInterp':
                pct += 2

            if args.incCorr:
                pct += num_corr_params
                if args.gwbModelSelect:
                    pct += 1

        # logspec_line, logfreq_line,
        # phi_line, costheta_line
        ind = np.unique(np.random.randint(0, 4, 1))

        for ii in ind:
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
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        if args.incGWB:
            if args.gwbSpecModel == 'powerlaw':
                pct += 1
                if not args.fix_slope:
                    pct += 1
            elif args.gwbSpecModel == 'spectrum':
                pct += nmodes_red
                if args.gwbPrior == 'gaussProc':
                    pct += 1 + gwb_popparam_ndims
            elif args.gwbSpecModel == 'turnover':
                if args.gwb_fb2env is not None:
                    pct += 2
                elif args.gwb_fb2env is None:
                    pct += 3
            elif args.gwbSpecModel == 'gpEnvInterp':
                pct += 2

            if args.incCorr:
                pct += num_corr_params
                if args.gwbModelSelect:
                    pct += 1

        if args.incGWline:
            pct += 4

        # logmass, qr, logdist, loghstrain,
        # logorbfreq, gwphi, costheta, cosinc,
        # gwpol, gwgamma0, l0
        if args.ecc_search:
            ind = np.unique(np.random.randint(0, 12, 1))
        elif not args.ecc_search:
            ind = np.unique(np.random.randint(0, 11, 1))

        for ii in ind:
            q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
            qxy += 0
        
        return q, qxy

    # cgw distance draws 
    def drawFromPsrDistPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        if args.incGWB:
            if args.gwbSpecModel == 'powerlaw':
                pct += 1
                if not args.fix_slope:
                    pct += 1
            elif args.gwbSpecModel == 'spectrum':
                pct += nmodes_red
                if args.gwbPrior == 'gaussProc':
                    pct += 1 + gwb_popparam_ndims
            elif args.gwbSpecModel == 'turnover':
                if args.gwb_fb2env is not None:
                    pct += 2
                elif args.gwb_fb2env is None:
                    pct += 3
            elif args.gwbSpecModel == 'gpEnvInterp':
                pct += 2

            if args.incCorr:
                pct += num_corr_params
                if args.gwbModelSelect:
                    pct += 1

        if args.incGWline:
            pct += 4

        # logmass, qr, logdist, loghstrain,
        # logorbfreq, gwphi, costheta, cosinc,
        # gwpol, gwgamma0, l0
        if args.ecc_search:
            pct += 12
        elif not args.ecc_search:
            pct += 11

        for ii, p in enumerate(psr):
            mu = p.h5Obj['pdist'].value
            sig = p.h5Obj['pdistErr'].value
            q[pct+ii] = mu + np.random.randn() * sig
            qxy -= (mu - parameters[pct+ii]) ** 2 / 2 / \
                sig ** 2 - (mu - q[pct+ii]) ** 2 / 2 / sig ** 2
        
        return q, qxy

    # cgw psrterm gamma0
    def drawFromPtermGamPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        if args.incGWB:
            if args.gwbSpecModel == 'powerlaw':
                pct += 1
                if not args.fix_slope:
                    pct += 1
            elif args.gwbSpecModel == 'spectrum':
                pct += nmodes_red
                if args.gwbPrior == 'gaussProc':
                    pct += 1 + gwb_popparam_ndims
            elif args.gwbSpecModel == 'turnover':
                if args.gwb_fb2env is not None:
                    pct += 2
                elif args.gwb_fb2env is None:
                    pct += 3
            elif args.gwbSpecModel == 'gpEnvInterp':
                pct += 2

            if args.incCorr:
                pct += num_corr_params
                if args.gwbModelSelect:
                    pct += 1

        if args.incGWline:
            pct += 4

        if args.ecc_search:
            pct += 12
        elif not args.ecc_search:
            pct += 11

        # psr distances
        pct += len(psr)

        ind = np.unique(np.random.randint(0, len(psr), 1))

        for ii in ind:
            q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
            qxy += 0
        
        return q, qxy


    # cgw psrterm l0
    def drawFromPtermEllPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        if args.incGWB:
            if args.gwbSpecModel == 'powerlaw':
                pct += 1
                if not args.fix_slope:
                    pct += 1
            elif args.gwbSpecModel == 'spectrum':
                pct += nmodes_red
                if args.gwbPrior == 'gaussProc':
                    pct += 1 + gwb_popparam_ndims
            elif args.gwbSpecModel == 'turnover':
                if args.gwb_fb2env is not None:
                    pct += 2
                elif args.gwb_fb2env is None:
                    pct += 3
            elif args.gwbSpecModel == 'gpEnvInterp':
                pct += 2

            if args.incCorr:
                pct += num_corr_params
                if args.gwbModelSelect:
                    pct += 1

        if args.incGWline:
            pct += 4

        if args.ecc_search:
            pct += 12
        elif not args.ecc_search:
            pct += 11

        # psr distances
        pct += len(psr)
        # psrterm gamma0
        pct += len(psr)

        ind = np.unique(np.random.randint(0, len(psr), 1))

        for ii in ind:
            q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
            qxy += 0
        
        return q, qxy

    # cgw psrterm l0
    def drawFromCGWModelIndexPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        if args.incGWB:
            if args.gwbSpecModel == 'powerlaw':
                pct += 1
                if not args.fix_slope:
                    pct += 1
            elif args.gwbSpecModel == 'spectrum':
                pct += nmodes_red
                if args.gwbPrior == 'gaussProc':
                    pct += 1 + gwb_popparam_ndims
            elif args.gwbSpecModel == 'turnover':
                if args.gwb_fb2env is not None:
                    pct += 2
                elif args.gwb_fb2env is None:
                    pct += 3
            elif args.gwbSpecModel == 'gpEnvInterp':
                pct += 2

            if args.incCorr:
                pct += num_corr_params
                if args.gwbModelSelect:
                    pct += 1

        if args.incGWline:
            pct += 4

        if args.ecc_search:
            pct += 12
        elif not args.ecc_search:
            pct += 11

        if args.psrTerm:
            # psr distances
            pct += len(psr)
            # psrterm gamma0
            pct += len(psr)
            # psrterm l0
            pct += len(psr)

        q[pct] = np.random.uniform(pmin[pct], pmax[pct])
        qxy += 0
        
        return q, qxy
    

    # bwm draws 
    def drawFromBWMPrior(parameters, iter, beta):
    
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        if args.incGWB:
            if args.gwbSpecModel == 'powerlaw':
                pct += 1
                if not args.fix_slope:
                    pct += 1
            elif args.gwbSpecModel == 'spectrum':
                pct += nmodes_red
                if args.gwbPrior == 'gaussProc':
                    pct += 1 + gwb_popparam_ndims
            elif args.gwbSpecModel == 'turnover':
                if args.gwb_fb2env is not None:
                    pct += 2
                elif args.gwb_fb2env is None:
                    pct += 3
            elif args.gwbSpecModel == 'gpEnvInterp':
                pct += 2

            if args.incCorr:
                pct += num_corr_params
                if args.gwbModelSelect:
                    pct += 1

        if args.incGWline:
            pct += 4

        # burst_mjd, burst_amp, phi, costheta, gwpol
        ind = np.unique(np.random.randint(0, 5, 1))

        for ii in ind:
            q[pct+ii] = np.random.uniform(pmin[pct+ii], pmax[pct+ii])
            qxy += 0
        
        return q, qxy

    # bwm model index draws 
    def drawFromBWMModelIndexPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        if args.incGWB:
            if args.gwbSpecModel == 'powerlaw':
                pct += 1
                if not args.fix_slope:
                    pct += 1
            elif args.gwbSpecModel == 'spectrum':
                pct += nmodes_red
                if args.gwbPrior == 'gaussProc':
                    pct += 1 + gwb_popparam_ndims
            elif args.gwbSpecModel == 'turnover':
                if args.gwb_fb2env is not None:
                    pct += 2
                elif args.gwb_fb2env is None:
                    pct += 3
            elif args.gwbSpecModel == 'gpEnvInterp':
                pct += 2

            if args.incCorr:
                pct += num_corr_params
                if args.gwbModelSelect:
                    pct += 1

        if args.incGWline:
            pct += 4

        if args.det_signal and args.bwm_search:
            pct += 5
            # indexing parameter is at end of list
            q[pct] = np.random.uniform(pmin[pct], pmax[pct])
            qxy += 0
        
        return q, qxy

    # bwm model index draws 
    def drawFromEphPlanetDeltaPrior(parameters, iter, beta):

        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0

        npsr = len(psr)
        pct = 0
        if not args.fixRed:
            if args.redSpecModel == 'powerlaw':
                pct = 2*npsr
            elif args.redSpecModel == 'spectrum':
                pct = npsr*nmodes_red
    
        if args.incDM and not args.fixDM:
            if args.dmSpecModel == 'powerlaw':
                pct += 2*npsr
            elif args.dmSpecModel == 'spectrum':
                pct += npsr*nmodes_dm

        if args.varyWhite:
            for ii,p in enumerate(psr):
                systems = p.sysflagdict[args.sysflag_target]
                pct += 2*len(systems)
                pct += len(p.sysflagdict['nano-f'].keys())

        if args.incBand:
            if args.bandSpecModel == 'powerlaw':
                pct += 2*(len(bands)-1)
            elif args.bandSpecModel == 'spectrum':
                pct += (len(bands)-1)*nmodes_band
                    
        if args.incClk:
            if args.clkSpecModel == 'powerlaw':
                pct += 2
            elif args.clkSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incCm:
            if args.cmSpecModel == 'powerlaw':
                pct += 2
            elif args.cmSpecModel == 'spectrum':
                pct += nmodes_red

        if args.incEph:
            if args.ephSpecModel == 'powerlaw':
                pct += 6
            elif args.ephSpecModel == 'spectrum':
                pct += 3*nmodes_eph

        if args.incGWB:
            if args.gwbSpecModel == 'powerlaw':
                pct += 1
                if not args.fix_slope:
                    pct += 1
            elif args.gwbSpecModel == 'spectrum':
                pct += nmodes_red
                if args.gwbPrior == 'gaussProc':
                    pct += 1 + gwb_popparam_ndims
            elif args.gwbSpecModel == 'turnover':
                if args.gwb_fb2env is not None:
                    pct += 2
                elif args.gwb_fb2env is None:
                    pct += 3
            elif args.gwbSpecModel == 'gpEnvInterp':
                pct += 2

            if args.incCorr:
                pct += num_corr_params
                if args.gwbModelSelect:
                    pct += 1

        if args.incGWline:
            pct += 4

        # choose a planet mass to perturb
        ind = np.unique(np.random.randint(0, num_planets, 1))
        if args.det_signal and args.eph_planetdelta:
            q[pct+ind] = np.random.uniform(pmin[pct+ind], pmax[pct+ind])
            q[pct+ind+num_planets] = np.random.uniform(pmin[pct+ind+num_planets],
                                                       pmax[pct+ind+num_planets])
            qxy += 0
        
        return q, qxy

  
    # add jump proposals
    if not args.fixRed:
        if args.redSpecModel == 'powerlaw':
            sampler.addProposalToCycle(drawFromRedNoisePowerlawPrior, 10)
        elif args.redSpecModel == 'spectrum':
            sampler.addProposalToCycle(drawFromRedNoiseSpectrumPrior, 10)
    if args.incDM and not args.fixDM:
        if args.dmSpecModel == 'powerlaw':
            sampler.addProposalToCycle(drawFromDMNoisePowerlawPrior, 10)
        elif args.dmSpecModel == 'spectrum':
            sampler.addProposalToCycle(drawFromDMNoiseSpectrumPrior, 10)
    if args.varyWhite:
        sampler.addProposalToCycle(drawFromWhiteNoisePrior, 10)
    if args.incBand:
        if args.bandSpecModel == 'powerlaw':
            sampler.addProposalToCycle(drawFromBandNoisePowerlawPrior, 10)
        elif args.bandSpecModel == 'spectrum':
            sampler.addProposalToCycle(drawFromBandNoiseSpectrumPrior, 10)
    if args.incClk:
        if args.clkSpecModel == 'powerlaw':
            sampler.addProposalToCycle(drawFromClkNoisePowerlawPrior, 10)
        elif args.clkSpecModel == 'spectrum':
            sampler.addProposalToCycle(drawFromClkNoiseSpectrumPrior, 10)
    if args.incCm:
        if args.cmSpecModel == 'powerlaw':
            sampler.addProposalToCycle(drawFromCmNoisePowerlawPrior, 10)
        elif args.cmSpecModel == 'spectrum':
            sampler.addProposalToCycle(drawFromCmNoiseSpectrumPrior, 10)
    if args.incEph:
        if args.ephSpecModel == 'powerlaw':
            sampler.addProposalToCycle(drawFromEphNoisePowerlawPrior, 10)
        elif args.ephSpecModel == 'spectrum':
            sampler.addProposalToCycle(drawFromEphNoiseSpectrumPrior, 10)
    if args.incGWB:
        if args.gwbSpecModel == 'powerlaw':
            sampler.addProposalToCycle(drawFromGWBPowerlawPrior, 10)
        elif args.gwbSpecModel == 'spectrum':
            sampler.addProposalToCycle(drawFromGWBSpectrumPrior, 10)
            if args.gwbPrior == 'gaussProc':
                sampler.addProposalToCycle(drawFromGWBSpectrumHyperPrior, 5)
        elif args.gwbSpecModel == 'turnover':
            sampler.addProposalToCycle(drawFromGWBTurnoverPrior, 10)
        elif args.gwbSpecModel == 'gpEnvInterp':
            sampler.addProposalToCycle(drawFromGWBGaussProcPrior, 10)
        if args.incCorr:
            if num_corr_params>0:
                sampler.addProposalToCycle(drawFromGWBcorrPrior, 10)
            if args.gwbModelSelect:
                sampler.addProposalToCycle(drawFromGWBModSelectPrior, 10)
    if args.incGWline:
        sampler.addProposalToCycle(drawFromGWlinePrior, 10)
    if args.det_signal and args.cgw_search:
        sampler.addProposalToCycle(drawFromCWPrior, 10)
        if args.psrTerm:
            sampler.addProposalToCycle(drawFromPsrDistPrior, 10)
            sampler.addProposalToCycle(drawFromPtermGamPrior, 10)
            sampler.addProposalToCycle(drawFromPtermEllPrior, 10)
        if args.cgwModelSelect:
            sampler.addProposalToCycle(drawFromCGWModelIndexPrior, 5)
    elif args.det_signal and args.bwm_search:
        sampler.addProposalToCycle(drawFromBWMPrior, 10)
        if args.bwm_model_select:
            sampler.addProposalToCycle(drawFromBWMModelIndexPrior, 5)
    elif args.det_signal and args.eph_planetdelta:
        sampler.addProposalToCycle(drawFromEphPlanetDeltaPrior, 10)
    

    sampler.sample(p0=x0, Niter=int(args.niter), thin=10,
                covUpdate=1000, AMweight=20,
                SCAMweight=30, DEweight=50,
                writeHotChains=args.writeHotChains,
                hotChain=args.hotChain)
