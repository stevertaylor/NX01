#!/usr/bin/env python

import numpy as np
import sys, os, glob
import libstempo as T2
import ephem
from ephem import *

class PsrObj(object):
    T2psr = None
    psr_locs = None
    toas = None
    toaerrs = None
    res = None
    obs_freqs = None
    G = None
    bkends = None
    bkend_names = None
    Gcmat = None
    ptmpars = []
    ptmparerrs = []
    ptmdescription = []
    flags = None
    name = "J0000+0000"
    Tmax = None

    # The auxiliary quantities
    Fmat = None
   
    # Auxiliaries used in the likelihood
    twoComponentNoise = False       # Whether we use the 2-component noise model
    Nvec = None             # The total white noise (eq^2 + ef^2*err)
    Wvec = None             # The weights in 2-component noise
    Wovec = None            # The weights in 2-component orthogonal noise
    Nwvec = None            # Total noise in 2-component basis (eq^2 + ef^2*Wvec)
    Nwovec = None           # Total noise in 2-component orthogonal basis
    Jweight = None          # The weight of the jitter noise in compressed basis
    Jvec = None

    def __init__(self, t2obj):
        self.T2psr = t2obj
        self.psr_locs = None
        self.toas = None
        self.toaerrs = None
        self.res = None
        self.obs_freqs = None
        self.G = None
        self.Gcmat = None
        self.name = "J0000+0000"
        self.Tmax = None
        self.bkends = None
        self.bkend_names = None

        self.Fmat = None

    """
    Initialise the libstempo object for use in nonlinear timing model modelling.
    No parameters are required, all content must already be in memory
    """
    def grab_all_vars(self):
        self.name = self.T2psr.name
        self.toas = self.T2psr.toas()
        self.res = self.T2psr.residuals()
        self.toaerrs = self.T2psr.toaerrs
        self.obs_freqs = self.T2psr.freqs

        self.des = self.T2psr.designmatrix()
        U,S,V = np.linalg.svd(self.des)
        self.G = U[:,len(S):len(U)]

        if 'RAJ' and 'DECJ' in self.T2psr.pars:
            self.psr_locs = [self.T2psr['RAJ'].val,self.T2psr['DECJ'].val]
        elif 'ELONG' and 'ELAT' in self.T2psr.pars:
            fac = 180./np.pi
            coords = Equatorial(Ecliptic(str(self.T2psr['ELONG'].val*fac), str(self.T2psr['ELAT'].val*fac)))
            self.psr_locs = [float(repr(coords.ra)),float(repr(coords.dec))]

    def get_backends(self):
        if 'sys' in self.T2psr.flags:
            self.bkend_names = np.unique(self.T2psr.flags['sys'])
            self.bkends = [0.0]*len(np.unique(self.T2psr.flags['sys']))
            for k in range(len(np.unique(self.T2psr.flags['sys']))):
                self.bkends[k] = np.where(self.T2psr.flags['sys'] == np.unique(self.T2psr.flags['sys'])[k])[0]
        else:
            print "No 'sys' flags found :("
            print "Using one overall EFAC for {0}\n".format(psr[ii].name)
       
