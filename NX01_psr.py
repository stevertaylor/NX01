#!/usr/bin/env python

import numpy as np
import sys, os, glob
import libstempo as T2
import ephem
from ephem import *
import NX01_utils as utils

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
    Fred = None
    Fdm = None
    Ftot = None
    diag_white = None
    res_prime = None
    Ftot_prime = None
    Gcmat = None
    name = "J0000+0000"
    Tmax = None

    def __init__(self, t2obj):
        self.T2psr = t2obj
        self.psr_locs = None
        self.toas = None
        self.toaerrs = None
        self.res = None
        self.obs_freqs = None
        self.G = None
        self.Fred = None
        self.Fdm = None
        self.Ftot = None
        self.diag_white = None
        self.res_prime = None
        self.Ftot_prime = None
        self.Gcmat = None
        self.name = "J0000+0000"
        self.Tmax = None
        self.bkends = None
        self.bkend_names = None

    """
    Initialise the libstempo object for use in nonlinear timing model modelling.
    No parameters are required, all content must already be in memory
    """
    def grab_all_vars(self):
        self.name = self.T2psr.name
        self.toas = self.T2psr.toas()
        self.res = self.T2psr.residuals()
        self.toaerrs = self.T2psr.toaerrs * 1e-6
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

    def makeFred(self, nmodes, Ttot):
        self.Fred = utils.createfourierdesignmatrix_RED(self.toas, nmodes, Tspan=Ttot)

    def makeFdm(self, nmodes, Ttot):
        self.Fdm = utils.createfourierdesignmatrix_DM(self.toas, nmodes, self.obs_freqs, Tspan=Ttot)
    
    def makeFtot(self, nmodes, Ttot):
        self.Fred = utils.createfourierdesignmatrix_RED(self.toas, nmodes, Tspan=Ttot)
        self.Fdm = utils.createfourierdesignmatrix_DM(self.toas, nmodes, self.obs_freqs, Tspan=Ttot)

        self.Ftot = np.append(self.Fred, self.Fdm, axis=1)

    def two_comp_noise(self, MLerrors):
        efac_bit = np.dot(self.G.T, np.dot( np.diag(MLerrors**2.0), self.G ) )
        equad_bit = np.dot(self.G.T,self.G)
        Lequad = np.linalg.cholesky(equad_bit)
        Lequad_inv = np.linalg.inv(Lequad)
        sand = np.dot(Lequad_inv, np.dot(efac_bit, Lequad_inv.T))
        u,s,v = np.linalg.svd(sand)
        proj = np.dot(u.T, np.dot(Lequad_inv, self.G.T))
        ########
        self.diag_white = s
        self.res_prime = np.dot(proj, self.res) 
        self.Ftot_prime = np.dot(proj, self.Ftot)
