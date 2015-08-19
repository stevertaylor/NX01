#!/usr/bin/env python

"""
Created by stevertaylor
Copyright (c) 2014 Stephen R. Taylor

Code contributions by Rutger van Haasteren (piccard) and Justin Ellis (PAL/PAL2).

"""

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
    Mmat = None
    systems = None
    sysnames = None
    Fred = None
    Fdm = None
    Ftot = None
    diag_white = None
    res_prime = None
    Ftot_prime = None
    Gc = None
    Te = None
    name = "J0000+0000"
    Tmax = None
    Gres = None
    epflags = None

    def __init__(self, t2obj):
        self.T2psr = t2obj
        self.psr_locs = None
        self.toas = None
        self.toaerrs = None
        self.res = None
        self.obs_freqs = None
        self.G = None
        self.Mmat = None
        self.Fred = None
        self.Fdm = None
        self.Ftot = None
        self.diag_white = None
        self.res_prime = None
        self.Ftot_prime = None
        self.Gc = None
        self.Te = None
        self.Umat = None
        self.Uinds = None
        self.name = "J0000+0000"
        self.Tmax = None
        self.systems = None
        self.sysnames = None
        self.Gres = None
        self.epflags = None

    """
    Initialise the libstempo object for use in nonlinear timing model modelling.
    No parameters are required, all content must already be in memory
    """
    def grab_all_vars(self):

        # basic quantities
        self.name = self.T2psr.name
        self.toas = self.T2psr.toas()
        self.res = self.T2psr.residuals()
        self.toaerrs = self.T2psr.toaerrs * 1e-6
        self.obs_freqs = self.T2psr.freqs

        self.Mmat = self.T2psr.designmatrix()
        '''u,s,v = np.linalg.svd(self.Mmat)
        self.G = u[:,len(s):len(u)]
        self.Gc =  u[:,:len(s)]

        self.Gres = np.dot(self.G.T, self.res)'''
        
        # get the sky position
        if 'RAJ' and 'DECJ' in self.T2psr.pars():
            self.psr_locs = [self.T2psr['RAJ'].val,self.T2psr['DECJ'].val]
        elif 'ELONG' and 'ELAT' in self.T2psr.pars():
            fac = 180./np.pi
            coords = Equatorial(Ecliptic(str(self.T2psr['ELONG'].val*fac), str(self.T2psr['ELAT'].val*fac)))
            self.psr_locs = [float(repr(coords.ra)),float(repr(coords.dec))]

        '''# get the relevant system flags
        self.systems = []
        self.sysnames = []
        system_flags = ['group','sys','i','f']
        for systm in system_flags:
            try:
                sys_uflagvals = list(set(self.T2psr.flagvals(systm)))
                if systm in self.T2psr.flags():
                    self.sysnames = np.append( self.sysnames, sys_uflagvals )
                    for kk in range(len(sys_uflagvals)):
                        self.systems.append(np.where(self.T2psr.flagvals(systm) == sys_uflagvals[kk]))
            except KeyError:
                pass
        
        if len(self.systems)==0:
            print "No relevant flags found"
            print "Assuming one overall system for {0}\n".format(self.name)
            self.sysnames = self.name
            self.systems = np.arange(len(self.toas))'''

        # now order everything
        isort, iisort = utils.argsortTOAs(self.toas, self.T2psr.flagvals('f'), which='jitterext', dt=10./86400.)

        # sort data
        self.toas = self.toas[isort]
        self.toaerrs = self.toaerrs[isort]
        self.res = self.res[isort]
        self.obs_freqs = self.obs_freqs[isort]
        flags = self.T2psr.flagvals('f')[isort]
        self.Mmat = self.Mmat[isort, :]
        detresiduals = self.res.copy()

        # get quantization matrix
        avetoas, self.Umat, Ui = utils.quantize_split(self.toas, flags, dt=10./86400., calci=True)
        #print Umat.shape

        # get only epochs that need jitter/ecorr
        self.Umat, avetoas, aveflags = utils.quantreduce(self.Umat, avetoas, flags)
        #print Umat.shape

        # get quantization indices
        self.Uinds = utils.quant2ind(self.Umat)
        self.epflags = flags[self.Uinds[:, 0]]

        print utils.checkTOAsort(self.toas, flags, which='jitterext', dt=10./86400.)
        print utils.checkquant(self.Umat, flags)

        # get the relevant system flags
        self.systems = []
        self.sysnames = []
        system_flags = ['group','sys','i','f']
        for systm in system_flags:
            try:
                sys_uflagvals = list(set(self.T2psr.flagvals(systm)))
                if systm in self.T2psr.flags():
                    self.sysnames = np.append( self.sysnames, sys_uflagvals )
                    for kk in range(len(sys_uflagvals)):
                        self.systems.append(np.where(self.T2psr.flagvals(systm) == sys_uflagvals[kk]))
            except KeyError:
                pass
        
        if len(self.systems)==0:
            print "No relevant flags found"
            print "Assuming one overall system for {0}\n".format(self.name)
            self.sysnames = self.name
            self.systems = np.arange(len(self.toas))

        #self.Mmat = self.T2psr.designmatrix()
        u,s,v = np.linalg.svd(self.Mmat)
        self.G = u[:,len(s):len(u)]
        self.Gc =  u[:,:len(s)]

        self.Gres = np.dot(self.G.T, self.res)

    def makeFred(self, nmodes, Ttot):
        self.Fred = utils.createfourierdesignmatrix_RED(self.toas, nmodes, Tspan=Ttot)

    def makeFdm(self, nmodes, Ttot):
        self.Fdm = utils.createfourierdesignmatrix_DM(self.toas, nmodes, self.obs_freqs, Tspan=Ttot)
    
    def makeFtot(self, nmodes, Ttot):
        self.Fred = utils.createfourierdesignmatrix_RED(self.toas, nmodes, Tspan=Ttot)
        self.Fdm = utils.createfourierdesignmatrix_DM(self.toas, nmodes, self.obs_freqs, Tspan=Ttot)

        self.Ftot = np.append(self.Fred, self.Fdm, axis=1)

    def makeTe(self, nmodes, Ttot):
        self.Fred = utils.createfourierdesignmatrix_RED(self.toas, nmodes, Tspan=Ttot)
        self.Fdm = utils.createfourierdesignmatrix_DM(self.toas, nmodes, self.obs_freqs, Tspan=Ttot)

        self.Ftot = np.append(self.Fred, self.Fdm, axis=1)
        self.Te = np.append(self.Gc, self.Ftot, axis=1)

    def two_comp_noise(self, mlerrors):
        efac_bit = np.dot(self.G.T, np.dot( np.diag(mlerrors**2.0), self.G ) )
        equad_bit = np.dot(self.G.T,self.G)
        Lequad = np.linalg.cholesky(equad_bit)
        Lequad_inv = np.linalg.inv(Lequad)
        sand = np.dot(Lequad_inv, np.dot(efac_bit, Lequad_inv.T))
        u,s,v = np.linalg.svd(sand)
        proj = np.dot(u.T, np.dot(Lequad_inv, self.G.T))
        ########
        self.diag_white = s
        self.res_prime = np.dot(proj, self.res)
        if self.Ftot is not None:
            self.Ftot_prime = np.dot(proj, self.Ftot)
        else:
            self.Ftot_prime = np.dot(proj, self.Fred)
