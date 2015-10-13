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
from collections import OrderedDict

class PsrObj(object):
    T2psr = None
    psr_locs = None
    toas = None
    toaerrs = None
    res = None
    obs_freqs = None
    G = None
    Mmat = None
    sysflagdict = None
    Fred = None
    Fdm = None
    Ftot = None
    diag_white = None
    res_prime = None
    Ftot_prime = None
    Gc = None
    Te = None
    name = "J0000+0000"
    Gres = None
    epflags = None

    def __init__(self, t2obj):
        self.T2psr = t2obj
        self.parfile = None
        self.timfile = None
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
        self.sysflagdict = None
        self.Gres = None
        self.epflags = None

    """
    Initialise the libstempo object.
    """
    def grab_all_vars(self, jitterbin=10.): # jitterbin is in seconds

        print "--> Processing {0}".format(self.T2psr.name)
        
        # basic quantities
        self.name = self.T2psr.name
        self.toas = np.double(self.T2psr.toas())
        self.res = np.double(self.T2psr.residuals())
        self.toaerrs = np.double(self.T2psr.toaerrs) * 1e-6
        self.obs_freqs = np.double(self.T2psr.ssbfreqs())
        self.Mmat = np.double(self.T2psr.designmatrix())
              
        # get the sky position
        if 'RAJ' and 'DECJ' in self.T2psr.pars():
            self.psr_locs = [self.T2psr['RAJ'].val,self.T2psr['DECJ'].val]
        elif 'ELONG' and 'ELAT' in self.T2psr.pars():
            fac = 180./np.pi
            # check for B name
            if 'B' in self.name:
                epoch = '1950'
            else:
                epoch = '2000'
            coords = Equatorial(Ecliptic(str(self.T2psr['ELONG'].val*fac), str(self.T2psr['ELAT'].val*fac)), epoch=epoch)
            self.psr_locs = [float(repr(coords.ra)),float(repr(coords.dec))]

        print "--> Grabbed the pulsar position."
        ################################################################################################
            
        # These are all the relevant system flags used by the PTAs.
        system_flags = ['group','sys','i','f']
        self.sysflagdict = OrderedDict.fromkeys(system_flags)

        # Put the systems into a dictionary which 
        # has the locations of their toa placements.
        for systm in self.sysflagdict:
            try:
                if systm in self.T2psr.flags():
                    sys_uflagvals = list(set(self.T2psr.flagvals(systm)))
                    self.sysflagdict[systm] = OrderedDict.fromkeys(sys_uflagvals)
                    for kk,subsys in enumerate(sys_uflagvals):
                        self.sysflagdict[systm][subsys] = np.where(self.T2psr.flagvals(systm) == sys_uflagvals[kk])
            except KeyError:
                pass

        # If we have some NANOGrav data, then separate
        # this off for later ECORR assignment.
        if 'pta' in self.T2psr.flags():
            pta_names = list(set(self.T2psr.flagvals('pta')))
            pta_mask = [self.T2psr.flagvals('pta')==ptaname for ptaname in pta_names]
            pta_maskdict = OrderedDict.fromkeys(pta_names)
            for ii,item in enumerate(pta_maskdict):
                pta_maskdict[item] = pta_mask[ii]
            if len(pta_names)!=0 and ('NANOGrav' in pta_names):
                try:
                    nanoflagdict = OrderedDict.fromkeys(['nano-f'])
                    nano_flags = list(set(self.T2psr.flagvals('group')[pta_maskdict['NANOGrav']]))
                    nanoflagdict['nano-f'] = OrderedDict.fromkeys(nano_flags)
                    for kk,subsys in enumerate(nano_flags):
                        nanoflagdict['nano-f'][subsys] = np.where(self.T2psr.flagvals('group') == nano_flags[kk])
                    self.sysflagdict.update(nanoflagdict)
                except KeyError:
                    nanoflagdict = OrderedDict.fromkeys(['nano-f'])
                    nano_flags = list(set(self.T2psr.flagvals('f')[pta_maskdict['NANOGrav']]))
                    nanoflagdict['nano-f'] = OrderedDict.fromkeys(nano_flags)
                    for kk,subsys in enumerate(nano_flags):
                        nanoflagdict['nano-f'][subsys] = np.where(self.T2psr.flagvals('f') == nano_flags[kk])
                    self.sysflagdict.update(nanoflagdict)
                    
        
        # If there are really no relevant flags,
        # then just make a full list of the toa indices.
        if np.all([self.sysflagdict[sys] is None for sys in self.sysflagdict]):
            print "No relevant flags found"
            print "Assuming one overall system for {0}\n".format(self.T2psr.name)
            self.sysflagdict[self.T2psr.name] = OrderedDict.fromkeys([self.T2psr.name])
            self.sysflagdict[self.T2psr.name][self.T2psr.name] = np.arange(len(self.toas))

        print "--> Processed all relevant flags plus associated locations."
        ##################################################################################################

        if 'pta' in self.T2psr.flags():
            if 'NANOGrav' in pta_names:
                # now order everything
                try:
                    isort, iisort = utils.argsortTOAs(self.toas, self.T2psr.flagvals('group'), which='jitterext', dt=jitterbin/86400.)
                    flags = self.T2psr.flagvals('group')[isort]
                except KeyError:
                    isort, iisort = utils.argsortTOAs(self.toas, self.T2psr.flagvals('f'), which='jitterext', dt=jitterbin/86400.)
                    flags = self.T2psr.flagvals('f')[isort]
        
                # sort data
                self.toas = self.toas[isort]
                self.toaerrs = self.toaerrs[isort]
                self.res = self.res[isort]
                self.obs_freqs = self.obs_freqs[isort]
                self.Mmat = self.Mmat[isort, :]
                detresiduals = self.res.copy()

                print "--> Sorted data."
    
                # get quantization matrix
                avetoas, self.Umat, Ui = utils.quantize_split(self.toas, flags, dt=jitterbin/86400., calci=True)
                print "--> Computed quantization matrix."

                # get only epochs that need jitter/ecorr
                self.Umat, avetoas, aveflags = utils.quantreduce(self.Umat, avetoas, flags)
                print "--> Excized epochs without jitter."

                # get quantization indices
                self.Uinds = utils.quant2ind(self.Umat)
                self.epflags = flags[self.Uinds[:, 0]]

                print "--> Checking TOA sorting and quantization..."
                print utils.checkTOAsort(self.toas, flags, which='jitterext', dt=jitterbin/86400.)
                print utils.checkquant(self.Umat, flags)
                print "...Finished checks."

        # perform SVD of design matrix to stabilise
        print "--> Performing SVD of design matrix for stabilization..."   
        u,s,v = np.linalg.svd(self.Mmat)
        self.G = u[:,len(s):len(u)]
        self.Gc =  u[:,:len(s)]

        self.Gres = np.dot(self.G.T, self.res)

        print "--> Done reading in pulsar :-)"

    def makeFred(self, nmodes, Ttot):
        self.Fred = utils.createfourierdesignmatrix_RED(self.toas, nmodes, Tspan=Ttot)

    def makeFdm(self, nmodes, Ttot):
        self.Fdm = utils.createfourierdesignmatrix_DM(self.toas, nmodes, self.obs_freqs, Tspan=Ttot)
    
    def makeFtot(self, nmodes, Ttot):
        self.Fred = utils.createfourierdesignmatrix_RED(self.toas, nmodes, Tspan=Ttot)
        self.Fdm = utils.createfourierdesignmatrix_DM(self.toas, nmodes, self.obs_freqs, Tspan=Ttot)

        self.Ftot = np.append(self.Fred, self.Fdm, axis=1)

    def makeTe(self, nmodes, Ttot, makeDM=False):

        self.Fred = utils.createfourierdesignmatrix_RED(self.toas, nmodes, Tspan=Ttot)

        if makeDM==True:
            self.Fdm = utils.createfourierdesignmatrix_DM(self.toas, nmodes, self.obs_freqs, Tspan=Ttot)
            self.Ftot = np.append(self.Fred, self.Fdm, axis=1)

        else:
            self.Ftot = self.Fred

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
