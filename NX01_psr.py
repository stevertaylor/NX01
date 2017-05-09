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
import NX01_utils as utils
from collections import OrderedDict
import cPickle as pickle

mjd2jd = 2400000.5
f1yr = 1./(86400.0*365.25)

class PsrObj(object):
    T2psr = None
    parfile = None
    timfile = None
    noisefile = None
    psr_locs = None
    raj = None
    decj = None
    elong = None
    elat = None
    psrPos = None
    toas = None
    toaerrs = None
    res = None
    obs_freqs = None
    G = None
    Mmat = None
    flags = None
    flagvals = None
    sysflagdict = None
    tmask = None
    isort = None
    iisort = None
    ephemeris = None
    ephemname = None
    roemer = None
    Fred = None
    Fdm = None
    Fephx = None
    Fephy = None
    Fephz = None
    Ftot = None
    ranphase = None
    diag_white = None
    res_prime = None
    Ftot_prime = None
    Gc = None
    Te = None
    name = "J0000+0000"
    Gres = None
    epflags = None
    detsig_avetoas = None
    detsig_Uinds = None
    planet_ssb = None
    parRedamp = None
    parRedind = None
    parDMamp = None
    parDMind = None
    Redamp = None
    Redind = None
    DMamp = None
    DMind = None

    def __init__(self, t2obj):
        self.T2psr = t2obj
        self.parfile = None
        self.timfile = None
        self.noisefile = None
        self.psr_locs = None
        self.raj = None
        self.decj = None
        self.elong = None
        self.elat = None
        self.psrPos = None
        self.toas = None
        self.toaerrs = None
        self.res = None
        self.obs_freqs = None
        self.G = None
        self.Mmat = None
        self.flags = None
        self.flagvals = None
        self.Fred = None
        self.Fdm = None
        self.Fephx = None
        self.Fephy = None
        self.Fephz = None
        self.Ftot = None
        self.ranphase = None
        self.diag_white = None
        self.res_prime = None
        self.Ftot_prime = None
        self.Gc = None
        self.Te = None
        self.Umat = None
        self.Uinds = None
        self.name = "J0000+0000"
        self.sysflagdict = None
        self.tmask = None
        self.isort = None
        self.iisort = None
        self.ephemeris = None
        self.ephemname = None
        self.roemer = None
        self.Gres = None
        self.epflags = None
        self.detsig_avetoas = None
        self.detsig_Uinds = None
        self.planet_ssb = None
        self.parRedamp = 1e-20
        self.parRedind = 0.0
        self.parDMamp = 1e-20
        self.parDMind = 0.0
        self.Redamp = 1e-20
        self.Redind = 0.0
        self.DMamp = 1e-20
        self.DMind = 0.0

    """
    Initialise the libstempo object.
    """
    def grab_all_vars(self, jitterbin=10., makeGmat=False, fastDesign=True,
                      planetssb=False, allEphem=False,
                      startMJD=None, endMJD=None):
        # jitterbin is in seconds

        print "--> Processing {0}".format(self.T2psr.name)

        # basic quantities
        self.name = self.T2psr.name
        self.psrPos = self.T2psr.psrPos
        self.toas = np.double(self.T2psr.toas())
        self.res = np.double(self.T2psr.residuals())
        self.toaerrs = np.double(self.T2psr.toaerrs) * 1e-6
        self.obs_freqs = np.double(self.T2psr.ssbfreqs())
        self.Mmat = np.double(self.T2psr.designmatrix())
        self.flags = self.T2psr.flags()
        self.flagvals = OrderedDict.fromkeys(self.flags)
        for flag in self.flags:
            self.flagvals[flag] = self.T2psr.flagvals(flag)

        # getting ephemeris properties
        self.ephemeris = self.T2psr.ephemeris
        if '436' in self.T2psr.ephemeris:
            self.ephemname = 'DE436'
        elif '435' in self.T2psr.ephemeris:
            self.ephemname = 'DE435'
        elif '430' in self.T2psr.ephemeris:
            self.ephemname = 'DE430'
        elif '421' in self.T2psr.ephemeris:
            self.ephemname = 'DE421'

        # populating roemer-delay dictionary
        self.roemer = OrderedDict()
        self.roemer[self.ephemname] = np.double(self.T2psr.roemer)

        # time filtering
        if startMJD is not None and endMJD is not None:
            self.tmask = np.logical_and(self.T2psr.toas() >= startMJD,
                                        self.T2psr.toas() <= endMJD)

            self.toas = self.toas[self.tmask]
            self.toaerrs = self.toaerrs[self.tmask]
            self.res = self.res[self.tmask]
            self.obs_freqs = self.obs_freqs[self.tmask]

            self.Mmat = self.Mmat[self.tmask,:]
            dmx_mask = np.sum(self.Mmat, axis=0) != 0.0
            self.Mmat = self.Mmat[:,dmx_mask]

            for flag in self.flags:
                self.flagvals[flag] = self.T2psr.flagvals(flag)[self.tmask]

            for eph in self.roemer:
                self.roemer[eph] = self.roemer[eph][self.tmask]

        # get the position vectors of the planets
        if planetssb:
            if allEphem:
                from jplephem.spk import SPK
                from scipy import constants as sc

                ephemchoices = sorted(glob.glob(os.environ['TEMPO2']+'/ephemeris/*'))
                matchers = ['421.bsp', '430t.bsp', '435t.bsp', '436t.bsp']
                ephemfiles = [s for s in ephemchoices if any(xs in s for xs in matchers)]

                self.planet_ssb = OrderedDict()
                for eph in ephemfiles:

                    if '436' in eph:
                        ephemname = 'DE436'
                    elif '435' in eph:
                        ephemname = 'DE435'
                    elif '430' in eph:
                        ephemname = 'DE430'
                    elif '421' in eph:
                        ephemname = 'DE421'

                    kernel = SPK.open(eph)
                    jd = self.toas + mjd2jd
                    self.planet_ssb[ephemname] = np.zeros((self.toas.shape[0],9,6))
                    for ii in range(9):
                        position, velocity = kernel[0,ii+1].compute_and_differentiate(jd)
                        position = np.hstack([position.T * 1e3 / sc.c,
                                              velocity.T * 1e3 / sc.c / 86400.])
                        self.planet_ssb[ephemname][:,ii,:] = position
            else:
                for ii in range(1,10):
                    tag = 'DMASSPLANET'+str(ii)
                    self.T2psr[tag].val = 0.0
                self.T2psr.formbats()
                self.planet_ssb = OrderedDict.fromkeys([self.ephemname])
                self.planet_ssb[self.ephemname] = np.zeros((len(self.T2psr.toas()),9,6))
                self.planet_ssb[self.ephemname][:,0,:] = self.T2psr.mercury_ssb
                self.planet_ssb[self.ephemname][:,1,:] = self.T2psr.venus_ssb
                self.planet_ssb[self.ephemname][:,2,:] = self.T2psr.earth_ssb
                self.planet_ssb[self.ephemname][:,3,:] = self.T2psr.mars_ssb
                self.planet_ssb[self.ephemname][:,4,:] = self.T2psr.jupiter_ssb
                self.planet_ssb[self.ephemname][:,5,:] = self.T2psr.saturn_ssb
                self.planet_ssb[self.ephemname][:,6,:] = self.T2psr.uranus_ssb
                self.planet_ssb[self.ephemname][:,7,:] = self.T2psr.neptune_ssb
                self.planet_ssb[self.ephemname][:,8,:] = self.T2psr.pluto_ssb

            if startMJD is not None and endMJD is not None:
                for eph in self.planet_ssb:
                    self.planet_ssb[eph] = \
                      self.planet_ssb[eph][self.tmask,:,:]

            print "--> Grabbed the planet position-vectors at the pulsar timestamps."

        self.isort, self.iisort = None, None
        if 'pta' in self.flags:
            if 'NANOGrav' in list(set(self.flagvals['pta'])):
                # now order everything
                try:
                    self.isort, self.iisort = utils.argsortTOAs(self.toas, self.flagvals['group'],
                                                      which='jitterext', dt=jitterbin/86400.)
                except KeyError:
                    self.isort, self.iisort = utils.argsortTOAs(self.toas, self.flagvals['f'],
                                                      which='jitterext', dt=jitterbin/86400.)

                # sort data
                self.toas = self.toas[self.isort]
                self.toaerrs = self.toaerrs[self.isort]
                self.res = self.res[self.isort]
                self.obs_freqs = self.obs_freqs[self.isort]
                self.Mmat = self.Mmat[self.isort, :]
                for eph in self.roemer:
                    self.roemer[eph] = self.roemer[eph][self.isort]
                if planetssb:
                    for eph in self.planet_ssb:
                        self.planet_ssb[eph] = \
                          self.planet_ssb[eph][self.isort, :, :]

                print "--> Initial sorting of data."

        # get the sky position
        # check for B name
        if 'B' in self.T2psr.name:
            epoch = '1950'
        else:
            epoch = '2000'
        if 'RAJ' and 'DECJ' in self.T2psr.pars():
            self.raj = np.double(self.T2psr['RAJ'].val)
            self.decj = np.double(self.T2psr['DECJ'].val)

            self.psr_locs = [self.raj, self.decj]

            eq = ephem.Equatorial(self.T2psr['RAJ'].val,
                                  self.T2psr['DECJ'].val)
            ec = ephem.Ecliptic(eq, epoch=epoch)
            self.elong = np.double(ec.lon)
            self.elat = np.double(ec.lat)

        elif 'ELONG' and 'ELAT' in self.T2psr.pars():
            self.elong = np.double(self.T2psr['ELONG'].val)
            self.elat = np.double(self.T2psr['ELAT'].val)

            ec = ephem.Ecliptic(self.elong, self.elat)
            eq = ephem.Equatorial(ec, epoch=epoch)
            self.raj = np.double(eq.ra)
            self.decj = np.double(eq.dec)

            self.psr_locs = [self.raj, self.decj]

        print "--> Grabbed the pulsar position."

        ################################################################################################

        # These are all the relevant system flags used by the PTAs.
        system_flags = ['group','f','sys','g','h']
        self.sysflagdict = OrderedDict.fromkeys(system_flags)

        # Put the systems into a dictionary which
        # has the locations of their toa placements.
        for systm in self.sysflagdict:
            try:
                if systm in self.flags:
                    sys_uflagvals = list(set(self.flagvals[systm]))
                    self.sysflagdict[systm] = OrderedDict.fromkeys(sys_uflagvals)
                    for kk,subsys in enumerate(sys_uflagvals):
                        if self.isort is not None:
                            self.sysflagdict[systm][subsys] = \
                              np.where(self.flagvals[systm][self.isort] == sys_uflagvals[kk])
                        elif self.isort is None:
                            self.sysflagdict[systm][subsys] = \
                              np.where(self.flagvals[systm] == sys_uflagvals[kk])
            except KeyError:
                pass

        # If we have some NANOGrav data, then separate
        # this off for later ECORR assignment.
        if 'pta' in self.flags:
            pta_names = list(set(self.flagvals['pta']))
            pta_mask = [self.flagvals['pta'][self.isort]==ptaname for ptaname in pta_names]
            pta_maskdict = OrderedDict.fromkeys(pta_names)
            for ii,item in enumerate(pta_maskdict):
                pta_maskdict[item] = pta_mask[ii]
            if len(pta_names)!=0 and ('NANOGrav' in pta_names):
                try:
                    nanoflagdict = OrderedDict.fromkeys(['nano-f'])
                    nano_flags = list(set(self.flagvals['group'][self.isort][pta_maskdict['NANOGrav']]))
                    nanoflagdict['nano-f'] = OrderedDict.fromkeys(nano_flags)
                    for kk,subsys in enumerate(nano_flags):
                        nanoflagdict['nano-f'][subsys] = \
                          np.where(self.flagvals['group'][self.isort] == nano_flags[kk])
                    self.sysflagdict.update(nanoflagdict)
                except KeyError:
                    nanoflagdict = OrderedDict.fromkeys(['nano-f'])
                    nano_flags = list(set(self.flagvals['f'][self.isort][pta_maskdict['NANOGrav']]))
                    nanoflagdict['nano-f'] = OrderedDict.fromkeys(nano_flags)
                    for kk,subsys in enumerate(nano_flags):
                        nanoflagdict['nano-f'][subsys] = \
                          np.where(self.flagvals['f'][self.isort] == nano_flags[kk])
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

        if 'pta' in self.flags:
            if 'NANOGrav' in pta_names:
                # now order everything
                try:
                    dummy_flags = self.flagvals['group'][self.isort]
                except KeyError:
                    dummy_flags = self.flagvals['f'][self.isort]

                print "--> Sorted data."

                # get quantization matrix
                avetoas, self.Umat, Ui = utils.quantize_split(self.toas, dummy_flags, dt=jitterbin/86400., calci=True)
                print "--> Computed quantization matrix."

                self.detsig_avetoas = avetoas.copy()
                self.detsig_Uinds = utils.quant2ind(self.Umat)

                # get only epochs that need jitter/ecorr
                self.Umat, avetoas, aveflags = utils.quantreduce(self.Umat, avetoas, dummy_flags)
                print "--> Excized epochs without jitter."

                # get quantization indices
                self.Uinds = utils.quant2ind(self.Umat)
                self.epflags = dummy_flags[self.Uinds[:, 0]]

                print "--> Checking TOA sorting and quantization..."
                print utils.checkTOAsort(self.toas, dummy_flags, which='jitterext', dt=jitterbin/86400.)
                print utils.checkquant(self.Umat, dummy_flags)
                print "...Finished checks."

        # perform SVD of design matrix to stabilise
        if fastDesign:
            print "--> Stabilizing the design matrix the fast way..."

            Mm = self.Mmat.copy()
            norm = np.sqrt(np.sum(Mm ** 2, axis=0))
            Mm /= norm

            self.Gc = Mm
        else:
            print "--> Performing SVD of design matrix for stabilization..."

            if makeGmat:
                u,s,v = np.linalg.svd(self.Mmat)

                self.G = u[:,len(s):len(u)]
                self.Gres = np.dot(self.G.T, self.res)

                self.Gc =  u[:,:len(s)]

            elif not makeGmat:
                u,s,v = np.linalg.svd(self.Mmat, full_matrices=0)

                self.Gc =  u

        print "--> Done reading in pulsar :-) \n"


    def makeFred(self, fqs_red, wgts_red, Ttot, phaseshift=False):

        self.Fred, self.ranphase = \
          utils.createFourierDesignmatrix_red(self.toas, fqs_red, wgts_red,
                                              pshift=phaseshift, Tspan=Ttot)

    def makeFdm(self, fqs_dm, wgts_dm, obs_freqs, Ttot):

        self.Fdm = utils.createFourierDesignmatrix_dm(self.toas, fqs_dm, wgts_dm,
                                                      obs_freqs=self.obs_freqs,
                                                      Tspan=Ttot)

    def makeFeph(self, fqs_eph, wgts_eph, psr_locs, Ttot):

        self.Fephx, self.Fephy, self.Fephz = \
          utils.createFourierDesignmatrix_eph(self.toas, fqs_eph, wgts_eph,
                                              psr_locs=self.psr_locs, Tspan=Ttot)

    def makeFtot(self, nmodes, Ttot, phaseshift=False):

        self.Fred, self.ranphase = \
          utils.createFourierDesignmatrix_red(self.toas, fqs_red, wgts_red,
                                             pshift=phaseshift, Tspan=Ttot)
        self.Fdm = utils.createFourierDesignmatrix_dm(self.toas, fqs_dm, wgts_dm,
                                                      self.obs_freqs, Tspan=Ttot)

        self.Ftot = np.append(self.Fred, self.Fdm, axis=1)

    def makeTe(self, Ttot, fqs_red, wgts_red, makeDM=False, fqs_dm=None, wgts_dm=None,
               makeEph=False, jplBasis=False, fqs_eph=None, wgts_eph=None, ephFreqs=None,
               makeClk=False, clkDesign=False,
               makeBand=False, bands=None,
               phaseshift=False):

        self.Fred, self.ranphase = \
          utils.createFourierDesignmatrix_red(self.toas, fqs_red, wgts_red,
                                              pshift=phaseshift, Tspan=Ttot)

        self.Ftot = self.Fred
        if makeDM:
            if fqs_dm is None:
                fqs_tmp = fqs_red
                wgts_tmp = wgts_red
            else:
                fqs_tmp = fqs_dm
                wgts_tmp = wgts_dm
            self.Fdm = utils.createFourierDesignmatrix_dm(self.toas, fqs_tmp, wgts_tmp,
                                                          self.obs_freqs, Tspan=Ttot)
            self.Ftot = np.append(self.Ftot, self.Fdm, axis=1)
        if makeEph:
            if jplBasis:
                Fmother = np.load('./data/jplephbasis/Fmother.npy')
                mjd = np.load('./data/jplephbasis/mjd.npy')

                posvec = np.array([np.sin(np.pi/2.- self.elat)*np.cos(self.elong),
                                   np.sin(np.pi/2.- self.elat)*np.sin(self.elong),
                                   np.cos(np.pi/2.- self.elat)])

                Fproj = np.dot(Fmother[:,:,:],posvec)
                Feph = np.vstack(np.interp(self.toas,mjd,Fproj[:,ii])
                                 for ii in range(Fproj.shape[1])).T
                self.Ftot = np.append(self.Ftot, Feph, axis=1)
            else:
                if fqs_eph is None:
                    fqs_tmp = fqs_red
                    wgts_tmp = wgts_red
                else:
                    fqs_tmp = fqs_eph
                    wgts_tmp = wgts_eph
                self.Fephx, self.Fephy, self.Fephz = \
                  utils.createFourierDesignmatrix_eph(self.toas, fqs_tmp, wgts_tmp, self.psrPos,
                                                      Tspan=Ttot, input_freqs=ephFreqs)
                self.Ftot = np.append(self.Ftot, self.Fephx, axis=1)
                self.Ftot = np.append(self.Ftot, self.Fephy, axis=1)
                self.Ftot = np.append(self.Ftot, self.Fephz, axis=1)
        if makeClk and clkDesign:
            self.Fclk, _ = utils.createFourierDesignmatrix_red(self.toas, fqs_red, wgts_red,
                                                               pshift=False, Tspan=Ttot)
            self.Ftot = np.append(self.Ftot, self.Fclk, axis=1)
        if makeBand:
            if fqs_band is None:
                fqs_tmp = fqs_red
                wgts_tmp = wgts_red
            else:
                fqs_tmp = fqs_band
                wgts_tmp = wgts_band
            ##
            if bands is None:
                bands = np.array([0.0, 1.0, 2.0, 3.0])
            elif bands is not None:
                bands = np.array([float(item) for item in bands.split(',')])

            Fband_tmp = utils.createFourierDesignmatrix_red(self.toas, fqs_tmp, wgts_tmp,
                                                            pshift=False, Tspan=Ttot)
            for ii in range(len(bands)-1):
                Fband_dummy = Fband_tmp.copy()
                Fband_dummy[np.logical(self.obs_freqs > 1e9*bands[ii],
                                       self.obs_freqs <= 1e9*bands[ii+1]),:] = 0.0
                self.Ftot = np.append(self.Ftot, Fband_dummy, axis=1)

        self.Te = np.append(self.Gc, self.Ftot, axis=1)




######################
######################

class PsrObjFromH5(object):
    h5Obj = None
    psr_locs = None
    raj = None
    decj = None
    elong = None
    elat = None
    psrPos = None
    parfile = None
    timfile = None
    noisefile = None
    toas = None
    toaerrs = None
    res = None
    obs_freqs = None
    G = None
    Mmat = None
    sysflagdict = None
    tmask = None
    isort = None
    iisort = None
    ephemeris = None
    ephemname = None
    roemer = None
    Fred = None
    Fdm = None
    Fephx = None
    Fephy = None
    Fephz = None
    Fclk = None
    Ftot = None
    ranphase = None
    diag_white = None
    res_prime = None
    Ftot_prime = None
    Gc = None
    Te = None
    name = "J0000+0000"
    Gres = None
    epflags = None
    detsig_avetoas = None
    detsig_Uinds = None
    t2efacs = None
    t2equads = None
    t2ecorrs = None
    parRedamp = None
    parRedind = None
    parDMamp = None
    parDMind = None
    efacs = None
    equads = None
    ecorrs = None
    Redamp = None
    Redind = None
    DMamp = None
    DMind = None
    planet_ssb = None

    def __init__(self, h5Obj):
        self.h5Obj = h5Obj
        self.parfile = None
        self.timfile = None
        self.noisefile = None
        self.psr_locs = None
        self.raj = None
        self.decj = None
        self.elong = None
        self.elat = None
        self.psrPos = None
        self.toas = None
        self.toaerrs = None
        self.res = None
        self.obs_freqs = None
        self.G = None
        self.Mmat = None
        self.Fred = None
        self.Fdm = None
        self.Fephx = None
        self.Fephy = None
        self.Fephz = None
        self.Fclk = None
        self.Ftot = None
        self.ranphase = None
        self.diag_white = None
        self.res_prime = None
        self.Ftot_prime = None
        self.Gc = None
        self.Te = None
        self.Umat = None
        self.Uinds = None
        self.name = "J0000+0000"
        self.sysflagdict = None
        self.tmask = None
        self.isort = None
        self.iisort = None
        self.ephemeris = None
        self.ephemname = None
        self.roemer = None
        self.Gres = None
        self.epflags = None
        self.detsig_avetoas = None
        self.detsig_Uinds = None
        self.t2efacs = None
        self.t2equads = None
        self.t2ecorrs = None
        self.parRedamp = None
        self.parRedind = None
        self.parDMamp = None
        self.parDMind = None
        self.efacs = None
        self.equads = None
        self.ecorrs = None
        self.Redamp = None
        self.Redind = None
        self.DMamp = None
        self.DMind = None
        self.planet_ssb = None

    """
    Read data from hdf5 file into pulsar object
    """
    def grab_all_vars(self, rescale=True, sysflag_target=None):

        print "--> Extracting {0} from hdf5 file".format(self.h5Obj['name'].value)

        # basic quantities
        self.name = self.h5Obj['name'].value
        self.parfile = self.h5Obj['parfilepath'].value
        self.timfile = self.h5Obj['timfilepath'].value
        try:
            self.noisefile = self.h5Obj['noisefilepath'].value
        except:
            self.noisefile = None

        self.toas = self.h5Obj['TOAs'].value
        self.res = self.h5Obj['postfitRes'].value
        self.toaerrs = self.h5Obj['toaErr'].value
        self.obs_freqs = self.h5Obj['freq'].value

        self.psr_locs = self.h5Obj['psrlocs'].value
        self.raj = self.h5Obj['raj'].value
        self.decj = self.h5Obj['decj'].value
        self.elong = self.h5Obj['elong'].value
        self.elat = self.h5Obj['elat'].value
        self.psrPos = self.h5Obj['psrPos'].value

        self.ephemeris = self.h5Obj['ephemeris'].value
        self.ephemname = self.h5Obj['ephemname'].value

        self.roemer = pickle.loads(self.h5Obj['RoemerDict'].value)

        try:
            self.planet_ssb = pickle.loads(self.h5Obj['PlanetSSBDict'].value)
        except:
            self.planet_ssb = None

        self.Mmat = self.h5Obj['designmatrix'].value
        try:
            self.G = self.h5Obj['Gmatrix'].value
            self.Gres = self.h5Obj['Gres'].value
        except:
            self.G = None
            self.Gres = None
        self.Gc = self.h5Obj['GCmatrix'].value
        try:
            self.Umat = self.h5Obj['QuantMat'].value
            self.Uinds = self.h5Obj['QuantInds'].value
            self.epflags = self.h5Obj['EpochFlags'].value
            self.detsig_avetoas = self.h5Obj['DetSigAveToas'].value
            self.detsig_Uinds = self.h5Obj['DetSigQuantInds'].value
        except:
            self.Umat = None
            self.Uinds = None
            self.epflags = None
            self.detsig_avetoas = None
            self.detsig_Uinds = None

        try:
            self.isort = self.h5Obj['isort'].value
            self.iisort = self.h5Obj['iisort'].value
        except:
            self.isort = None
            self.iisort = None

        try:
            self.tmask = self.h5Obj['tmask'].value
        except:
            self.tmask = None

        self.sysflagdict = pickle.loads(self.h5Obj['SysFlagDict'].value)

        # Let's rip out EFACS, EQUADS and ECORRS from parfile
        parlines = self.h5Obj['parfile'].value.split('\n')
        t2efacs = []
        t2equads = []
        t2ecorrs = []
        for ll in parlines:
            if 'T2EFAC' in ll:
                t2efacs.append([ll.split()[2], np.double(ll.split()[3])])
            if 'T2EQUAD' in ll:
                t2equads.append([ll.split()[2], np.double(ll.split()[3])*1e-6])
            if 'ECORR' in ll:
                t2ecorrs.append([ll.split()[2], np.double(ll.split()[3])*1e-6])

        self.t2efacs = OrderedDict(t2efacs)
        self.t2equads = OrderedDict(t2equads)
        self.t2ecorrs = OrderedDict(t2ecorrs)

        # Let's rip out the red-noise/DM properties if present
        self.parRedamp = 1e-20
        self.parRedind = 0.0
        self.parDMamp = 1e-20
        self.parDMind = 0.0
        for ll in parlines:
            if 'RNAMP' in ll:
                self.parRedamp = np.double(ll.split()[1]) # 1e-6 * f1yr * np.sqrt(12.0*np.pi**2.0) * np.double(ll.split()[1])
            elif 'TNRedAmp' in ll:
                self.parRedamp = 10.0**np.double(ll.split()[1])

            if 'RNIDX' in ll:
                self.parRedind = -np.double(ll.split()[1])
            elif 'TNRedGam' in ll:
                self.parRedind = np.double(ll.split()[1])

            if 'TNDMAmp' in ll:
                self.parDMamp = 10.0**np.double(ll.split()[1]) * np.sqrt(12.0*np.pi**2.0)
            if 'TNDMGam' in ll:
                self.parDMind = np.double(ll.split()[1])

        # Let's also find single pulsar analysis EFACS, EQUADS, ECORRS
        self.Redamp = 1e-20
        self.Redind = 0.0
        self.DMamp = 1e-20
        self.DMind = 0.0
        if self.noisefile is not None:
            noiselines = self.h5Obj['noisefile'].value.split('\n')
            efacs = []
            equads = []
            ecorrs = []
            for ll in noiselines:
                if 'efac' in ll:
                    efacs.append([ll.split()[0].split('efac-')[1], np.double(ll.split()[1])])
                if 'equad' in ll:
                    equads.append([ll.split()[0].split('equad-')[1], 10.0**np.double(ll.split()[1])])
                if 'jitter' in ll:
                    ecorrs.append([ll.split()[0].split('jitter_q-')[1], 10.0**np.double(ll.split()[1])])

            self.efacs = OrderedDict(efacs)
            self.equads = OrderedDict(equads)
            self.ecorrs = OrderedDict(ecorrs)

            # Let's get the red noise properties from single-pulsar analysis
            self.Redamp = 1e-20
            self.Redind = 0.0
            self.DMamp = 1e-20
            self.DMind = 0.0
            for ll in noiselines:
                if 'RN-Amplitude' in ll:
                    self.Redamp = 10.0**np.double(ll.split()[1]) # 1e-6 * f1yr * np.sqrt(12.0*np.pi**2.0) * np.double(ll.split()[1])
                if 'RN-spectral-index' in ll:
                    self.Redind = np.double(ll.split()[1])
                if 'DM-Amplitude' in ll:
                    # special edit below since AP's DM-amps use TN convention
                    self.DMamp = 10.0**np.double(ll.split()[1]) * np.sqrt(12.0*np.pi**2.0)
                if 'DM-spectral-index' in ll:
                    self.DMind = np.double(ll.split()[1])

            # Time to rescale the TOA uncertainties by single-pulsar EFACS and EQUADS
            if sysflag_target is not None:
                systems = self.sysflagdict[sysflag_target]
            else:
                if self.sysflagdict['group'] is not None:
                    # for nanograv/ipta
                    systems = self.sysflagdict['group']
                elif self.sysflagdict['f'] is not None:
                    # for nanograv
                    systems = self.sysflagdict['f']
                elif self.sysflagdict['sys'] is not None:
                    # for the epta
                    systems = self.sysflagdict['sys']
                elif self.sysflagdict['be'] is not None:
                    # for nanograv 5-yr
                    systems = self.sysflagdict['be']

            if rescale:
                tmp_errs = self.toaerrs.copy()

                for sysname in systems:
                    tmp_errs[systems[sysname]] *= self.efacs[sysname]

                t2equad_bit = np.ones(len(tmp_errs))
                for sysname in systems:
                    t2equad_bit[systems[sysname]] *= self.equads[sysname]

                tmp_errs = np.sqrt( tmp_errs**2.0 + t2equad_bit**2.0 )
                self.toaerrs = tmp_errs


        print "--> Done extracting pulsar from hdf5 file :-) \n"

    def makeFred(self, fqs_red, wgts_red, Ttot, phaseshift=False):

        self.Fred, self.ranphase = \
          utils.createFourierDesignmatrix_red(self.toas, fqs_red, wgts_red,
                                              pshift=phaseshift, Tspan=Ttot)

    def makeFdm(self, fqs_dm, wgts_dm, obs_freqs, Ttot):

        self.Fdm = utils.createFourierDesignmatrix_dm(self.toas, fqs_dm, wgts_dm,
                                                      obs_freqs=self.obs_freqs,
                                                      Tspan=Ttot)

    def makeFeph(self, fqs_eph, wgts_eph, psr_locs, Ttot):

        self.Fephx, self.Fephy, self.Fephz = \
          utils.createFourierDesignmatrix_eph(self.toas, fqs_eph, wgts_eph,
                                              psr_locs=self.psr_locs, Tspan=Ttot)

    def makeFtot(self, nmodes, Ttot, phaseshift=False):

        self.Fred, self.ranphase = \
          utils.createFourierDesignmatrix_red(self.toas, fqs_red, wgts_red,
                                             pshift=phaseshift, Tspan=Ttot)
        self.Fdm = utils.createFourierDesignmatrix_dm(self.toas, fqs_dm, wgts_dm,
                                                      self.obs_freqs, Tspan=Ttot)

        self.Ftot = np.append(self.Fred, self.Fdm, axis=1)

    def makeTe(self, Ttot, fqs_red, wgts_red, makeDM=False, fqs_dm=None, wgts_dm=None,
               makeEph=False, jplBasis=False, fqs_eph=None, wgts_eph=None, ephFreqs=None,
               makeClk=False, clkDesign=False,
               makeBand=False, bands=None,
               phaseshift=False):

        self.Fred, self.ranphase = \
          utils.createFourierDesignmatrix_red(self.toas, fqs_red, wgts_red,
                                              pshift=phaseshift, Tspan=Ttot)

        self.Ftot = self.Fred
        if makeDM:
            if fqs_dm is None:
                fqs_tmp = fqs_red
                wgts_tmp = wgts_red
            else:
                fqs_tmp = fqs_dm
                wgts_tmp = wgts_dm
            self.Fdm = utils.createFourierDesignmatrix_dm(self.toas, fqs_tmp, wgts_tmp,
                                                          self.obs_freqs, Tspan=Ttot)
            self.Ftot = np.append(self.Ftot, self.Fdm, axis=1)
        if makeEph:
            if jplBasis:
                Fmother = np.load('./data/jplephbasis/Fmother.npy')
                mjd = np.load('./data/jplephbasis/mjd.npy')

                posvec = np.array([np.sin(np.pi/2.- self.elat)*np.cos(self.elong),
                                   np.sin(np.pi/2.- self.elat)*np.sin(self.elong),
                                   np.cos(np.pi/2.- self.elat)])

                Fproj = np.dot(Fmother[:,:,:],posvec)
                Feph = np.vstack(np.interp(self.toas,mjd,Fproj[:,ii])
                                 for ii in range(Fproj.shape[1])).T
                self.Ftot = np.append(self.Ftot, Feph, axis=1)
            else:
                if fqs_eph is None:
                    fqs_tmp = fqs_red
                    wgts_tmp = wgts_red
                else:
                    fqs_tmp = fqs_eph
                    wgts_tmp = wgts_eph
                self.Fephx, self.Fephy, self.Fephz = \
                  utils.createFourierDesignmatrix_eph(self.toas, fqs_tmp, wgts_tmp, self.psrPos,
                                                      Tspan=Ttot, input_freqs=ephFreqs)
                self.Ftot = np.append(self.Ftot, self.Fephx, axis=1)
                self.Ftot = np.append(self.Ftot, self.Fephy, axis=1)
                self.Ftot = np.append(self.Ftot, self.Fephz, axis=1)
        if makeClk and clkDesign:
            self.Fclk, _ = utils.createFourierDesignmatrix_red(self.toas, fqs_red, wgts_red,
                                                               pshift=False, Tspan=Ttot)
            self.Ftot = np.append(self.Ftot, self.Fclk, axis=1)
        if makeBand:
            if fqs_band is None:
                fqs_tmp = fqs_red
                wgts_tmp = wgts_red
            else:
                fqs_tmp = fqs_band
                wgts_tmp = wgts_band
            ##
            if bands is None:
                bands = np.array([0.0, 1.0, 2.0, 3.0])
            elif bands is not None:
                bands = np.array([float(item) for item in bands.split(',')])

            Fband_tmp = utils.createFourierDesignmatrix_red(self.toas, fqs_tmp, wgts_tmp,
                                                            pshift=False, Tspan=Ttot)
            for ii in range(len(bands)-1):
                Fband_dummy = Fband_tmp.copy()
                Fband_dummy[np.logical(self.obs_freqs > 1e9*bands[ii],
                                       self.obs_freqs <= 1e9*bands[ii+1]),:] = 0.0
                self.Ftot = np.append(self.Ftot, Fband_dummy, axis=1)

        self.Te = np.append(self.Gc, self.Ftot, axis=1)
