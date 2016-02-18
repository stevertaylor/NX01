#!/usr/bin/env python

"""
Created by stevertaylor
Copyright (c) 2014 Stephen R. Taylor

Code contributions by Rutger van Haasteren (piccard) and Justin Ellis (PAL/PAL2).

"""

from __future__ import division
import numpy as np
from numpy import *
import os, optparse, corner, json
import h5py as h5

import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import FormatStrFormatter, \
     LinearLocator, NullFormatter, NullLocator, \
     AutoMinorLocator
import newcmaps

import NX01_bayesutils as bu
import NX01_psr

import scipy
import scipy.interpolate as interp
from scipy.interpolate import interp1d

import acor

##################
##################

parser = optparse.OptionParser(description = "NX01 - It's been a long road, getting from there to here...")

parser.add_option('--parentpath', dest='parentpath', action='store', type=str,
                   help='Provide the name of the parent directory containing your run directories')
parser.add_option('--chaindir', dest='chaindir', action='store', type=str,
                   help='Provide the directory name containing your chain files')
parser.add_option('--intelburn', dest='intelburn', action='store_true', default=False,
                   help='Do you want to try intelligent burning-in and chain thinning?')
parser.add_option('--manualburn', dest='manualburn', action='store', type=int, default=1000,
                   help='How many samples do you want to cut from the start of the chain as burn-in?')
parser.add_option('--strainAnis', dest='strainAnis', action='store_true', default=False,
                   help='Do you want to investigate strain anisotropy rather than power anisotropy [default = False]?')

(args, x) = parser.parse_args()


#################
# Read in chain
#################

try:
    chain = np.loadtxt(args.parentpath+'/'+args.chaindir+'/chain_1.0.txt')
except IOError:
    chain = np.loadtxt(args.parentpath+'/'+args.chaindir+'/chain_1.txt')
    print chain.shape

param_list = np.genfromtxt(args.parentpath+'/'+args.chaindir+'/parameter_list.txt',dtype=str)
print param_list

print "Chain has been read-in"

jsonModel = args.parentpath+'/'+args.chaindir+'/run_args.json'
with open(jsonModel) as json_file:
    json_data = json.load(json_file)
    json_file.close()

print "Model has been read-in"

psr_pathinfo = np.genfromtxt(json_data['psrlist'], dtype=str, skip_header=2)

if json_data['from_h5']:
    tmp_psr = []
    if json_data['psrIndices'] is not None:
        psr_inds = [int(item) for item in json_data['psrIndices'].split(',')]
        for ii,tmp_name in zip(psr_inds,psr_pathinfo[psr_inds,0]):
            tmp_psr.append(h5.File(psr_pathinfo[ii,1], 'r')[tmp_name])
    else:
        for ii,tmp_name in enumerate(psr_pathinfo[json_data['psrStartIndex']:json_data['psrEndIndex'],0],
                                     start=json_data['psrStartIndex']):
            tmp_psr.append(h5.File(psr_pathinfo[ii,1], 'r')[tmp_name])

    psr = [NX01_psr.PsrObjFromH5(p) for p in tmp_psr]
    [p.grab_all_vars() for p in psr]

    psr_positions = [np.array([psr[ii].psr_locs[0],
                           np.pi/2. - psr[ii].psr_locs[1]])
                           for ii in range(len(psr))]
    positions = np.array(psr_positions).copy()

##################
# Burn in samples
##################

if args.intelburn:

    loglike = chain[-3] # likelihood column
    dim = chain.shape[1]-4 # dimensionality of search
    likemax = loglike.max() # max likelihood

    burniter = np.where(loglike > (likemax-dim/2.))[0]
    burntdata = chain[burniter:,:]

    # GWB amplitude index
    amp_ind = int(param_list[param_list[:,1]=='Agwb',0][0])

    corrlength, mean, sigma = acor.acor(burntdata[:,amp_ind])  
    indsamples = burntdata[::int(corrlength)]
    chain = indsamples

if not args.intelburn: chain = chain[args.manualburn:,:]
    
############################################################
# Doing a quick plot to manually cut-off the burn-in stage
############################################################

if 'gam4p33' in args.chaindir:
    # checking Agwb
    Agwb = chain[:,int(param_list[param_list[:,1]=='Agwb',0][0])]

    gwb_params = np.zeros((len(chain),1))
    gwb_params[:,0] = Agwb

    bu.makePostPlots_show(gwb_params,ndim=1,labels=[r'$A_h$']) 
    plt.show()
else:
    # checking Agwb and gam_gwb
    Agwb = chain[:,int(param_list[param_list[:,1]=='Agwb',0][0])]
    gam_gwb = chain[:,int(param_list[param_list[:,1]=='gam_gwb',0][0])]

    gwb_params = np.zeros((len(chain),2))
    gwb_params[:,0] = Agwb
    gwb_params[:,1] = gam_gwb
    
    bu.makePostPlots_show(gwb_params,ndim=2,labels=[r'$A_h$', r'$\gamma$']) 
    plt.show()


##########################
# Now processing results
##########################

if 'gam4p33' in args.chaindir:

    ################################################
    # make a 1D posterior plot of the GWB amplitude
    ################################################
    
    fig, ax = plt.subplots()
    
    Agwb = chain[:,int(param_list[param_list[:,1]=='Agwb',0][0])]
    bu.makesubplot1d(ax, Agwb, bins=60)

    plt.xlabel('$\log_{10}(A_h)$', fontsize=20)
    plt.tick_params(labelsize=18)

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    plt.show()

    #########################################
    # make a CDF plot to get the upper limit
    #########################################
    
    bu.makeCDF(10.0**Agwb/1e-15, linestyle='k-', linewidth=3.0,
               labels=None, legendbox=True,
               title='$\gamma_{\mathrm{GWB}}=13/3$')

    plt.xlabel(r'$A_h\; /\; 10^{-15}$', fontsize=20)
    plt.ylabel('Cumulative probability', fontsize=20)

    plt.tick_params(labelsize=18)
    
    plt.show()

elif 'gamVary' in args.chaindir:
    
    Agwb = chain[:,int(param_list[param_list[:,1]=='Agwb',0][0])]
    gam_gwb = chain[:,int(param_list[param_list[:,1]=='gam_gwb',0][0])]

    gwb_params = np.zeros((len(chain),2))
    gwb_params[:,0] = Agwb
    gwb_params[:,1] = gam_gwb

    ##################################################
    # make a quick corner plot of Agwb versus gam_gwb
    ##################################################
    
    corner.corner(gwb_params, labels=[r'$\log_{10}(A_h)$', r'$\gamma$'])
    plt.show()

    ################################################################################
    # make an upper limit plot for the GWB amplitude as a function of spectral index
    ################################################################################
    
    #bu.upperlimitplot2d(Agwb, gam_gwb, sigma=0.68, bins=30, logA=True, \
    #                 labels=[r'$\gamma=3-2\alpha$', r'$A_h$'], hold=True, linestyle='solid', \
    #                 color='black', linewidth=3.0, leglabel=r'68\% upper-limit')
    bu.upperlimitplot2d(Agwb, gam_gwb, sigma=0.95, bins=30, log=True, \
                     labels=[r'$\gamma=3-2\alpha$', r'$A_h$'], hold=True, linestyle='dashed', \
                     color='black', linewidth=3.0)
    plt.grid(which='major')
    plt.grid(which='minor')
    plt.legend(fancybox=True, shadow=True)
    plt.tick_params(labelsize=20)
    plt.show()


if 'Lmax' in args.chaindir:
    st = args.chaindir.split('_')
    for ii in st:
        if 'Lmax' in ii: LMAX = int(ii[-1])

############################################
# If the search is anisotropic then make a
# plot of the angular power spectrum
############################################

if 'Lmax0' not in args.chaindir:

    fig, ax = plt.subplots()

    #if 'EvAnis' in args.chaindir:
    #clm = chain[var:,-4 - 5*( (LMAX+1)**2 - 1):-4]  # assuming 5 windows
    #else:

    clm_inds = np.array(['clm' in p for p in param_list[:,1]])
    clm_inds = np.append(clm_inds,np.array([False,False,False,False]))
    clm = chain[:,clm_inds]
    # clm = chain[var:,-4 - 1*( (LMAX+1)**2 - 1):-4 - 0*( (LMAX+1)**2 - 1)]


    Cl = [0.0]*(LMAX+1)
    Cl[0] = 4.0*np.pi*np.ones(clm.shape[0])
    ll = np.arange(LMAX+1)

    for ii in ll[:-1]:
        Cl[ii+1] = np.sum( clm[:,(ll[ii]+1)**2-1:(ll[ii+1]+1)**2-1]**2
                           / (2*ll[ii+1]+1.) , axis=1)

    Cl = np.array(Cl)
    if args.strainAnis:
        Cl = np.array([Cl[ii,:]*(10**(4*Agwb)) for ii in range(Cl.shape[0])])
        #print Cl.shape

    ######################################################
    # Make posterior density map of angular power spectrum
    ######################################################
    
    delta_ell = np.linspace(0.0,LMAX*1.0,1000)
    func = np.zeros((len(clm), len(delta_ell)))

    for ii in range(len(clm)):
        if not args.strainAnis:
            tmp = interp1d(ll, Cl[:,ii]/(4.0*np.pi) )
        elif args.strainAnis:
            tmp = interp1d(ll, np.log10((Cl[:,ii]/(4.0*np.pi))**(1./4.)) )
        func[ii,:] = tmp(delta_ell) 

    xx = np.tile(delta_ell,len(clm))
    yy = func.reshape((len(xx),))

    # get the central and bound of the lines
    func_c = np.zeros(len(delta_ell))
    func_r = np.zeros(len(delta_ell[10:]))
    row,col = func.shape

    for ii in range(len(delta_ell)):
        dat = func[:,ii]
        func_c[ii] = np.mean(dat)

    for ii in range(10,len(delta_ell)):
        dat = func[:,ii]
        r = bu.postdensity_getupper(dat, 100, 0.95)
        func_r[ii-10] = r

    # 2D hist of the line distributions
    if not args.strainAnis:
        H, xedges, yedges = np.histogram2d(xx, yy, bins=(100, 100),
                                    range=([0.0,1.0*LMAX], [0.0,1.0]))
    elif args.strainAnis:
        H, xedges, yedges = np.histogram2d(xx, yy, bins=(100, 100),
                                    range=([0.0,1.0*LMAX], [-18.,-11.]))

    H = H.transpose()

    x2edges = (xedges[:-1] + xedges[1:]) / 2
    y2edges = (yedges[:-1] + yedges[1:]) / 2
    mxx, mxy = np.meshgrid(x2edges, y2edges)

    plt.contourf(mxx, mxy, H/np.sum(H), 50, cmap='Greys')
    plt.colorbar()

    plt.plot(delta_ell, func_c, linestyle='solid',
             color='black', linewidth=3.0)
    plt.plot(delta_ell[10:], func_r, linestyle='dashed',
             color='red', linewidth=3.0)

    ax.set_xticks(ll)
    plt.tick_params(labelsize=18)
    plt.xlabel('$l$', fontsize=20)
    if not args.strainAnis:
        plt.ylabel('$C_l/4\pi$', fontsize=20)
    elif args.strainAnis:
        plt.ylabel('$(A_h^4 C_l/4\pi)^{1/4}$', fontsize=20)
    plt.show()

    ###################################
    # Make a posterior-averaged skymap
    ###################################

    if args.strainAnis:
        #strainClm = np.array([clm[:,ii]*(10**(2*Agwb)) for ii in range(clm.shape[1])]).T
        bu.makeSkyMap(clm, lmax=LMAX, cmap=newcmaps.viridis, strain=Agwb, psrs=positions)
    else:
        bu.makeSkyMap(clm, lmax=LMAX, cmap=newcmaps.viridis, psrs=positions)
    plt.show()

    ################################
    ################################
    '''
    fig, ax = plt.subplots()
    plt.contourf(mxx, mxy, H/np.sum(H), 50, cmap='Greys')
    plt.colorbar()

    plt.plot(delta_ell, func_c, linestyle='solid', color='black', linewidth=3.0)
    #plt.plot(delta_ell[10:], func_l, linestyle='dashed', color='red', linewidth=3.0)
    plt.plot(delta_ell[10:], func_r, linestyle='dashed', color='red', linewidth=3.0)
    #fil = open('global_lmax4_upperlimits_prior.dat','w')
    #for ii in range(len(func_r)):
    #    print>>fil, delta_ell[10:][ii], func_r[ii]

    ax.set_xticks(ll)
    plt.xlim(xmin=1)
    tmplist = delta_ell.tolist()
    indl1 = min(range(len(tmplist)), key=lambda i: abs(tmplist[i]-1.0))
    plt.ylim(0.0, np.max(func_r[indl1:]))
    plt.tick_params(labelsize=18)
    plt.xlabel('$l$', fontsize=20)
    plt.ylabel('$C_l/4\pi$', fontsize=20)
    plt.show()
    '''
    
