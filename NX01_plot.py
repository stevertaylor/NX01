#!/usr/bin/env python

"""
Created by stevertaylor
Copyright (c) 2014 Stephen R. Taylor

Code contributions by Rutger van Haasteren (piccard) and Justin Ellis (PAL/PAL2).

"""

import numpy as np
from numpy import *
import os
import optparse
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LinearLocator, NullFormatter, NullLocator, AutoMinorLocator
import matplotlib.ticker
import NX01_bayesutils as bu
#import pylab

parser = optparse.OptionParser(description = 'NX01 - Precursor to the PANTHER Group ENTERPRISE project')

parser.add_option('--chain-dir', dest='chaindir', action='store', type=str,
                   help='Provide the directory name containing chain_1.txt')
parser.add_option('--fix-slope', dest='fix_slope', action='store_true', default=False,
                  help='Did you fix the spectral-slope of the GWB in your search (default = False)')

(args, x) = parser.parse_args()

path = os.getcwd()

chain = np.loadtxt(path+'/chains_Analysis/'+args.chaindir+'/chain_1.txt')

############################################################
# Doing a quick plot to manually cut-off the burn-in stage
############################################################
plt.rcParams['text.usetex'] = True

if args.fix_slope:
    bu.makePostPlots_show(chain,ndim=1,labels=[r'$A_h$']) # checking Agwb
    plt.show()
else:
    bu.makePostPlots_show(chain,ndim=2,labels=[r'$A_h$', r'$\gamma$']) # checking Agwb and gam_gwb
    plt.show()

var = raw_input("\n How many iterations do you want to cut-off as burn-in?: ")
print "\n"
var = int(var)

if args.fix_slope:
    fig, ax = plt.subplots()
    bu.makesubplot1d(ax, chain[:,0], bins=60)
    plt.xlabel('$\log_{10}(A_h)$', fontsize=20)
    plt.tick_params(labelsize=18)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.show()
   
    bu.makeCDF(10.0**chain[:,0]/1e-15, linestyle='k-', linewidth=3.0, labels=None, legendbox=True,\
               title='ML fixed-noise;\n Isotropic search;\n $\gamma_{\mathrm{GWB}}=13/3$')
    plt.xlabel(r'$A_h\; /\; 10^{-15}$', fontsize=20)
    plt.ylabel('Cumulative probability', fontsize=20)
    plt.tick_params(labelsize=18)
    plt.show()
else:
    bu.triplot(chain[var:,0:2], labels=[r'$\log_{10}(A_h)$', r'$\gamma$'])
    plt.show()

    bu.upperlimitplot2d(chain[var:,0], chain[var:,1], sigma=0.68, bins=30, logA=True, \
                     labels=[r'$\gamma=3-2\alpha$', r'$A_h$'], hold=True, linestyle='solid', \
                     color='black', linewidth=3.0, leglabel=r'68\% upper-limit')
    bu.upperlimitplot2d(chain[var:,0], chain[var:,1], sigma=0.95, bins=30, logA=True, \
                     labels=[r'$\gamma=3-2\alpha$', r'$A_h$'], hold=True, linestyle='dashed', \
                     color='black', linewidth=3.0, leglabel=r'95\% upper-limit')
    plt.grid(which='major')
    plt.grid(which='minor')
    plt.legend(fancybox=True, shadow=True)
    plt.tick_params(labelsize=20)
    plt.show()
    
