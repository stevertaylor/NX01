import numpy as np
import libstempo as T
import os, glob
import re
from collections import OrderedDict
from scipy import linalg as sl
import matplotlib.pyplot as plt

import libstempo.plot as LP, libstempo.toasim as LT

#########################################################################
#########################################################################

def quantize_fast(times, dt=1.0, calci=False):
    """ Adapted from libstempo: produce the quantisation matrix fast """
    isort = np.argsort(times)
    
    bucket_ref = [times[isort[0]]]
    bucket_ind = [[isort[0]]]
    
    for i in isort[1:]:
        if times[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(times[i])
            bucket_ind.append([i])
    
    t = np.array([np.mean(times[l]) for l in bucket_ind],'d')
    
    U = np.zeros((len(times),len(bucket_ind)),'d')
    for i,l in enumerate(bucket_ind):
        U[l,i] = 1
    
    rv = (t, U)

    if calci:
        Ui = ((1.0/np.sum(U, axis=0)) * U).T
        rv = (t, U, Ui)

    return rv


def createfourierdesignmatrix_RED(t, nmodes=10, freq=False, Tspan=None):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    @param t: vector of time series in seconds
    @param nmodes: number of fourier coefficients to use
    @param freq: option to output frequencies
    @param Tspan: option to some other Tspan

    @return: F: fourier design matrix
    @return: f: Sampling frequencies (if freq=True)

    """

    N = len(t)
    F = np.zeros((N, 2*nmodes))

    if Tspan is not None:
        T = Tspan
    else:
        T = t.max() - t.min()

    # define sampling frequencies
    fqs = np.linspace(1/T, nmodes/T, nmodes)

    # The sine/cosine modes
    ct = 0
    for ii in range(0, 2*nmodes-1, 2):
        
        F[:,ii] = np.cos(2*np.pi*fqs[ct]*t)
        F[:,ii+1] = np.sin(2*np.pi*fqs[ct]*t)
        ct += 1
    
    if freq:
        return F, fqs
    else:
        return F


##############################

# read in actual nanograv par and tim files
par_path = '/Users/staylor/Research/NANOGrav/NANOGrav_9y/par/'
ideal_par_path = '/Users/staylor/Research/PAL/ecc_grid_runs/ideal_nano_pars/'
tim_path = '/Users/staylor/Research/NANOGrav/NANOGrav_9y/tim/'

par_files = []
for file in sorted(glob.glob(par_path+"*.par")):
    par_files.append(file)

ideal_par_files = []
for file in sorted(glob.glob(ideal_par_path+"*.par")):
    ideal_par_files.append(file)
    
tim_files = []
for file in sorted(glob.glob(tim_path+"*.tim")):
    tim_files.append(file)

# delete tempo J1713 par file
if par_path+'J1713+0747_NANOGrav_9yv1.gls.par' in par_files:
    par_files.remove(par_path+'J1713+0747_NANOGrav_9yv1.gls.par')

# read the pulsars into libstempo
psr=[]
for ii in range(len(par_files)):
    psr.append(T.tempopulsar(parfile = par_files[ii], timfile = tim_files[ii]))

print "Checkpoint 1"
# form a dictionary of the parfiles
par_file_dict = OrderedDict.fromkeys([p.name for p in psr])
for ii,p in enumerate(psr):
    par_file_dict[p.name] = par_files[ii]

# form a dictionary of the parfiles
ideal_par_file_dict = OrderedDict.fromkeys([p.name for p in psr])
for ii,p in enumerate(psr):
    ideal_par_file_dict[p.name] = ideal_par_files[ii]

# get epochs of nanograv sampling
tsamp = OrderedDict.fromkeys([p.name for p in psr])
for p in psr:
    tmp, U = quantize_fast(p.toas(), dt=10.0/86400.0)
    tsamp[p.name] = tmp
    #print p.toas().shape, tmp.shape

##############################################

# define the 18 NANOGrav isotropic-limit pulsars as active
active_psrs = ['J1713+0747', 'J1909-3744', 'J1640+2224', 'J1600-3053', 'J2317+1439', 
               'J1918-0642', 'J1744-1134', 'J0030+0451', 'J0613-0200', 'J1614-2230',
                'B1855+09', 'J1853+1303', 'J2145-0750', 'J1455-3330', 'J1012+5307',
              'J1741+1351', 'J2010-1323', 'J1024-0719']

active_psrs.sort()

for ii in range(len(active_psrs)):
    active_psrs[ii] = re.sub('\\xe2\\x88\\x92', '-', active_psrs[ii])

active_psrs = OrderedDict.fromkeys(active_psrs)

# get libstempo objects from intersection of total with active
psr = [psr_dummy for psr_dummy in psr if psr_dummy.name in active_psrs]

# storing red-noise amplitude (nano style),
# gamma (nano style), white-noise level (microseconds)
# [from Arzoumanian et al. (2015a)]
active_psrs['B1855+09'] = [0.017,-4.9,0.505]
active_psrs['J0030+0451'] = [0.014,-4.8,0.212]
active_psrs['J0613-0200'] = [0.093,-2.9,0.165]
active_psrs['J1012+5307'] = [0.669,-1.0,0.355]
active_psrs['J1024-0719'] = [0.0,0.0,0.280]
active_psrs['J1455-3330'] = [0.0,0.0,0.694]
active_psrs['J1600-3053'] = [0.0,0.0,0.197]
active_psrs['J1614-2230'] = [0.0,0.0,0.189]
active_psrs['J1640+2224'] = [0.0,0.0,0.158]
active_psrs['J1713+0747'] = [0.0,0.0,0.116]
active_psrs['J1741+1351'] = [0.0,0.0,0.103]
active_psrs['J1744-1134'] = [0.0,0.0,0.334]
active_psrs['J1853+1303'] = [0.0,0.0,0.235]
active_psrs['J1909-3744'] = [0.0,0.0,0.080]
active_psrs['J1918-0642'] = [0.0,0.0,0.340]
active_psrs['J2010-1323'] = [0.0,0.0,0.312]
active_psrs['J2145-0750'] = [0.0,0.0,0.370]
active_psrs['J2317+1439'] = [0.0,0.0,0.267]

print "Checkpoint 2"
###################################################################################
###################################################################################
    
# Some random seeds for later noise injections

#seeds = [1234, 1952, 5298, 6723, 9887, 2481] # first will be a noiseless injection
#snrs = [4.0, 8.0, 12.0, 16.0, 20.0]
#orb_freqs = [1e-9, 5e-9, 1e-8, 5e-8]
#eccs = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

seeds = [1234] # first will be a noiseless injection
snrs = [4.0, 8.0, 12.0, 16.0, 20.0]
orb_freqs = [5e-9]
eccs = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

os.system('mkdir ecc_grid_runs') 
for ranseed in seeds:
    os.system('mkdir ecc_grid_runs/Seed_{0}'.format(ranseed))
    for rho in snrs:
        os.system('mkdir ecc_grid_runs/Seed_{0}/SNR_{1}'.format(ranseed,rho))
        for forb in orb_freqs:
            os.system('mkdir ecc_grid_runs/Seed_{0}/SNR_{1}/Forb_{2}'.format(ranseed,rho,forb))
            for e0 in eccs:
                os.system('mkdir ecc_grid_runs/Seed_{0}/SNR_{1}/Forb_{2}/Ecc_{3}'.format(ranseed,rho,forb,e0))
                
###################################################################################

for ranseed in seeds:
    for rho in snrs:
        for forb in orb_freqs:
            for e0 in eccs:

                # Define sampling and add noise
                psr_noise=[]
                psr_signal=[]
                for ii,p in enumerate(psr):
                    fake_obs = tsamp[p.name] 
                    
                    psr_noise.append(LT.fakepulsar(parfile=ideal_par_file_dict[p.name],
                                    obstimes = fake_obs,  
                                    toaerr = active_psrs[p.name][2]))
    
                    psr_signal.append(LT.fakepulsar(parfile=ideal_par_file_dict[p.name],
                                    obstimes = fake_obs,  
                                    toaerr = active_psrs[p.name][2]))
        

                tref_fake = np.min([p.toas().min() for p in psr_noise])*86400.0 # needs to be in seconds

                print "--> Performing SVD of design matrix to get G-matrix..."       
                Gcmat = []
                for ii,p in enumerate(psr_signal):
                    u,s,v = np.linalg.svd(p.designmatrix())
                    Gcmat.append(u[:,:len(s)])


                Ttot = np.min([p.toas().max() - p.toas().min() for p in psr_signal])
                nmodes = 50
                fqs = np.linspace(1/Ttot, nmodes/Ttot, nmodes)

                Fred = []
                Te = []
                for ii,p in enumerate(psr_signal):
                    Fred.append( createfourierdesignmatrix_RED(p.toas(), nmodes, Tspan=Ttot) )
                    Te.append( np.append(Gcmat[ii], Fred[ii], axis=1) )
                    print Te[ii].shape

                default_distance = 1.0 # Mpc
                for ii,p in enumerate(psr_signal):
                    LT.add_ecc_cgw(p, 2.17, 0.95, 1e9, default_distance, forb, 1.57, 1.26, 0.5, 
                                    e0, 0.99, 1.0, periEv=False, psrTerm=False, tref=tref_fake)


                d = []
                dtNdt = []
                TtNT = []
                for ii,p in enumerate(psr_signal):
                    errs = 1e-6 * p.toaerrs
                    d.append( np.dot(Te[ii].T, p.residuals()/( errs**2.0 )) )

                    N = 1./( errs**2.0 )
                    right = (N*Te[ii].T).T
                    TtNT.append( np.dot(Te[ii].T, right) )

                    # triple product in likelihood function
                    dtNdt.append( np.sum(p.residuals()**2.0/( errs**2.0 )) )


                snr_tot = 0.0
                for ii,p in enumerate(psr_signal):    
                    # parameterize intrinsic red noise as power law
                    Tspan = (1/fqs[0])*86400.0
                    f1yr = 1/3.16e7
                    nmode = len(fqs)

                    Ared = 1e-6 * active_psrs[p.name][0] * f1yr * np.sqrt(12.0*np.pi**2.0)
                    gam_red = -active_psrs[p.name][1]

                    if Ared == 0.0:
                        Ared = 1e-25

                    # parameterize intrinsic red-noise as power law
                    kappa = np.log10( Ared**2/12/np.pi**2 * f1yr**(gam_red-3) * (fqs/86400.0)**(-gam_red)/Tspan )

                    # construct elements of sigma array
                    diagonal = np.zeros(2*nmode)
                    diagonal[0::2] =  10**kappa
                    diagonal[1::2] = 10**kappa

                    # compute Phi inverse 
                    red_phi = np.diag(1./diagonal)

                    # now fill in real covariance matrix
                    Phi = np.zeros( TtNT[ii].shape ) 
                    for kk in range(0,2*nmode):
                        Phi[kk+Gcmat[ii].shape[1],kk+Gcmat[ii].shape[1]] = red_phi[kk,kk]

                    # symmeterize Phi
                    Phi = Phi + Phi.T - np.diag(np.diag(Phi))
    
                    # compute sigma
                    Sigma = TtNT[ii] + Phi

    
                    # cholesky decomp for second term in exponential
                    try:
                        cf = sl.cho_factor(Sigma)
                        expval2 = sl.cho_solve(cf, d[ii])
                        #logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

                    except np.linalg.LinAlgError:
                        print 'Cholesky Decomposition Failed second time!! Using SVD instead'
                        u,s,v = sl.svd(Sigma)
                        expval2 = np.dot(v.T, 1/s*np.dot(u.T, d[ii]))
                        #logdet_Sigma = np.sum(np.log(s))
        
                    snr_tot += dtNdt[ii] - np.dot(d[ii], expval2)


                snr_scaled_dist = default_distance * np.sqrt(snr_tot)/rho

                #############################################################
                # now inject an eccentric signal of desired snr in psr_noise
                #############################################################
                
                for ii,p in enumerate(psr_noise):
                    LT.add_ecc_cgw(p, 2.17, 0.95, 1e9, snr_scaled_dist, forb, 1.57, 1.26, 0.5, 
                                    e0, 0.99, 1.0, periEv=False, psrTerm=False, tref=tref_fake)

                #if ranseed != 1234:
                #    for ii,p in enumerate(psr_noise):
                #        LT.add_efac(p,efac=1.0,seed=ranseed+5*ii) ## need red noise here too!!!
                
                for ii,p in enumerate(psr_noise):
                    p.savepar('ecc_grid_runs/Seed_{0}/SNR_{1}/Forb_{2}/Ecc_{3}/{4}_sim.par'.format(ranseed,rho,forb,e0,p.name))
                    p.savetim('ecc_grid_runs/Seed_{0}/SNR_{1}/Forb_{2}/Ecc_{3}/{4}_sim.tim'.format(ranseed,rho,forb,e0,p.name))

                '''
                for ii,p in enumerate(psr_noise):
                    check_psr = T.tempopulsar(parfile = 'ecc_grid_runs/Seed_{0}/SNR_{1}/Forb_{2}/Ecc_{3}/{4}_sim.par'.format(ranseed,rho,forb,e0,p.name), timfile = 'ecc_grid_runs/Seed_{0}/SNR_{1}/Forb_{2}/Ecc_{3}/{4}_sim.tim'.format(ranseed,rho,forb,e0,p.name))
                    
                    for kk in range(10):
                        try:
                            print kk
                            check_psr.fit()
                        except:
                            "Pulsar {0} failed at iteration {1}".format(ii,kk)
                '''
                    
                print ranseed,rho,forb,e0
    



