
from __future__ import division
import numpy as np

class gaussproc(object):
    
    def __init__(self, x, y, yerr=None):
        
        self.x = x
        self.y = y
        self.yerr = yerr
        
        self.pmax = np.array([20.0, 50.0])
        self.pmin = np.array([-20.0, -20.0])
        self.chain = None
        self.lnprob = None
        self.kernel_map = None
    
    def lnprior(self, p):
    
        logp = 0.
    
        if np.all(p <= self.pmax) and np.all(p >= self.pmin):
            logp = np.sum(np.log(1/(self.pmax-self.pmin)))
        else:
            logp = -np.inf

        return logp

    def lnlike(self, p):

        # Update the kernel and compute the lnlikelihood.
        a, tau = np.exp(p[:2])
        
        lnlike = 0.0
        try:
            gp = george.GP(a * kernels.ExpSquaredKernel(tau))
            #gp = george.GP(a * kernels.Matern32Kernel(tau))
            if self.yerr is None:
                gp.compute(self.x)
            elif self.yerr is not None:
                gp.compute(self.x, self.yerr)
            
            lnlike = gp.lnlikelihood(self.y, quiet=True)
        except np.linalg.LinAlgError:
            lnlike = -np.inf
        
        return lnlike
    
    def lnprob(self, p):
        
        return self.lnprior(p) + self.lnlike(p)
