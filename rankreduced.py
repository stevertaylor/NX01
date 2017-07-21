# rankreduced.py -- Example of rank-reduced covariance matrices for time-domain
#                   approaches
#
# Rutger van Haasteren, July 7 2014, Pasadena
#
# Requirements: matplotlib, numpy, scipy
#
# When using this code, please cite "van Haasteren & Vallisneri (2014)
#
# Modified by S. R. Taylor on 04/12/2017


from __future__ import division

import numpy as np
import scipy.linalg as sl
import scipy.interpolate as si
import scipy.fftpack as sf
import scipy.special as ss


# Power-law PSD
def pl_psd(f, Si, fL):
    """
    Return the power spectral density P(f) = (f / yr) ^ {-Si} * Theta(f - fL)
    @param f:   Frequency
    @param Si:  Spectral index
    @param fL:  Low-frequency cut-off
    """
    rv = f**(-Si)
    rv[f < fL] = 0.0
    return rv


# Analytic power-law covariance matrix for pl_psd
def pl_cov(t, Si=4.33, fL=1.0/20, approx_ksum=False):
    """
    Analytically calculate the covariance matrix for a stochastic signal with a
    power spectral density given by pl_psd.

    @param t:   Time-series timestamps
    @param Si:  Spectral index of power-law spectrum
    @param fL:  Low-frequency cut-off
    @param approx_ksum:     Whether we approximate the infinite sum

    @return:    Covariance matrix
    """

    EulerGamma = 0.5772156649015329
    alpha = 0.5*(3.0-Si)

    # Make a mesh-grid for the covariance matrix elements
    t1, t2 = np.meshgrid(t,t)
    x = 2 * np.pi * fL * np.abs(t1 - t2)
    del t1
    del t2

    # The tolerance for which to use the Gamma function expansion
    tol = 1e-5

    # the exact solutions for alpha = 0, -1 should be acceptable in a small
    # interval around them...
    if abs(alpha) < 1e-7:
        cosx, sinx = np.cos(x), np.sin(x)

        power = cosx - x * sinx
        sinint, cosint = sl.sici(x)

        corr = (fL**-2) / (24 * math.pi**2) * (power + x**2 * cosint)
    elif abs(alpha + 1) < 1e-7:
        cosx, sinx = np.cos(x), np.sin(x)

        power = 6 * cosx - 2 * x * sinx - x**2 * cosx + x**3 * sinx
        sinint, cosint = ss.sici(x)

        corr = (fL**-4) / (288 * np.pi**2) * (power - x**4 * cosint)
    else:
        # leading-order expansion of Gamma[-2+2*alpha]*Cos[Pi*alpha] around -0.5
        # and 0.5
        if   abs(alpha - 0.5) < tol:
            cf =  np.pi/2   + (np.pi - np.pi*EulerGamma) * (alpha - 0.5)
        elif abs(alpha + 0.5) < tol:
            cf = -np.pi/12  + (-11*np.pi/36 + EulerGamma*math.pi/6) * (alpha + 0.5)
        elif abs(alpha + 1.5) < tol:
            cf =  np.pi/240 + (137*np.pi/7200 - EulerGamma*np.pi/120) * (alpha + 1.5)
        else:
            cf = ss.gamma(-2+2*alpha) * np.cos(np.pi*alpha)

        power = cf * x**(2-2*alpha)

        # Mathematica solves Sum[(-1)^n x^(2 n)/((2 n)! (2 n + 2 alpha - 2)),
        # {n, 0, Infinity}] as HypergeometricPFQ[{-1+alpha}, {1/2,alpha},
        # -(x^2/4)]/(2 alpha - 2) the corresponding scipy.special function is
        # hyp1f2 (which returns value and error)
        if approx_ksum:
            ksum = 1.0 / (2*alpha - 2) - x**2 / (4*alpha) + x**4 / (24 * (2 + 2*alpha))
        else:
            ksum = ss.hyp1f2(alpha-1,0.5,alpha,-0.25*x**2)[0]/(2*alpha-2)

        del x

        corr = -(fL**(-2+2*alpha)) * (power + ksum)

    return corr




# Define the low-frequency Riemann and Simpson's binning
def linBinning(T, logmode, f_min, nlin, nlog):
    """
    Get the frequency binning for the low-rank approximations, including
    log-spaced low-frequency coverage.

    @param T:       Duration experiment
    @param logmode: From which linear mode to switch to log
    @param f_min:   Down to which frequency we'll sample
    @param nlin:    How many linear frequencies we'll use
    @param nlog:    How many log frequencies we'll use
    """
    if logmode < 0:
        raise ValueError("Cannot do log-spacing when all frequencies are linearly sampled")

    # First the linear spacing and weights
    df_lin = 1.0 / T
    f_min_lin = (1.0 + logmode) / T
    f_lin = np.linspace(f_min_lin, f_min_lin + (nlin-1)*df_lin, nlin)
    w_lin = np.sqrt(df_lin * np.ones(nlin))

    if nlog > 0:
        # Now the log-spacing, and weights
        f_min_log = np.log(f_min)
        f_max_log = np.log( (logmode+0.5)/T )
        df_log = (f_max_log - f_min_log) / (nlog)
        f_log = np.exp(np.linspace(f_min_log+0.5*df_log, f_max_log-0.5*df_log, nlog))
        w_log = np.sqrt(df_log * f_log)

        return np.append(f_log, f_lin), np.append(w_log, w_lin)

    else:
        return f_lin, w_lin



def simpsonBinning(T, logmode, f_min, nlin, nlog):
    """
    Get the frequency binning for the low-rank approximations, including
    log-spaced low-frequency coverage. Now do the integral with Simpson's rule,
    instead of a straightforward Riemann sum.

    @param T:       Duration experiment
    @param logmode: From which linear mode to switch to log
    @param f_min:   Down to which frequency we'll sample
    @param nlin:    How many linear frequencies we'll use
    @param nlog:    How many log frequencies we'll use
    """
    if logmode < 1:
        raise ValueError("Cannot do log-spacing when all frequencies are linearly sampled")

    if nlin%2==0 or nlog%2==0:
        raise ValueError("Simpson's rule requires nlin and nlog to be odd")

    # First the linear spacing and weights
    df_lin = 1.0 / T
    f_min_lin = (logmode) / T
    f_lin = np.linspace(f_min_lin, f_min_lin + (nlin-1)*df_lin, nlin)
    w_lin = np.sqrt(df_lin * np.ones(nlin))
    w_simp_lin = np.ones(nlin)
    w_simp_lin[2:-2:2] = 2.0
    w_simp_lin[1:-1:2] = 4.0
    w_simp_lin[0] = 1.0
    w_simp_lin[-1] = 1.0
    w_simp_lin[:] *= 1.0/3.0

    # Now the log-spacing, and weights
    f_min_log = np.log(f_min)
    f_max_log = np.log( (logmode)/T )
    df_log = (f_max_log - f_min_log) / (nlog-1)
    f_log = np.exp(np.linspace(f_min_log, f_max_log, nlog))
    w_log = np.sqrt(df_log * f_log)
    w_simp_log = np.ones(nlog)
    w_simp_log[2:-2:2] = 2.0
    w_simp_log[1:-1:2] = 4.0
    w_simp_log[0] = 1.0
    w_simp_log[-1] = 1.0
    w_simp_log[:] *= 1.0/3.0

    return np.append(f_log, f_lin), np.append(w_log*np.sqrt(w_simp_log), w_lin*np.sqrt(w_simp_lin))


def get_rr_rep(t, T, fmin, nlin, nlog, simpson=False):
    """
    Given some time-stamps, get a rank-reduced decomposition of the covariance
    matrix in terms of the PSD, consisting of: C = F Phi F^T, with Phi diagonal

    @param t:       Time-series timestamps
    @param T:       Duration of the time-series
    @param fmin:    Lowest frequency in RR representation
    @param nlin:    Number of linearly-spaced frequency bins
    @param nlog:    Number of log-spaced frequency bins
    @param simpson: If True, use Simpson's method

    @return:    frequencies, Fmat, Phi
    """
    # Get the binning for the rank-reduced approximation
    if simpson:
        freqs, weights = simpsonBinning(T, 1, f_min=fmin, nlin=nlin, nlog=nlog)
    else:
        freqs, weights = linBinning(T, 0, f_min=fmin, nlin=nlin, nlog=nlog)
    freqs_nd = np.array([freqs, freqs]).T.flatten()
    weights_nd = np.array([weights, weights]).T.flatten()

    # Set the F-matrix
    Fmat = np.zeros((len(t), len(freqs_nd)))
    for ii in range(0, len(freqs_nd), 2):
        omega = 2.0 * np.pi * freqs_nd[ii]
        Fmat[:,ii] = weights_nd[ii] * np.cos(omega * t)
        omega = 2.0 * np.pi * freqs_nd[ii+1]
        Fmat[:,ii+1] = weights_nd[ii+1] * np.sin(omega * t)

    return freqs_nd, Fmat


def get_rr_cov(Fmat, psd):
    """
    Given the rank-reduced approximation and the psd, return the covariance
    matrix

    @param Fmat:    The rank-reduced expansion matrix
    @param psd:     The PSD at the Fmat frequencies

    @return:    Covariance matix
    """
    return np.dot(Fmat * psd, Fmat.T)

def get_rr_covinv(N, Fmat, psd):
    """
    Given the rank-reduced approximation and a diagonal noise matrix, return the
    inverse covariance matrix (use Woodbury lemma)

    @param N:       The diagonal elements of the white-noise matrix
    @param Fmat:    The rank-reduced expansion matrix
    @param psd:     The PSD at the Fmat frequencies

    @return:    Inverse covariance matrix, log-det covariance matrix
    """
    Cov = np.diag(1.0 / N)

    FN = Fmat.T * (1.0 / N)

    Sigma = np.diag(1.0/psd) + np.dot(FN, Fmat)
    cf = sl.cho_factor(Sigma)

    logdet = np.sum(np.log(N)) + np.sum(np.log(psd)) + \
            2.0*np.sum(np.log(np.diag(cf[0])))

    A = Cov - np.dot(FN.T, sl.cho_solve(cf, FN))

    return Cov - np.dot(FN.T, sl.cho_solve(cf, FN)), logdet

def get_rr_cholesky_rep(N, Fmat, psd):
    """
    Use the Smola and Vishwanathan method to obtain a low-rank decomposition
    from which the Cholesky decomposition can be constructed. This is faster
    than using get_rr_cholesky, since the full Cholesky matrix does not need to
    be built

    @param N:       Vector with the elements of the diagonal matrix N
    @param Fmat:    (n x m) matrix consisting of the reduced rank basis
    @param psd:     PSD of the rank-reduced approximation

    @return: The Z-B-D decomposition for the Cholesky factor
    """
    Z = Fmat * np.sqrt(psd)
    m = Z.shape[1]
    n = N.shape[0]

    M = np.eye(m)
    B = np.zeros((n, m))
    D = np.zeros(n)

    # Create D and B
    for ii in range(n):
        t = np.dot(M, Z[ii, :])

        D[ii] = N[ii] + np.dot(Z[ii, :].T, t)
        if D[ii] > 0:
            B[ii, :] = t / D[ii]
            M = M - np.outer(t, t) / D[ii]          # Perhaps use the BLAS DSYRK here?

    # Construct B
    #BF = B.T * np.sqrt(D)

    return Z, B, D


def get_rr_cholesky(N, Fmat, psd):
    """
    Use the Smola and Vishwanathan method to obtain the lower-triangular
    Cholesky decomposition L of a matrix C = N + FF^{T} = LL^{T}.
    This is a fast version of lowRankUpdate_slow, assuming that N is diagonal.

    @param N:       Vector with the elements of the diagonal matrix N
    @param Fmat:    (n x m) matrix consisting of the reduced rank basis
    @param psd:     PSD of the rank-reduced approximation

    @return: The lower-triangular Cholesky decomposition of C
    """
    n = N.shape[0]
    Z, B, D = get_rr_cholesky_rep(N, Fmat, psd)

    # Construct L
    L = np.tril(np.dot(Z, B.T * np.sqrt(D)), -1)
    L[range(n), range(n)] = np.sqrt(D)

    return L

def get_rr_Lx(N, Fmat, psd, x):
    """
    Given a rank-reduced decomposition of the Cholesky factor L, calculate Lx
    where x is some vector. This way, we don't have to built L, which saves
    memory and computational time.

    @param N:       Vector with the elements of the diagonal matrix N
    @param Fmat:    (n x m) matrix consisting of the reduced rank basis
    @param psd:     PSD of the rank-reduced approximation
    @param x:       Vector we want to process as Lx

    @return Lx
    """
    n = N.shape[0]
    m = Fmat.shape[1]
    r = np.zeros(n)
    t = np.zeros(m)

    Z, B, D = get_rr_cholesky_rep(N, Fmat, psd)
    BD = (B.T * np.sqrt(D)).T

    for ii in range(n):
        r[ii] = x[ii]*np.sqrt(D[ii]) + np.dot(Z[ii,:].T, t)
        t += x[ii] * BD[ii,:]

    return r

def get_rr_Ux(N, Fmat, psd, x):
    """
    Given a rank-reduced decomposition of the Cholesky factor L, calculate L^{T}x
    where x is some vector. This way, we don't have to built L, which saves
    memory and computational time.

    @param N:       Vector with the elements of the diagonal matrix N
    @param Fmat:    (n x m) matrix consisting of the reduced rank basis
    @param psd:     PSD of the rank-reduced approximation
    @param x:       Vector we want to process as Lx

    @return Ux
    """
    n = N.shape[0]
    m = Fmat.shape[1]
    r = np.zeros(n)
    t = np.zeros(m)

    Z, B, D = get_rr_cholesky_rep(N, Fmat, psd)
    BD = (B.T * np.sqrt(D)).T

    for ii in range(n-1, -1, -1):
        r[ii] = x[ii]*np.sqrt(D[ii]) + np.dot(BD[ii,:].T, t)
        t += x[ii] * Z[ii,:]

    return r

def get_rr_Lix(N, Fmat, psd, x):
    """
    Given a rank-reduced decomposition of the Cholesky factor L, calculate
    L^{-1}x where x is some vector. This way, we don't have to built L, which
    saves memory and computational time.

    @param N:       Vector with the elements of the diagonal matrix N
    @param Fmat:    (n x m) matrix consisting of the reduced rank basis
    @param psd:     PSD of the rank-reduced approximation
    @param x:       Vector we want to process as Lx

    @return L^{-1}x
    """
    n = N.shape[0]
    m = Fmat.shape[1]
    y = np.zeros(n)
    t = np.zeros(m)

    Z, B, D = get_rr_cholesky_rep(N, Fmat, psd)
    BD = (B.T * np.sqrt(D)).T

    for ii in range(n):
        y[ii] = (x[ii] - np.dot(Z[ii,:], t)) / np.sqrt(D[ii])

        t = t + y[ii] * BD[ii,:]

    return y

def get_rr_Uix(N, Fmat, psd, x):
    """
    Given a rank-reduced decomposition of the Cholesky factor L, calculate
    L^{T,-1}x where x is some vector. This way, we don't have to built L^{T}, which
    saves memory and computational time.

    @param N:       Vector with the elements of the diagonal matrix N
    @param Fmat:    (n x m) matrix consisting of the reduced rank basis
    @param psd:     PSD of the rank-reduced approximation
    @param x:       Vector we want to process as Lx

    @return U^{-1}x
    """
    n = N.shape[0]
    m = Fmat.shape[1]
    y = np.zeros(n)
    t = np.zeros(m)

    Z, B, D = get_rr_cholesky_rep(N, Fmat, psd)

    BD = (B.T * np.sqrt(D)).T

    for ii in range(n-1, -1, -1):
        y[ii] = (x[ii] - np.dot(BD[ii,:], t)) / np.sqrt(D[ii])

        t = t + y[ii] * Z[ii,:]

    return y

def get_rr_Cx(N, Fmat, psd, x):
    """
    Given a rank-reduced decomposition, calculate Cx

    @param N:       Vector with the elements of the diagonal matrix N
    @param Fmat:    (n x m) matrix consisting of the reduced rank basis
    @param psd:     PSD of the rank-reduced approximation
    @param x:       Vector we want to process as Cx

    @return Cx
    """
    return np.dot(Fmat, psd*np.dot(Fmat.T, x)) + x*N

def get_rr_Cix(N, Fmat, psd, x):
    """
    Given a rank-reduced decomposition, calculate C^{-1}x

    @param N:       Vector with the elements of the diagonal matrix N
    @param Fmat:    (n x m) matrix consisting of the reduced rank basis
    @param psd:     PSD of the rank-reduced approximation
    @param x:       Vector we want to process as C^{-1}x

    @return C^{-1}x, logdet C
    """
    FN = Fmat.T * (1.0 / N)

    Sigma = np.diag(1.0/psd) + np.dot(FN, Fmat)
    cf = sl.cho_factor(Sigma)

    logdet = np.sum(np.log(N)) + np.sum(np.log(psd)) + \
            2.0*np.sum(np.log(np.diag(cf[0])))

    return x/N - np.dot(FN.T, sl.cho_solve(cf, np.dot(FN, x))), logdet


if __name__ == '__main__':
    # Little demo code

    # The time series (where we want the covariance matrix at)
    T = 10
    N = 1000
    fL  = 1.0 / (10.0*T)
    t = np.linspace(0, T, N, endpoint=False)
    err = 1.0e3
    W = np.ones(N) * err**2         # White noise

    # The signal parameters we're simulating
    Si = 4.33

    # The PSD at some pre-set frequencies (interpolation bins)
    Ni = 100000
    fi = np.linspace(fL, 10.0 * N/T, Ni)
    psd = pl_psd(fi, Si, fL)

    # The rank-reduced approximation is found with:
    fmin = 1.0 / (10.0*T)
    nlin = 21
    nlog = 21
    rrfreqs, Fmat = get_rr_rep(t, T, fmin, nlin, nlog, simpson=True)

    # Obtain the power-spectrum at the requested frequencies
    if True:
        # We have a function
        rrpsd = pl_psd(rrfreqs, Si, fL)
    else:
        # When we only have a numerical representation, interpolate like this
        # (Not done below)
        Ni = 1000000
        fi = np.linspace(fL, 10.0 * N/T, Ni)
        psd = pl_psd(fi, Si, fL)
        ifunc = si.interp1d(fi, psd, kind='linear')
        rrpsd = ifunc(rrfreqs)

    # The rank-reduced and the analytic covariance matrices
    Crr = get_rr_cov(Fmat, rrpsd)
    Can = pl_cov(t, Si=Si, fL=fL)
    Cov = np.diag(W) + Can

    # Get the inverse
    Crrinv, logdet = get_rr_covinv(W, Fmat, rrpsd)

    # Get the Cholesky decomposition
    CrrL = get_rr_cholesky(W, Fmat, rrpsd)

    # Tolerance at the 0.1% level
    atol = 1e-3
    tol = atol * np.mean(np.fabs(Can))
    print "Crr == Can:", np.allclose(Crr, Can, atol=tol)
    print "WCrr^-1 == Cov^-1:", np.allclose(np.eye(len(Crrinv)), \
            np.dot(Crrinv, Cov), atol=atol)
    print "LL^T == Cov:", np.allclose(np.dot(CrrL, CrrL.T), Cov, atol=tol)


    # Some random time-series
    x = np.random.randn(len(Can))
    y = x.copy()

    # Check the inversion
    Cx, logdet = get_rr_Cix(W, Fmat, rrpsd, x)
    CiCx = get_rr_Cx(W, Fmat, rrpsd, Cx)
    print "CiCx == x:", np.allclose(CiCx, x, atol=atol*np.mean(np.fabs(x)))

    # Figure out whether the O(n) Cholesky decompositions are ok
    # (should work for any y)
    #L = sl.cholesky(Cov, lower=True)    # Use L or CrrL ?
    Ly = get_rr_Lx(W, Fmat, rrpsd, y)
    Lya = np.dot(CrrL, y)
    print "Ly == Lya:", np.allclose(Ly, Lya, atol=atol*np.mean(np.fabs(Ly)))

    Lix = get_rr_Lix(W, Fmat, rrpsd, x)
    print "L Lix == x:", np.allclose(np.dot(CrrL, Lix), x, atol=atol*np.mean(np.fabs(x)))

    LTy = get_rr_Ux(W, Fmat, rrpsd, y)
    LTya = np.dot(CrrL.T, y)
    print "LTy == LTya:", np.allclose(LTy, LTya, atol=atol*np.mean(np.fabs(LTy)))

    LiTx = get_rr_Uix(W, Fmat, rrpsd, y)
    print "LT LiTx == x:", np.allclose(np.dot(CrrL.T, LiTx), x, atol=atol*np.mean(np.fabs(x)))
