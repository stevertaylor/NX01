
import numpy as np
cimport numpy as np
from libc.math cimport log, sqrt


'''
ECORR/jitter routines from Rutger van Haasteren's "piccard" code.
'''

def cython_block_shermor_0D( \
        np.ndarray[np.double_t,ndim=1] r, \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)
    @param r:       The timing residuals, array (n)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)
    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, nir, nisum
    cdef np.ndarray[np.double_t,ndim=1] ni = np.empty(len(r), 'd')
    cdef np.ndarray[np.double_t,ndim=1] Nx = r / Nvec

    ni = 1.0 / Nvec

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nir = 0.0
            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                nir += r[ii]*ni[ii]

            beta = 1.0 / (nisum + ji)
            
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                Nx[ii] -= beta * nir * ni[ii]

    return Nx


def python_block_shermor_1D(r, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter
    @param r:       The timing residuals, array (n)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)
    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    ni = 1.0 / Nvec
    Jldet = np.einsum('i->', np.log(Nvec))
    xNx = np.dot(r, r * ni)

    for cc, jv in enumerate(Jvec):
        if jv > 0.0:
            rblock = r[Uinds[cc,0]:Uinds[cc,1]]
            niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

            beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
            xNx -= beta * np.dot(rblock, niblock)**2
            Jldet += np.log(jv) - np.log(beta)

    return Jldet, xNx

def cython_block_shermor_1D( \
        np.ndarray[np.double_t,ndim=1] r, \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)
    @param r:       The timing residuals, array (n)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)
    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(r), cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, xNx=0.0, nir, nisum
    cdef np.ndarray[np.double_t,ndim=1] ni = np.empty(rows, 'd')

    ni = 1.0 / Nvec

    for cc in range(rows):
        Jldet += log(Nvec[cc])
        xNx += r[cc]*r[cc]*ni[cc]

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nir = 0.0
            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                nir += r[ii]*ni[ii]

            beta = 1.0 / (nisum + ji)
            Jldet += log(Jvec[cc]) - log(beta)
            xNx -= beta * nir * nir
    
    return Jldet, xNx


# Proposals for calculating the Z.T * N^-1 * Z combinations
def python_block_shermor_2D(Z, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter, ZNiZ
    @param Z:       The design matrix, array (n x m)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)
    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    
    N = D + U*J*U.T
    calculate: log(det(N)), Z.T * N^-1 * Z
    """
    ni = 1.0 / Nvec
    Jldet = np.einsum('i->', np.log(Nvec))
    zNz = np.dot(Z.T*ni, Z)

    for cc, jv in enumerate(Jvec):
        if jv > 0.0:
            Zblock = Z[Uinds[cc,0]:Uinds[cc,1], :]
            niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

            beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
            zn = np.dot(niblock, Zblock)
            zNz -= beta * np.outer(zn.T, zn)
            Jldet += np.log(jv) - np.log(beta)

    return Jldet, zNz

def cython_block_shermor_2D( \
        np.ndarray[np.double_t,ndim=2] Z, \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.int_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)
    @param Z:       The design matrix, array (n x m)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param Uinds:   The start/finish indices for the jitter blocks (k x 2)
    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    N = D + U*J*U.T
    calculate: log(det(N)), Z.T * N^-1 * Z
    """
    cdef unsigned int cc, ii, rows = len(Nvec), cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, nir, nisum
    cdef np.ndarray[np.double_t,ndim=1] ni = np.empty(len(Nvec), 'd')
    cdef np.ndarray[np.double_t,ndim=2] zNz

    ni = 1.0 / Nvec
    zNz = np.dot(Z.T*ni, Z)

    for cc in range(rows):
        Jldet += log(Nvec[cc])

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            Zblock = Z[Uinds[cc,0]:Uinds[cc,1], :]
            niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

            nisum = 0.0
            for ii in range(len(niblock)):
                nisum += niblock[ii]

            beta = 1.0 / (nisum+1.0/Jvec[cc])
            Jldet += log(Jvec[cc]) - log(beta)
            zn = np.dot(niblock, Zblock)
            zNz -= beta * np.outer(zn.T, zn)

    return Jldet, zNz

