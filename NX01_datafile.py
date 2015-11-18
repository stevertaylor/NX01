#!/usr/bin/env python

"""

NX01_datafile.py

This file will use the libstempo library to add all relavant data
into an hdf5 file. 

This file was originally developed by Rutger van Haasteren, used by
Justin Ellis, and is recycled here.

"""

from __future__ import division

import numpy as np
import h5py as h5
import os, sys
import tempfile
import ephem
import os
import cPickle as pickle

import NX01_utils

try:
    # If without libstempo, can still read hdf5 files
    import libstempo
    t2 = libstempo
except ImportError:
    t2 = None

"""
The DataFile class is the class that supports the HDF5 file format. All HDF5
file interactions happen in this class.
"""
class DataFile(object):
    filename = None
    h5file = None

    """
    Initialise the structure.

    @param filename:    name of the HDF5 file
    """
    def __init__(self, filename=None):
        # Open the hdf5 file?
        self.filename = filename

    def __del__(self):
        # Delete the instance, and close the hdf5 file?
        pass

    """
    Return a list of pulsars present in the HDF5 file
    """
    def getPulsarList(self):
        # 'r' means: read file, must exist
        self.h5file = h5.File(self.filename, 'r')
        psrlist = list(self.h5file)
        self.h5file.close()

        return psrlist

    """
    Obtain the hdf5 group of pulsar psrname, create it if it does not exist. If
    delete is toggled, delete the content first. This function assumes the hdf5
    file is opened (not checked)

    @param psrname: The name of the pulsar
    @param delete:  If True, the pulsar group will be emptied before use
    """
    def getPulsarGroup(self, psrname, delete=False):
        # datagroup = h5file.require_group('Data')

        if psrname in self.h5file and delete:
            del self.h5file[psrname]

        pulsarGroup = self.h5file.require_group(psrname)

        return pulsarGroup
    

    """
    Add data to a specific pulsar. Here the hdf5 file is opened, and the right
    group is selected

    @param psrname:     The name of the pulsar
    @param field:       The name of the field we will be writing to
    @param data:        The data we are writing to the field
    @param overwrite:   Whether the data should be overwritten if it exists
    """
    def addData(self, psrname, field, data, overwrite=True):
        if self.filename is None:
            raise RuntimeError, "HDF5 filename not provided"

        # 'a' means: read/write if exists, create otherwise
        self.h5file = h5.File(self.filename, 'a')

        psrGroup = self.getPulsarGroup(psrname, delete=False)
        self.writeData(psrGroup, field, data, overwrite=overwrite)

        self.h5file.close()
        self.h5file = None

        

    """
    Read data from a specific pulsar. If the data is not available, the hdf5
    file is properly closed, and an exception is thrown

    @param psrname:     Name of the pulsar we are reading data from
    @param field:       Field name of the data we are requestion
    @param subgroup:    If the data is in a subgroup, get it from there
    @param dontread:    If set to true, do not actually read anything
    @param required:    If not required, do not throw an exception, but return
                        'None'
    """
    def getData(self, psrname, field, subgroup=None, \
            dontread=False, required=True):
        # Dontread is useful for readability in the 'readPulsarAuxiliaries
        if dontread:
            return None

        if self.filename is None:
            raise RuntimeError, "HDF5 filename not provided"

        # 'r' means: read file, must exist
        self.h5file = h5.File(self.filename, 'r')
        psrGroup = self.getPulsarGroup(psrname, delete=False)

        datGroup = psrGroup
        if subgroup is not None:
            if subgroup in psrGroup:
                datGroup = psrGroup[subgroup]
            else:
                self.h5file.close()
                if required:
                    raise IOError, "Field {0} not present for pulsar {1}/{2}".format(field, psrname, subgroup)

        if field in datGroup:
            if field == 'parfile' or field == 'timfile':
                data = datGroup[field].value
            else:
                data = np.array(datGroup[field])
            self.h5file.close()
        else:
            self.h5file.close()
            if required:
                raise IOError, "Field {0} not present for pulsar {1}".format(field, psrname)
            else:
                data = None

        return data

    """
    Retrieve the shape of a specific dataset

    @param psrname:     Name of the pulsar we are reading data from
    @param field:       Field name of the data we are requestion
    @param subgroup:    If the data is in a subgroup, get it from there
    """
    def getShape(self, psrname, field, subgroup=None):
        if self.filename is None:
            raise RuntimeError, "HDF5 filename not provided"

        # 'r' means: read file, must exist
        self.h5file = h5.File(self.filename, 'r')
        psrGroup = self.getPulsarGroup(psrname, delete=False)

        datGroup = psrGroup
        if subgroup is not None:
            if subgroup in psrGroup:
                datGroup = psrGroup[subgroup]
            else:
                self.h5file.close()
                raise IOError, "Field {0} not present for pulsar {1}/{2}".format(field, psrname, subgroup)

        if field in datGroup:
            shape = datGroup[field].shape
            self.h5file.close()
        else:
            self.h5file.close()
            raise IOError, "Field {0} not present for pulsar {1}".format(field, psrname)

        return shape


    """
    (Over)write a field of data for a specific pulsar/group. Data group is
    required, instead of a name.

    @param dataGroup:   Group object
    @param field:       Name of field that we are writing to
    @param data:        The data that needs to be written
    @param overwrite:   If True, data will be overwritten (default True)
    """
    def writeData(self, dataGroup, field, data, overwrite=True):
        if field in dataGroup and overwrite:
            del dataGroup[field]

        if not field in dataGroup:
            dataGroup.create_dataset(field, data=data)

    """
    Add a pulsar to the HDF5 file, given a tempo2 par and tim file. No extra
    model matrices and auxiliary variables are added to the HDF5 file. This
    function interacts with the libstempo Python interface to Tempo2

    @param parfile:     Name of tempo2 parfile
    @param timfile:     Name of tempo2 timfile
    @param iterations:  Number of fitting iterations to do before writing
    @param mode:        Can be replace/overwrite/new. Replace first deletes the
                        entire pulsar group. Overwrite overwrites all data, but
                        does not delete the auxiliary fields. New requires the
                        pulsar not to exist, and throws an exception otherwise.
    """
    def addTempoPulsar(self, psr, mode='replace'):
        # Check whether the two files exist
        if not os.path.isfile(psr.parfile) or not os.path.isfile(psr.timfile):
            raise IOError, "Cannot find parfile (%s) or timfile (%s)!" % (parfile, timfile)

        if self.filename is None:
            raise RuntimeError, "HDF5 filename not provided"

        # Parse the default write behaviour
        deletepsr = False
        if mode == 'replace':
            deletepsr = True
        overwrite = False
        if mode == 'overwrite':
            overwrite = True

        # 'a' means: read/write if exists, create otherwise
        self.h5file = h5.File(self.filename, 'a')
        
        # Obtain the directory name of the timfile, and change to it
        timfiletup = os.path.split(psr.timfile)
        dirname = timfiletup[0]
        reltimfile = timfiletup[-1]
        relparfile = os.path.relpath(psr.parfile, dirname)
        savedir = os.getcwd()

        # Change directory to the base directory of the tim-file to deal with
        # INCLUDE statements in the tim-file
        os.chdir(dirname)

        # Load the entire par-file into memory, so that we can save it in the
        # HDF5 file
        with open(relparfile, 'r') as content_file:
            parfile_content = content_file.read()

        # Save the tim-file to a temporary file (so that we don't have to deal
        # with 'include' statements in the tim-file), and load that tim-file in
        # memory for HDF5 storage
        tempfilename = tempfile.mktemp()
        psr.T2psr.savetim(tempfilename)
        with open(tempfilename, 'r') as content_file:
            timfile_content = content_file.read()
        os.remove(tempfilename)

        # Change directory back to where we were
        os.chdir(savedir)

        # Get the pulsar group
        psrGroup = self.getPulsarGroup(str(psr.name), delete=deletepsr)

        # Save the par-file and the tim-file to the HDF5 file
        self.writeData(psrGroup, 'parfile', parfile_content, overwrite=overwrite)
        self.writeData(psrGroup, 'timfile', timfile_content, overwrite=overwrite)

        # Save the path to the parfile and timfile
        self.writeData(psrGroup, 'parfilepath', psr.parfile, overwrite=overwrite)
        self.writeData(psrGroup, 'timfilepath', psr.timfile, overwrite=overwrite)

        # Load the entire noise-file into memory, so that we can save it in the
        # HDF5 file
        with open(psr.noisefile, 'r') as content_file:
            noisefile_content = content_file.read()
        # Save the noise-file and path to the HDF5 file
        self.writeData(psrGroup, 'noisefilepath', psr.noisefile, overwrite=overwrite)
        self.writeData(psrGroup, 'noisefile', noisefile_content, overwrite=overwrite)

        # Save the basic quantities
        self.writeData(psrGroup, 'name', psr.name,
                       overwrite=overwrite)    # Days
        self.writeData(psrGroup, 'TOAs', psr.toas,
                       overwrite=overwrite)    # Days
        self.writeData(psrGroup, 'postfitRes', psr.res,
                       overwrite=overwrite)  # Seconds
        self.writeData(psrGroup, 'toaErr', psr.toaerrs,
                       overwrite=overwrite)    # Seconds
        self.writeData(psrGroup, 'freq', psr.obs_freqs,
                       overwrite=overwrite)    # MHz

        # Save some useful matrices
        self.writeData(psrGroup, 'designmatrix', psr.Mmat,
                       overwrite=overwrite)
        if psr.G is not None:
            self.writeData(psrGroup, 'Gmatrix', psr.G,
                           overwrite=overwrite)
            self.writeData(psrGroup, 'Gres', psr.Gres,
                           overwrite=overwrite)
        self.writeData(psrGroup, 'GCmatrix', psr.Gc,
                       overwrite=overwrite)
        
        self.writeData(psrGroup, 'QuantMat', psr.Umat,
                       overwrite=overwrite)
        self.writeData(psrGroup, 'QuantInds', psr.Uinds,
                       overwrite=overwrite)
        self.writeData(psrGroup, 'EpochFlags', psr.epflags,
                       overwrite=overwrite)
        self.writeData(psrGroup, 'DetSigAveToas', psr.detsig_avetoas,
                       overwrite=overwrite)
        self.writeData(psrGroup, 'DetSigQuantInds', psr.detsig_Uinds,
                       overwrite=overwrite)

        # pickle and store system flag dictionary
        storeFlagDict = pickle.dumps(psr.sysflagdict)
        self.writeData(psrGroup, 'SysFlagDict', storeFlagDict,
                       overwrite=overwrite)

        # Save the pulsar locations
        self.writeData(psrGroup, 'psrlocs', psr.psr_locs,
                       overwrite=overwrite)
        
        
        # get pulsar distance and uncertainty (need pulsarDistances.txt file for this)
        fin = open('pulsarDistances.txt', 'r')
        lines = fin.readlines()
        found = 0
        for line in lines:
            vals = line.split()
            if psr.name in vals[0]:
                pdist, pdistErr = np.double(vals[1]), np.double(vals[2])
                found = True
        if not(found):
            print ('WARNING: Could not find pulsar distance for PSR {0}. '
                   'Setting value to 1 with 20% uncertainty'.format(psr.name))
            pdist, pdistErr = 1.0, 0.2

        # close file
        fin.close()
        
        # write to file
        self.writeData(psrGroup, 'pdist', pdist, overwrite=overwrite)
        self.writeData(psrGroup, 'pdistErr', pdistErr, overwrite=overwrite)
        
        
        # Now obtain and write the timing model parameters
        tmpname = ['Offset'] + list(map(str,psr.T2psr.pars()))
        tmpvalpost = np.zeros(len(tmpname))
        tmperrpost = np.zeros(len(tmpname))
        for i in range(len(psr.T2psr.pars())):
            tmpvalpost[i+1] = psr.T2psr[tmpname[i+1]].val
            tmperrpost[i+1] = psr.T2psr[tmpname[i+1]].err

        self.writeData(psrGroup, 'tmp_name', tmpname, 
                       overwrite=overwrite) # TMP name
        self.writeData(psrGroup, 'tmp_valpost', tmpvalpost,
                       overwrite=overwrite) # TMP post-fit value
        self.writeData(psrGroup, 'tmp_errpost', tmperrpost,
                       overwrite=overwrite) # TMP post-fit error


        
        # Close the HDF5 file
        self.h5file.close()
        self.h5file = None

   
