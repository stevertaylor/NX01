NX01
====

Python code to perform isotropic, anisotropic (non-evolving/evolving)
searches for gravitational-waves via pulsar-timing.

Perhaps you want the [webpage](http://stevertaylor.github.io/NX01/)?

## Code list

* **NX01_master.py**: performs a full evolving-anisotropy GWB and
  noise analysis. Uses MultiNest or parallel-tempering sampling.
* **NX01_singlePsr.py**: performs a stochastic
  search for single-pulsar noise parameters within the reduced-rank
  time-frequency approximation. Uses MultiNest or parallel-tempering sampling.
* **NX01_AnisCoefficients.py**: utility file to create power-anisotropy
  basis-functions.
* **NX01_utils.py**: utility file.
* **NX01_plot.py**: plotting package, adapted and extended from PAL.
* **NX01_psr.py**: utility file which defines the pulsar class for
storing all relevant variables.
* **NX01_datafile.py**: creates an hdf5 container to store all the
information in the pulsar class. Useful for storing array products.
* **NX01_jitter.pxy**: cython code to perform Sherman-Morrison
  block noise-matrix inversions when handling ECORR (jitter). 
* **NX01_bayesutils.py**: utilities file for generating plotting data.

## Preliminaries

One important first thing to note is to make sure you have
correctly updated your tempo2 clock files with the corresponding files packaged with
the NANOGrav data download.

It is recommended that you initially open and follow the steps in the
`nanograv-pulsar-store.ipynb` notebook, and produce your own
`PsrListings_GWB.txt` and `PsrListings_CW.txt`. These latter files are
lists of pulsars in the order with which they contribute to the
stochastic background upper limit (`_GWB.txt`) and the single-source
SNR (`_CW.txt`). Each pulsar is associated with a path to an hdf5 file
(storing all pulsar properties), parfile, and timfile.

By following the steps in the notebook (only up until the
cross-validation plotting between NX01 and PAL2) you will produce your
own hdf5 files, which you should put in a directory of your
choice. These hdf5 files will store everything you need for subsequent
GW searches. Another important step in the notebook is to produce par
files which are stripped of tempo2 EFAC, EQUAD, ECORR, RedAmp, and
RedInd values. These values are replaced by mean values from Justin's
previous single-pulsar analyses.

## Single-pulsar noise analysis

It should be straightforward to perform a single-pulsar noise analysis
out of the box. 

Run `python NX01_singlePsr.py --help` for a list of all options.

An example run command would be:
```
python NX01_singlePsr.py
--parfile=./NANOGrav_9y/par/J1713+0747_NANOGrav_9yv1.t2.gls.strip.par
--timfile=./NANOGrav_9y/tim/J1713+0747_NANOGrav_9yv1.tim
--efacequad-sysflag=f --fullN --ptmcmc
```

Without the `--ptmcmc` option, the sampler will default to MultiNest.

If you have MPI installed you can parallelise by running the
following:
```
mpirun -np 4 NX01_singlePsr.py
--parfile=./NANOGrav_9y/par/J1713+0747_NANOGrav_9yv1.t2.gls.strip.par
--timfile=./NANOGrav_9y/tim/J1713+0747_NANOGrav_9yv1.tim
--efacequad-sysflag=f --fullN --ptmcmc
```
where 4 cores will produce 4 temperature chains in the
parallel-tempering MCMC sampling process. Without the `--ptmcmc`
command, 4 cores would have been used to update the live points in MultiNest.


## Gravitational-wave searches

It is recommended to read in pulsars from their respective hdf5 files,
which you should have previously produced from the
`nanograv-pulsar-store.ipynb` notebook.

Run `python NX01_master.py --help` for a list of all options.

An example run command would be:
```
python NX01_master.py --from-h5
--psrlist=./PsrListings_GWB.txt --nmodes=15
--incGWB --fix-slope --num_psrs=18 --fullN
```
which will perform a GW background upper-limit analysis (without
correlations...to include correlations add `--incCorr`) on the 18 pulsars analyzed in the 9-year NANOGrav limit paper.

As in the single-pulsar analysis case, you can use MPI for the PTMCMC,
however MultiNest functionality is not yet ready.
