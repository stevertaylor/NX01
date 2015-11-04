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

It is recommended that you initially open and follow the steps in the
`nanograv_pulsar_store.ipynb` notebook, and produce your own
`PsrListings_GWB.txt` and `PsrListings_CW.txt`.

## Single-pulsar noise analysis

It should be straightforward to perform a single-pulsar noise analysis
out of the box. One thing to note is to make sure you have
correctly updated your tempo2 clock files with files packaged with
the NANOGrav data download.

Run the following:
```
python NX01_singlePsr.py --help
```

for a list of all options.

An example run command would be:
```
python NX01_singlePsr.py
--parfile=./NANOGrav_9y/par/J1713+0747_NANOGrav_9yv1.t2.gls.strip.par
--timfile=./NANOGrav_9y/tim/J1713+0747_NANOGrav_9yv1.tim
--efacequad-sysflag=f --fullN --ptmcmc
```
