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
