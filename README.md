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

## Getting things installed (from scratch)

1. Make sure you have a good python installation with a comprehensive
   collection of popular modules. I suggest
   [Anaconda](https://www.continuum.io/downloads), which has a large
   set of modules that we need. If you don't want to mess with your
   existing python installation, then I suggest you proceed with the
   following steps, resolving module errors as they are encountered by
   either finding the relevant module git repository or trying `pip
   install [...]`.
2. Download a recent version of
   [tempo2](https://bitbucket.org/psrsoft/tempo2). The download
   website has quite a good installation guide. During the
   installation process, copy T2runtime to a
   location of your choice. When you set the TEMPO2 environment
   variable, make sure to also add it to your bashrc file, and reload
   your bashrc file (`source ~/.bashrc`) to reflect the environment
   change.
3. After installing `tempo2`, you may also need to add the `tempo2`
   libraries to your library path. Do so by adding `export
   LD_LIBRARY_PATH=$TEMPO2/lib` to your bashrc file, and reload the
   bashrc file to reflect your changes.
4. Install [libstempo](https://github.com/vallis/libstempo), which is
   our python interface to tempo2. You can clone the repository (`git
   clone https://github.com/vallis/libstempo.git`) to wherever you
   want to store software on your machine, and install with `python
   setup.py install --with-tempo2=$TEMPO2`. If needed, more complete
   instructions can be found at `http://vallis.github.io/libstempo/`.
5. For NANOGrav analyses, download the
   [dataset](http://data.nanograv.org/) and follow the instructions in
   README.clock to update your tempo2 clock files. Note that the
   instructions tell you to copy `gbt2gps.dat` but the file is
   actually `gbt2gps.clk`.
6. Install [PTMCMCSampler](https://github.com/jellis18/PTMCMCSampler),
   which is the main sampler we use to
   explore the signal and noise parameter space. As before, clone with
   `git clone https://github.com/jellis18/PTMCMCSampler.git` and
   install with `python setup.py install`.
7. Install the python `ephem` package by executing `conda install
ephem`.
8. Download [NX01](https://github.com/stevertaylor/NX01.git) by
   cloning the repository -- `git clone
   https://github.com/stevertaylor/NX01.git`.
9. Copy `nanograv-pulsar-store.ipynb` to a new notebook for your own
   specific use. Change all paths to reflect where you have NANOGrav
   par and tim data. The end result should produce a directory
   containing `hdf5` files for all NANOGrav pulsars in the dataset.
10. Copy `PsrListings_GWB.txt` to a new file for your own specific
    use. Edit the paths to the `hdf5` files, `par` files, and `tim`
    files to reflect where you have stored them on your machine.
11. Execute `python NX01_master.py --help` to see all of the options
    available to you. These are quite extensive, and are being
    actively updated and improved.
12. To run an analysis which will recover the upper limit on the
    dimensionless strain amplitude which matches that reported in the
    [NANOGrav 9-year limit paper](http://arxiv.org/abs/1508.03024),
    execute `python NX01_master.py --from-h5
    --psrlist=./PsrListings_GWB.txt --nmodes=15 --incGWB --fix-slope
    --psrEndIndex=18 --dirExt=./chains_firsttests/`.
13. This will initiate a run, producing a parent directory
    `chains_firsttests`, and run-specific sub-directory.  This
    sub-directory will contain MCMC sample files and many auxillary
    files, amongst which is `parameter_list.txt`. This contains two
    columns; the first is a list of indices corresponding to
    columns of `chain_1.txt` where one can find the parameters of the
    second column.
14. After identifying the column of `chain_1.txt` which corresponds to
    `Agwb`, you can get an $x\%$ upper limit by executing
    the following:
```
import NX01_bayesutils as bu
upper_lim = bu.confinterval(Agwb_samples, sigma=x/100, onesided=True)
```

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
--incGWB --fix-slope --psrEndIndex=18
```
which will perform a GW background upper-limit analysis (without
correlations...to include correlations add `--incCorr`) with PTMCMC on the 18 pulsars analyzed in the 9-year NANOGrav limit paper.

As in the single-pulsar analysis case, you can use MPI for both PTMCMC
and MultiNest samplers.
